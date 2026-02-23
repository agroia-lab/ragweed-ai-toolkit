#!/usr/bin/env python3
"""
Build Combined AMBEL Dataset (Chilean + International)
======================================================

Merges three AMBEL sources into a single YOLO dataset with source-stratified
train/valid/test splits:

  1. CL_Seba     (Chilean, Roboflow augmented)  - preserve existing splits
  2. CL_Alberto  (Chilean, Roboflow)            - preserve existing splits
  3. International (4 databases, flat)           - create 80/10/10 splits

All sources are single-class (class 0 = AMBEL).  Files are copied with
source-specific prefixes to prevent name collisions.

Usage:
    # Build full dataset
    python scripts/cli/build_combined_dataset.py

    # Dry run (count and report only, no file copying)
    python scripts/cli/build_combined_dataset.py --dry-run

    # Build and validate random labels
    python scripts/cli/build_combined_dataset.py --validate

    # Custom international split ratios
    python scripts/cli/build_combined_dataset.py --intl-split 0.7,0.15,0.15
"""

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm


# ============================================================================
# ROBOFLOW SOURCE HANDLER (CL_Seba, CL_Alberto)
# ============================================================================


def copy_roboflow_source(
    source_dir: Path,
    prefix: str,
    output_dir: Path,
    dry_run: bool,
) -> Dict[str, List[Dict]]:
    """
    Copy a Roboflow-structured dataset (train/valid/test) into output splits.

    Preserves the original Roboflow train/valid/test splits.  Each file is
    prefixed with ``prefix_`` to prevent name collisions.

    Args:
        source_dir: Root of the Roboflow dataset (contains train/, valid/, test/).
        prefix: Filename prefix (e.g. "CL_Seba").
        output_dir: Root of the combined output dataset.
        dry_run: If True, count only without copying files.

    Returns:
        Dict mapping split name to list of manifest row dicts.
    """
    splits = {"train": "train", "valid": "valid", "test": "test"}
    result: Dict[str, List[Dict]] = {"train": [], "valid": [], "test": []}

    for split_name, split_dir_name in splits.items():
        src_images = source_dir / split_dir_name / "images"
        src_labels = source_dir / split_dir_name / "labels"

        if not src_images.exists():
            print(f"    [!] Missing {src_images}")
            continue

        dst_images = output_dir / split_name / "images"
        dst_labels = output_dir / split_name / "labels"

        if not dry_run:
            dst_images.mkdir(parents=True, exist_ok=True)
            dst_labels.mkdir(parents=True, exist_ok=True)

        image_files = sorted(src_images.glob("*.jpg"), key=lambda p: p.name)

        for img_path in tqdm(
            image_files, desc=f"    {prefix}/{split_name}", leave=False
        ):
            stem = img_path.stem
            out_img_name = f"{prefix}_{img_path.name}"
            out_lbl_name = f"{prefix}_{stem}.txt"

            label_path = src_labels / f"{stem}.txt"
            n_annots = 0

            if label_path.exists():
                label_text = label_path.read_text()
                n_annots = sum(
                    1 for line in label_text.strip().splitlines() if line.strip()
                )
                if not dry_run:
                    shutil.copy2(img_path, dst_images / out_img_name)
                    (dst_labels / out_lbl_name).write_text(label_text)
            else:
                # Image without label -- copy image, create empty label
                if not dry_run:
                    shutil.copy2(img_path, dst_images / out_img_name)
                    (dst_labels / out_lbl_name).write_text("")

            result[split_name].append({
                "filename": out_img_name,
                "source": prefix,
                "source_db": prefix,
                "split": split_name,
                "original_path": str(img_path),
                "n_annotations": n_annots,
            })

    return result


# ============================================================================
# INTERNATIONAL SOURCE HANDLER (stratified splitting)
# ============================================================================


def split_international_source(
    intl_dir: Path,
    output_dir: Path,
    split_ratios: Tuple[float, float, float],
    seed: int,
    dry_run: bool,
) -> Dict[str, List[Dict]]:
    """
    Split the international dataset into train/valid/test, stratified by source_db.

    Reads manifest.csv to determine which source database each image belongs to,
    then randomly splits each database group according to ``split_ratios``.

    Args:
        intl_dir: Root of the international dataset (contains images/, labels/, manifest.csv).
        output_dir: Root of the combined output dataset.
        split_ratios: Tuple of (train, valid, test) ratios, must sum to 1.0.
        seed: Random seed for reproducibility.
        dry_run: If True, count only without copying files.

    Returns:
        Dict mapping split name to list of manifest row dicts.
    """
    manifest_path = intl_dir / "manifest.csv"
    if not manifest_path.exists():
        print(f"    [!] manifest.csv not found at {manifest_path}")
        return {"train": [], "valid": [], "test": []}

    # Read manifest and group by source_db
    groups: Dict[str, List[Dict]] = defaultdict(list)
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            groups[row["source_db"]].append(row)

    print(f"    [+] International source databases:")
    for db_name, rows in sorted(groups.items()):
        print(f"        {db_name}: {len(rows)} images")

    # Stratified split: for each source_db, split independently
    rng = random.Random(seed)
    split_names = ["train", "valid", "test"]
    result: Dict[str, List[Dict]] = {"train": [], "valid": [], "test": []}

    for db_name in sorted(groups.keys()):
        rows = groups[db_name]
        rng.shuffle(rows)

        n = len(rows)
        n_train = round(n * split_ratios[0])
        n_valid = round(n * split_ratios[1])
        # test gets the remainder to avoid rounding loss
        boundaries = [
            (0, n_train, "train"),
            (n_train, n_train + n_valid, "valid"),
            (n_train + n_valid, n, "test"),
        ]

        for start, end, split_name in boundaries:
            for row in rows[start:end]:
                result[split_name].append({
                    **row,
                    "source": "International",
                    "split": split_name,
                })

    # Copy files to output splits
    for split_name in split_names:
        dst_images = output_dir / split_name / "images"
        dst_labels = output_dir / split_name / "labels"

        if not dry_run:
            dst_images.mkdir(parents=True, exist_ok=True)
            dst_labels.mkdir(parents=True, exist_ok=True)

        for row in tqdm(
            result[split_name],
            desc=f"    International/{split_name}",
            leave=False,
        ):
            filename = row["filename"]
            stem = Path(filename).stem

            src_img = intl_dir / "images" / filename
            src_lbl = intl_dir / "labels" / f"{stem}.txt"

            if not src_img.exists():
                continue

            if not dry_run:
                shutil.copy2(src_img, dst_images / filename)
                if src_lbl.exists():
                    shutil.copy2(src_lbl, dst_labels / f"{stem}.txt")

    return result


# ============================================================================
# VALIDATION
# ============================================================================


def validate_random_labels(output_dir: Path, n: int = 5) -> None:
    """
    Spot-check random label files from each split.

    Verifies:
    - All lines have exactly 5 fields
    - Class is always 0
    - Coordinates are in [0, 1]

    Args:
        output_dir: Root of the combined output dataset.
        n: Number of labels to check per split.
    """
    print(f"\n[+] Validating {n} random labels per split...")
    all_ok = True

    for split in ["train", "valid", "test"]:
        label_dir = output_dir / split / "labels"
        if not label_dir.exists():
            continue

        label_files = list(label_dir.glob("*.txt"))
        if not label_files:
            continue

        sample = random.sample(label_files, min(n, len(label_files)))

        for lf in sample:
            issues = []
            text = lf.read_text().strip()
            if not text:
                continue  # empty label is fine (background image)

            for i, line in enumerate(text.splitlines(), 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    issues.append(f"  line {i}: expected 5 fields, got {len(parts)}")
                    continue
                cls = int(parts[0])
                if cls != 0:
                    issues.append(f"  line {i}: class={cls} (expected 0)")
                coords = [float(x) for x in parts[1:]]
                for j, v in enumerate(coords):
                    if v < 0 or v > 1:
                        issues.append(f"  line {i}: coord[{j}]={v:.4f} out of [0,1]")

            status = "OK" if not issues else "ISSUES"
            print(f"  [{split}] {lf.name}: {status}")
            if issues:
                all_ok = False
                for issue in issues:
                    print(f"    {issue}")

    if all_ok:
        print("[+] All validated labels OK")
    else:
        print("[!] Some labels have issues - review above")


# ============================================================================
# MAIN BUILD
# ============================================================================


def build_dataset(args) -> None:
    """
    Main dataset build function.

    Processes CL_Seba, CL_Alberto (preserve Roboflow splits), and International
    (create stratified splits), then writes manifest.csv and build_summary.json.

    Args:
        args: Parsed command-line arguments.
    """
    cl_seba_dir = Path(args.cl_seba)
    cl_alberto_dir = Path(args.cl_alberto)
    intl_dir = Path(args.international)
    output_dir = Path(args.output)

    # Parse international split ratios
    try:
        ratios = tuple(float(x) for x in args.intl_split.split(","))
        assert len(ratios) == 3
        assert abs(sum(ratios) - 1.0) < 0.01
    except (ValueError, AssertionError):
        print(f"[-] Invalid --intl-split: {args.intl_split}")
        print("    Expected format: 0.8,0.1,0.1 (must sum to 1.0)")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Banner
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  BUILD COMBINED AMBEL DATASET (Chilean + International)")
    if args.dry_run:
        print("  *** DRY RUN - no files will be copied ***")
    print("=" * 70)
    print(f"  CL_Seba:        {cl_seba_dir}")
    print(f"  CL_Alberto:     {cl_alberto_dir}")
    print(f"  International:  {intl_dir}")
    print(f"  Output:         {output_dir}")
    print(f"  Intl split:     {ratios[0]:.0%} / {ratios[1]:.0%} / {ratios[2]:.0%}")
    print(f"  Seed:           {args.seed}")
    print("=" * 70)
    print()

    # Validate source directories exist
    missing = []
    for name, path in [
        ("CL_Seba", cl_seba_dir),
        ("CL_Alberto", cl_alberto_dir),
        ("International", intl_dir),
    ]:
        if not path.exists():
            missing.append(f"  {name}: {path}")
    if missing:
        print("[-] Missing source directories:")
        for m in missing:
            print(m)
        sys.exit(1)

    # Create output directory structure
    if not args.dry_run:
        for split in ["train", "valid", "test"]:
            (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Process CL_Seba
    # -----------------------------------------------------------------------
    print("-" * 70)
    print("  [1/3] CL_Seba (Chilean, Roboflow augmented)")
    print("-" * 70)
    seba_result = copy_roboflow_source(
        cl_seba_dir, "CL_Seba", output_dir, args.dry_run
    )
    seba_counts = {s: len(rows) for s, rows in seba_result.items()}
    seba_total = sum(seba_counts.values())
    print(f"  [+] CL_Seba: {seba_total} images "
          f"(train={seba_counts['train']}, "
          f"valid={seba_counts['valid']}, "
          f"test={seba_counts['test']})")
    print()

    # -----------------------------------------------------------------------
    # Process CL_Alberto
    # -----------------------------------------------------------------------
    print("-" * 70)
    print("  [2/3] CL_Alberto (Chilean, Roboflow)")
    print("-" * 70)
    alberto_result = copy_roboflow_source(
        cl_alberto_dir, "CL_Alberto", output_dir, args.dry_run
    )
    alberto_counts = {s: len(rows) for s, rows in alberto_result.items()}
    alberto_total = sum(alberto_counts.values())
    print(f"  [+] CL_Alberto: {alberto_total} images "
          f"(train={alberto_counts['train']}, "
          f"valid={alberto_counts['valid']}, "
          f"test={alberto_counts['test']})")
    print()

    # -----------------------------------------------------------------------
    # Process International (stratified split)
    # -----------------------------------------------------------------------
    print("-" * 70)
    print("  [3/3] International (4 databases, stratified split)")
    print("-" * 70)
    intl_result = split_international_source(
        intl_dir, output_dir, ratios, args.seed, args.dry_run
    )
    intl_counts = {s: len(rows) for s, rows in intl_result.items()}
    intl_total = sum(intl_counts.values())
    print(f"  [+] International: {intl_total} images "
          f"(train={intl_counts['train']}, "
          f"valid={intl_counts['valid']}, "
          f"test={intl_counts['test']})")
    print()

    # -----------------------------------------------------------------------
    # Merge all manifest rows
    # -----------------------------------------------------------------------
    all_rows: List[Dict] = []
    for split_result in [seba_result, alberto_result, intl_result]:
        for split_name, rows in split_result.items():
            all_rows.extend(rows)

    # -----------------------------------------------------------------------
    # Write manifest.csv
    # -----------------------------------------------------------------------
    manifest_path = output_dir / "manifest.csv"
    if not args.dry_run:
        fieldnames = [
            "filename", "source", "source_db", "split",
            "original_path", "n_annotations",
        ]
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    # -----------------------------------------------------------------------
    # Write build_summary.json
    # -----------------------------------------------------------------------
    # Per-source-db breakdown for international
    intl_db_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for split_name, rows in intl_result.items():
        for row in rows:
            intl_db_counts[row["source_db"]][split_name] += 1

    summary = {
        "timestamp": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "dry_run": args.dry_run,
        "seed": args.seed,
        "intl_split_ratios": list(ratios),
        "sources": {
            "CL_Seba": {
                "path": str(cl_seba_dir),
                "split_strategy": "preserve_roboflow",
                "counts": seba_counts,
                "total": seba_total,
            },
            "CL_Alberto": {
                "path": str(cl_alberto_dir),
                "split_strategy": "preserve_roboflow",
                "counts": alberto_counts,
                "total": alberto_total,
            },
            "International": {
                "path": str(intl_dir),
                "split_strategy": "stratified_by_source_db",
                "counts": intl_counts,
                "total": intl_total,
                "per_database": {
                    db: dict(splits) for db, splits in intl_db_counts.items()
                },
            },
        },
        "totals": {
            "train": seba_counts["train"] + alberto_counts["train"] + intl_counts["train"],
            "valid": seba_counts["valid"] + alberto_counts["valid"] + intl_counts["valid"],
            "test": seba_counts["test"] + alberto_counts["test"] + intl_counts["test"],
            "total": seba_total + alberto_total + intl_total,
        },
    }

    summary_path = output_dir / "build_summary.json"
    if not args.dry_run:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------
    grand_total = summary["totals"]

    print("=" * 70)
    print("  BUILD SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Source':<20s} {'Train':>8s} {'Valid':>8s} {'Test':>8s} {'Total':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'CL_Seba':<20s} {seba_counts['train']:>8d} {seba_counts['valid']:>8d} "
          f"{seba_counts['test']:>8d} {seba_total:>8d}")
    print(f"  {'CL_Alberto':<20s} {alberto_counts['train']:>8d} {alberto_counts['valid']:>8d} "
          f"{alberto_counts['test']:>8d} {alberto_total:>8d}")
    print(f"  {'International':<20s} {intl_counts['train']:>8d} {intl_counts['valid']:>8d} "
          f"{intl_counts['test']:>8d} {intl_total:>8d}")
    print(f"  {'='*20} {'='*8} {'='*8} {'='*8} {'='*8}")
    print(f"  {'TOTAL':<20s} {grand_total['train']:>8d} {grand_total['valid']:>8d} "
          f"{grand_total['test']:>8d} {grand_total['total']:>8d}")
    print()

    # International per-database breakdown
    if intl_db_counts:
        print("  International per-database breakdown:")
        for db_name in sorted(intl_db_counts.keys()):
            splits = intl_db_counts[db_name]
            db_total = sum(splits.values())
            print(f"    {db_name:<24s} train={splits.get('train', 0):>4d}  "
                  f"valid={splits.get('valid', 0):>4d}  "
                  f"test={splits.get('test', 0):>4d}  "
                  f"total={db_total:>4d}")
        print()

    if args.dry_run:
        print("[+] Dry run complete. No files were written.")
    else:
        print(f"[+] Output:   {output_dir}")
        print(f"[+] Manifest: {manifest_path}")
        print(f"[+] Summary:  {summary_path}")

    print("=" * 70)

    # -----------------------------------------------------------------------
    # Optional validation
    # -----------------------------------------------------------------------
    if args.validate and not args.dry_run:
        validate_random_labels(output_dir, n=5)


# ============================================================================
# CLI ARGUMENT PARSER
# ============================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build combined AMBEL dataset from Chilean + International sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build full dataset (defaults)
  python scripts/cli/build_combined_dataset.py

  # Dry run: count images without copying
  python scripts/cli/build_combined_dataset.py --dry-run

  # Build and validate random labels
  python scripts/cli/build_combined_dataset.py --validate

  # Custom international split ratios
  python scripts/cli/build_combined_dataset.py --intl-split 0.7,0.15,0.15

  # Custom output directory
  python scripts/cli/build_combined_dataset.py \\
      --output /media/malezainia2/E/ProcessingData/ambel-combined-intl-v2
        """,
    )

    parser.add_argument(
        "--cl-seba",
        type=str,
        default="/media/malezainia2/E/ProcessingData/ambrosia-lentejas-seba-v1",
        help="Path to CL_Seba Roboflow dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--cl-alberto",
        type=str,
        default="/media/malezainia2/E/ProcessingData/ambrosia.dataset_alberto/ambrosia.dataset",
        help="Path to CL_Alberto Roboflow dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--international",
        type=str,
        default="/media/malezainia2/E/ProcessingData/ambel-international-v1",
        help="Path to international dataset with manifest.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/media/malezainia2/E/ProcessingData/ambel-combined-intl-v1",
        help="Output directory for combined dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--intl-split",
        type=str,
        default="0.8,0.1,0.1",
        help="Train,valid,test ratios for international data (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count and report without copying files",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Spot-check 5 random labels per split after build",
    )

    return parser.parse_args()


def main():
    """Entry point for CLI execution."""
    args = parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
