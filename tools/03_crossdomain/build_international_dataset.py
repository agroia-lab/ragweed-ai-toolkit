#!/usr/bin/env python3
"""
Build International Ragweed Dataset
====================================

Reads configs/crossdomain/international_databases.yaml and builds a unified
single-class YOLO dataset (class 0 = AMBEL) from multiple international
ragweed databases, each with different annotation formats.

Supported source formats:
  - yolo_txt           : Standard YOLO TXT (separate images/ and labels/ dirs)
  - yolo_txt_multiclass: YOLO TXT with multiple classes, filter to ragweed only
  - voc_xml            : Pascal VOC XML co-located with images (Michigan)

Usage:
    # Build full dataset
    python scripts/cli/build_international_dataset.py

    # Dry run (count and report only, no file copying)
    python scripts/cli/build_international_dataset.py --dry-run

    # Build and validate 5 random labels
    python scripts/cli/build_international_dataset.py --validate

    # Custom config and output
    python scripts/cli/build_international_dataset.py \\
        --config configs/crossdomain/international_databases.yaml \\
        --output /media/malezainia2/E/ProcessingData/ambel-international-v1

    # Verbose logging for debugging
    python scripts/cli/build_international_dataset.py --dry-run --verbose
"""

import argparse
import csv
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm


# ============================================================================
# LABEL PROCESSING FUNCTIONS
# ============================================================================


def process_yolo_labels(
    label_path: Path,
    ragweed_class: int,
    filter_ragweed_only: bool,
) -> Optional[List[str]]:
    """
    Read a YOLO TXT label file, filter/remap ragweed annotations to class 0.

    If filter_ragweed_only is True, only lines matching ragweed_class are kept.
    Otherwise all lines are expected to be ragweed_class (single-class db).

    Args:
        label_path: Path to the YOLO .txt label file.
        ragweed_class: Source class ID for ragweed.
        filter_ragweed_only: If True, keep only ragweed lines from multi-class.

    Returns:
        List of remapped YOLO label lines (class 0), or None if no ragweed found.
    """
    if not label_path.exists():
        return None

    lines_out = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id != ragweed_class:
                if filter_ragweed_only:
                    continue
                else:
                    # Single-class DB but unexpected class -- skip line
                    continue
            # Remap to class 0
            lines_out.append(f"0 {' '.join(parts[1:])}")

    return lines_out if lines_out else None


def process_voc_xml(
    xml_path: Path,
    ragweed_class_name: str,
) -> Optional[Tuple[List[str], int, int]]:
    """
    Parse a Pascal VOC XML annotation file, extract ragweed bboxes as YOLO format.

    Args:
        xml_path: Path to the .xml annotation file.
        ragweed_class_name: VOC class name for ragweed (e.g., "Ragweed").

    Returns:
        Tuple of (label_lines, img_width, img_height) or None if no ragweed.
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None

    root = tree.getroot()

    # Get image dimensions
    size_elem = root.find("size")
    if size_elem is None:
        return None
    img_width = int(size_elem.findtext("width", "0"))
    img_height = int(size_elem.findtext("height", "0"))
    if img_width == 0 or img_height == 0:
        return None

    lines_out = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "")
        if name != ragweed_class_name:
            continue
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.findtext("xmin", "0"))
        ymin = float(bndbox.findtext("ymin", "0"))
        xmax = float(bndbox.findtext("xmax", "0"))
        ymax = float(bndbox.findtext("ymax", "0"))

        # VOC -> YOLO conversion (normalized)
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        lines_out.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return (lines_out, img_width, img_height) if lines_out else None


# ============================================================================
# PER-DATABASE HANDLERS
# ============================================================================


def process_yolo_txt(
    db_name: str,
    db_cfg: Dict[str, Any],
    out_images: Path,
    out_labels: Path,
    dry_run: bool,
    verbose: bool,
) -> List[Dict]:
    """
    Process a database with YOLO TXT labels (separate images/ and labels/ dirs).

    Handles: nd_individual, nd_aerial (yolo_txt and yolo_txt_multiclass).

    Args:
        db_name: Database key from YAML config.
        db_cfg: Database configuration dict.
        out_images: Output images directory.
        out_labels: Output labels directory.
        dry_run: If True, count only without copying files.
        verbose: If True, print per-file details.

    Returns:
        List of manifest row dicts.
    """
    images_dir = Path(db_cfg["images"])
    label_dir = Path(db_cfg["labels"])
    prefix = db_cfg["prefix"]
    ragweed_class = db_cfg["ragweed_class_id"]
    filter_ragweed = db_cfg.get("filter_ragweed_only", False)
    image_ext = db_cfg.get("image_ext", ".jpg")

    # Collect image files
    image_files = sorted(images_dir.glob(f"*{image_ext}"), key=lambda p: p.name)

    manifest_rows = []
    skipped = 0

    for img_path in tqdm(image_files, desc=f"  {db_name}", leave=False):
        stem = img_path.stem

        # Find label file
        label_path = label_dir / f"{stem}.txt"

        # Process labels
        label_lines = process_yolo_labels(label_path, ragweed_class, filter_ragweed)
        if label_lines is None:
            skipped += 1
            if verbose:
                print(f"    [-] No ragweed labels: {stem}")
            continue

        # Output filenames
        out_name = f"{prefix}_{img_path.name}"
        out_label_name = f"{prefix}_{stem}.txt"

        if not dry_run:
            shutil.copy2(img_path, out_images / out_name)
            with open(out_labels / out_label_name, "w") as f:
                f.write("\n".join(label_lines) + "\n")

        manifest_rows.append({
            "filename": out_name,
            "source_db": db_name,
            "original_path": str(img_path),
            "n_annotations": len(label_lines),
        })

    if verbose and skipped > 0:
        print(f"    [!] Skipped {skipped} images (no ragweed)")

    return manifest_rows


def process_voc_xml_db(
    db_name: str,
    db_cfg: Dict[str, Any],
    out_images: Path,
    out_labels: Path,
    dry_run: bool,
    verbose: bool,
) -> List[Dict]:
    """
    Process a database with Pascal VOC XML annotations (co-located with images).

    Handles: michigan_3season (images and XMLs in yearly subdirectories).

    Args:
        db_name: Database key from YAML config.
        db_cfg: Database configuration dict.
        out_images: Output images directory.
        out_labels: Output labels directory.
        dry_run: If True, count only without copying files.
        verbose: If True, print per-file details.

    Returns:
        List of manifest row dicts.
    """
    prefix = db_cfg["prefix"]
    ragweed_class_name = db_cfg["ragweed_class_name"]
    data_dirs = db_cfg.get("data_dirs", [])
    image_ext = db_cfg.get("image_ext", ".jpg")

    # Collect image files from all data directories
    image_files = []
    for dir_path_str in data_dirs:
        dir_path = Path(dir_path_str)
        if not dir_path.exists():
            print(f"    [!] Data dir not found: {dir_path}")
            continue
        image_files.extend(dir_path.glob(f"*{image_ext}"))
    image_files.sort(key=lambda p: p.name)

    manifest_rows = []
    skipped = 0
    errors = 0

    for img_path in tqdm(image_files, desc=f"  {db_name}", leave=False):
        stem = img_path.stem

        # XML file is co-located with image
        xml_path = img_path.with_suffix(".xml")
        if not xml_path.exists():
            skipped += 1
            if verbose:
                print(f"    [-] No XML: {stem}")
            continue

        try:
            result = process_voc_xml(xml_path, ragweed_class_name)
        except Exception as e:
            errors += 1
            print(f"    [!] Error parsing {xml_path.name}: {e}")
            continue

        if result is None:
            skipped += 1
            if verbose:
                print(f"    [-] No ragweed in XML: {stem}")
            continue

        label_lines, _, _ = result

        # Output filenames
        out_name = f"{prefix}_{img_path.name}"
        out_label_name = f"{prefix}_{stem}.txt"

        if not dry_run:
            shutil.copy2(img_path, out_images / out_name)
            with open(out_labels / out_label_name, "w") as f:
                f.write("\n".join(label_lines) + "\n")

        manifest_rows.append({
            "filename": out_name,
            "source_db": db_name,
            "original_path": str(img_path),
            "n_annotations": len(label_lines),
        })

    if verbose and skipped > 0:
        print(f"    [!] Skipped {skipped} images (no ragweed or no XML)")
    if errors > 0:
        print(f"    [!] {errors} XML parsing errors")

    return manifest_rows


def process_yolo_txt_multiclass_flat(
    db_name: str,
    db_cfg: Dict[str, Any],
    out_images: Path,
    out_labels: Path,
    dry_run: bool,
    verbose: bool,
) -> List[Dict]:
    """
    Process a database with YOLO TXT labels co-located with images in subdirs.

    Handles: weedcrop_precag (images and labels in crop subdirectories,
    .JPG + .txt pairs, classes.txt must be excluded).

    Args:
        db_name: Database key from YAML config.
        db_cfg: Database configuration dict.
        out_images: Output images directory.
        out_labels: Output labels directory.
        dry_run: If True, count only without copying files.
        verbose: If True, print per-file details.

    Returns:
        List of manifest row dicts.
    """
    base_dir = Path(db_cfg["base_dir"])
    prefix = db_cfg["prefix"]
    ragweed_class = db_cfg["ragweed_class_id"]
    filter_ragweed = db_cfg.get("filter_ragweed_only", False)
    crop_subdirs = db_cfg.get("crop_subdirs", [])
    image_ext = db_cfg.get("image_ext", ".JPG")
    exclude_files = set(db_cfg.get("exclude_files", []))

    # Collect image files from all crop subdirectories
    image_files = []
    for subdir_name in crop_subdirs:
        subdir = base_dir / subdir_name
        if not subdir.exists():
            print(f"    [!] Subdirectory not found: {subdir}")
            continue
        image_files.extend(subdir.glob(f"*{image_ext}"))
    image_files.sort(key=lambda p: p.name)

    manifest_rows = []
    skipped = 0

    for img_path in tqdm(image_files, desc=f"  {db_name}", leave=False):
        stem = img_path.stem

        # Skip excluded files
        if f"{stem}.txt" in exclude_files:
            continue

        # Label file is co-located: same directory, same stem, .txt extension
        label_path = img_path.with_suffix(".txt")

        # Process labels
        label_lines = process_yolo_labels(label_path, ragweed_class, filter_ragweed)
        if label_lines is None:
            skipped += 1
            if verbose:
                print(f"    [-] No ragweed labels: {stem}")
            continue

        # Output filenames (normalize extension to lowercase .jpg)
        out_ext = img_path.suffix.lower()
        out_name = f"{prefix}_{stem}{out_ext}"
        out_label_name = f"{prefix}_{stem}.txt"

        if not dry_run:
            shutil.copy2(img_path, out_images / out_name)
            with open(out_labels / out_label_name, "w") as f:
                f.write("\n".join(label_lines) + "\n")

        manifest_rows.append({
            "filename": out_name,
            "source_db": db_name,
            "original_path": str(img_path),
            "n_annotations": len(label_lines),
        })

    if verbose and skipped > 0:
        print(f"    [!] Skipped {skipped} images (no ragweed)")

    return manifest_rows


# ============================================================================
# FORMAT DISPATCHER
# ============================================================================

# Maps format string from YAML to handler function
FORMAT_HANDLERS = {
    "yolo_txt": process_yolo_txt,
    "yolo_txt_multiclass": process_yolo_txt,  # Same handler, filter_ragweed_only differs
    "voc_xml": process_voc_xml_db,
}


def dispatch_database(
    db_name: str,
    db_cfg: Dict[str, Any],
    out_images: Path,
    out_labels: Path,
    dry_run: bool,
    verbose: bool,
) -> List[Dict]:
    """
    Dispatch a database to the appropriate handler based on its format and structure.

    Args:
        db_name: Database key from YAML config.
        db_cfg: Database configuration dict.
        out_images: Output images directory.
        out_labels: Output labels directory.
        dry_run: If True, count only without copying files.
        verbose: If True, print per-file details.

    Returns:
        List of manifest row dicts.
    """
    fmt = db_cfg.get("format", "yolo_txt")

    # WeedCrop uses yolo_txt_multiclass but has a flat layout with crop subdirs
    if "base_dir" in db_cfg and "crop_subdirs" in db_cfg:
        return process_yolo_txt_multiclass_flat(
            db_name, db_cfg, out_images, out_labels, dry_run, verbose
        )

    # VOC XML databases (Michigan)
    if fmt == "voc_xml":
        return process_voc_xml_db(
            db_name, db_cfg, out_images, out_labels, dry_run, verbose
        )

    # Standard YOLO TXT (ND Individual, ND Aerial)
    handler = FORMAT_HANDLERS.get(fmt)
    if handler is None:
        print(f"[!] Unknown format '{fmt}' for {db_name}, skipping")
        return []

    return handler(db_name, db_cfg, out_images, out_labels, dry_run, verbose)


# ============================================================================
# VALIDATION
# ============================================================================


def validate_random_labels(out_labels: Path, n: int = 5) -> None:
    """
    Spot-check a random sample of output label files.

    Reads n random label files and verifies:
    - All lines have exactly 5 fields
    - Class is always 0
    - Coordinates are in [0, 1]

    Args:
        out_labels: Path to the output labels directory.
        n: Number of labels to check.
    """
    label_files = list(out_labels.glob("*.txt"))
    if not label_files:
        print("[!] No label files found for validation")
        return

    sample = random.sample(label_files, min(n, len(label_files)))
    print(f"\n[+] Validating {len(sample)} random labels...")
    all_ok = True

    for lf in sample:
        issues = []
        with open(lf, "r") as f:
            for i, line in enumerate(f, 1):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                parts = line_stripped.split()
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
        print(f"  {lf.name}: {status}")
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

    Reads the YAML config, processes each database, copies images and labels
    to the output directory, and generates manifest.csv + build_summary.json.

    Args:
        args: Parsed command-line arguments.
    """
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[-] Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Output directory
    output_dir = Path(args.output) if args.output else Path(config["output_dir"])
    out_images = output_dir / "images"
    out_labels = output_dir / "labels"

    target_class_name = config.get("target_class_name", "AMBEL")

    # -----------------------------------------------------------------------
    # Banner
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  BUILD INTERNATIONAL RAGWEED DATASET")
    if args.dry_run:
        print("  *** DRY RUN - no files will be copied ***")
    print("=" * 70)
    print(f"  Config:  {config_path}")
    print(f"  Output:  {output_dir}")
    print(f"  Target:  class 0 = {target_class_name}")
    databases = config.get("databases", {})
    print(f"  Sources: {len(databases)} databases")
    print("=" * 70)
    print()

    # Create output directories
    if not args.dry_run:
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Process each database
    # -----------------------------------------------------------------------
    all_manifest_rows: List[Dict] = []
    db_summaries: Dict[str, Dict] = {}

    for db_name, db_cfg in databases.items():
        fmt = db_cfg.get("format", "yolo_txt")
        prefix = db_cfg.get("prefix", db_name)

        print("-" * 70)
        print(f"  Processing: {db_name}")
        print(f"  Format: {fmt} | Prefix: {prefix}")
        print("-" * 70)

        # Check that at least one required path exists
        check_path = db_cfg.get("images") or db_cfg.get("base_dir") or (
            db_cfg.get("data_dirs", [None])[0]
        )
        if check_path and not Path(check_path).exists():
            print(f"  [!] Path not found: {check_path}")
            print(f"  [!] Skipping {db_name}")
            db_summaries[db_name] = {
                "status": "skipped",
                "reason": "path not found",
                "n_images": 0,
                "n_annotations": 0,
            }
            print()
            continue

        try:
            rows = dispatch_database(
                db_name, db_cfg, out_images, out_labels, args.dry_run, args.verbose
            )
        except Exception as e:
            print(f"  [!] Error processing {db_name}: {e}")
            import traceback
            traceback.print_exc()
            db_summaries[db_name] = {
                "status": "error",
                "reason": str(e),
                "n_images": 0,
                "n_annotations": 0,
            }
            print()
            continue

        n_images = len(rows)
        n_annots = sum(r["n_annotations"] for r in rows)
        all_manifest_rows.extend(rows)

        expected = db_cfg.get("expected_count", 0)
        match_str = ""
        if expected > 0:
            diff = n_images - expected
            if diff == 0:
                match_str = " (matches expected)"
            else:
                match_str = f" (expected {expected}, diff {diff:+d})"

        db_summaries[db_name] = {
            "status": "ok",
            "n_images": n_images,
            "n_annotations": n_annots,
            "prefix": prefix,
            "format": fmt,
            "expected_count": expected,
            "annots_per_image": round(n_annots / n_images, 1) if n_images > 0 else 0,
        }

        print(f"  [+] {db_name}: {n_images} images, {n_annots} annotations{match_str}")
        print()

    # -----------------------------------------------------------------------
    # Write manifest.csv
    # -----------------------------------------------------------------------
    manifest_path = output_dir / "manifest.csv"
    if not args.dry_run:
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["filename", "source_db", "original_path", "n_annotations"]
            )
            writer.writeheader()
            writer.writerows(all_manifest_rows)

    # -----------------------------------------------------------------------
    # Write build_summary.json
    # -----------------------------------------------------------------------
    total_images = sum(s["n_images"] for s in db_summaries.values())
    total_annots = sum(s["n_annotations"] for s in db_summaries.values())

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": str(config_path),
        "output_dir": str(output_dir),
        "dry_run": args.dry_run,
        "total_images": total_images,
        "total_annotations": total_annots,
        "databases": db_summaries,
    }

    summary_path = output_dir / "build_summary.json"
    if not args.dry_run:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  BUILD SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Database':<20s} {'Images':>8s} {'Annots':>8s} {'Per img':>8s}  Status")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}  {'-'*8}")

    for db_name, s in db_summaries.items():
        n_img = s["n_images"]
        n_ann = s["n_annotations"]
        per_img = s.get("annots_per_image", 0)
        status = s["status"]
        print(f"  {db_name:<20s} {n_img:>8d} {n_ann:>8d} {per_img:>8.1f}  {status}")

    print(f"  {'='*20} {'='*8} {'='*8}")
    print(f"  {'TOTAL':<20s} {total_images:>8d} {total_annots:>8d}")
    print()

    if args.dry_run:
        print("[+] Dry run complete. No files were written.")
    else:
        print(f"[+] Images:   {out_images}")
        print(f"[+] Labels:   {out_labels}")
        print(f"[+] Manifest: {manifest_path}")
        print(f"[+] Summary:  {summary_path}")

    print("=" * 70)

    # -----------------------------------------------------------------------
    # Optional validation
    # -----------------------------------------------------------------------
    if args.validate and not args.dry_run:
        validate_random_labels(out_labels, n=5)


# ============================================================================
# CLI ARGUMENT PARSER
# ============================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build unified single-class YOLO dataset from international ragweed databases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build full dataset (default config and output from YAML)
  python scripts/cli/build_international_dataset.py

  # Dry run: count images and annotations without copying
  python scripts/cli/build_international_dataset.py --dry-run

  # Build and validate random labels
  python scripts/cli/build_international_dataset.py --validate

  # Custom config and output
  python scripts/cli/build_international_dataset.py \\
      --config configs/crossdomain/international_databases.yaml \\
      --output /media/malezainia2/E/ProcessingData/ambel-international-v1

  # Verbose mode for debugging
  python scripts/cli/build_international_dataset.py --dry-run --verbose
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/crossdomain/international_databases.yaml",
        help="Path to database configuration YAML (default: configs/crossdomain/international_databases.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: from YAML config output_dir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count and report without copying files",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Spot-check 5 random labels after build",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file skip/error details",
    )

    return parser.parse_args()


def main():
    """Entry point for CLI execution."""
    args = parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
