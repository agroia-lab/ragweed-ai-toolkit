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
    python tools/03_crossdomain/build_combined_dataset.py

    # Dry run (count and report only, no file copying)
    python tools/03_crossdomain/build_combined_dataset.py --dry-run

    # Build and validate random labels
    python tools/03_crossdomain/build_combined_dataset.py --validate

    # Custom international split ratios
    python tools/03_crossdomain/build_combined_dataset.py --intl-split 0.7,0.15,0.15
"""

import argparse
import sys
from pathlib import Path

from ragweed_toolkit.crossdomain import build_combined_dataset, validate_labels

# ============================================================================
# MAIN BUILD
# ============================================================================


def build_dataset(args) -> None:
    """
    Main dataset build function.

    Delegates to ragweed_toolkit.crossdomain.build_combined_dataset and
    prints a formatted report.

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

    # -----------------------------------------------------------------------
    # Build dataset using library function
    # -----------------------------------------------------------------------
    roboflow_sources = [
        {"path": str(cl_seba_dir), "prefix": "CL_Seba"},
        {"path": str(cl_alberto_dir), "prefix": "CL_Alberto"},
    ]

    source_summaries = build_combined_dataset(
        roboflow_sources=roboflow_sources,
        international_dir=str(intl_dir),
        output_dir=str(output_dir),
        intl_split=ratios,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  BUILD SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Source':<20s} {'Train':>8s} {'Valid':>8s} {'Test':>8s} {'Total':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    grand_train = 0
    grand_valid = 0
    grand_test = 0
    grand_total = 0

    for source_name, counts in source_summaries.items():
        train = counts.get("train", 0)
        valid = counts.get("valid", 0)
        test = counts.get("test", 0)
        total = train + valid + test
        grand_train += train
        grand_valid += valid
        grand_test += test
        grand_total += total
        print(f"  {source_name:<20s} {train:>8d} {valid:>8d} {test:>8d} {total:>8d}")

    print(f"  {'='*20} {'='*8} {'='*8} {'='*8} {'='*8}")
    print(f"  {'TOTAL':<20s} {grand_train:>8d} {grand_valid:>8d} {grand_test:>8d} {grand_total:>8d}")
    print()

    manifest_path = output_dir / "manifest.csv"
    summary_path = output_dir / "build_summary.json"

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
        print("\n[+] Validating random labels per split...")
        all_ok = True
        for split in ["train", "valid", "test"]:
            label_dir = output_dir / split / "labels"
            if not label_dir.exists():
                continue
            issues = validate_labels(label_dir, n=5)
            if issues:
                all_ok = False
                for issue in issues:
                    print(f"  [{split}] {issue}")
        if all_ok:
            print("[+] All validated labels OK")
        else:
            print("[!] Some labels have issues - review above")


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
  python tools/03_crossdomain/build_combined_dataset.py

  # Dry run: count images without copying
  python tools/03_crossdomain/build_combined_dataset.py --dry-run

  # Build and validate random labels
  python tools/03_crossdomain/build_combined_dataset.py --validate

  # Custom international split ratios
  python tools/03_crossdomain/build_combined_dataset.py --intl-split 0.7,0.15,0.15

  # Custom output directory
  python tools/03_crossdomain/build_combined_dataset.py \\
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
