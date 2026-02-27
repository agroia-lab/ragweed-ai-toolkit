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
    python tools/03_crossdomain/build_international_dataset.py

    # Dry run (count and report only, no file copying)
    python tools/03_crossdomain/build_international_dataset.py --dry-run

    # Build and validate 5 random labels
    python tools/03_crossdomain/build_international_dataset.py --validate

    # Custom config and output
    python tools/03_crossdomain/build_international_dataset.py \\
        --config configs/crossdomain/international_databases.yaml \\
        --output /media/malezainia2/E/ProcessingData/ambel-international-v1

    # Verbose logging for debugging
    python tools/03_crossdomain/build_international_dataset.py --dry-run --verbose
"""

import argparse
import sys
from pathlib import Path

import yaml

from ragweed_toolkit.crossdomain import (
    build_international_dataset,
    validate_labels,
)


# ============================================================================
# MAIN BUILD
# ============================================================================


def build_dataset(args) -> None:
    """
    Main dataset build function.

    Reads the YAML config, delegates to ragweed_toolkit.crossdomain.build_international_dataset,
    and prints a formatted report.

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

    # -----------------------------------------------------------------------
    # Build dataset using library function
    # -----------------------------------------------------------------------
    db_summaries = build_international_dataset(
        config_path=str(config_path),
        output_dir=str(output_dir),
        dry_run=args.dry_run,
        validate=False,  # We handle validation ourselves for better output
    )

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------
    total_images = sum(s.get("n_images", 0) for s in db_summaries.values())
    total_annots = sum(s.get("n_annotations", 0) for s in db_summaries.values())

    print("=" * 70)
    print("  BUILD SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Database':<20s} {'Images':>8s} {'Annots':>8s} {'Per img':>8s}  Status")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}  {'-'*8}")

    for db_name, s in db_summaries.items():
        n_img = s.get("n_images", 0)
        n_ann = s.get("n_annotations", 0)
        per_img = s.get("annots_per_image", 0)
        status = s.get("status", "unknown")
        print(f"  {db_name:<20s} {n_img:>8d} {n_ann:>8d} {per_img:>8.1f}  {status}")

    print(f"  {'='*20} {'='*8} {'='*8}")
    print(f"  {'TOTAL':<20s} {total_images:>8d} {total_annots:>8d}")
    print()

    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    manifest_path = output_dir / "manifest.csv"
    summary_path = output_dir / "build_summary.json"

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
        issues = validate_labels(out_labels, n=5)
        if issues:
            print(f"\n[!] Validation issues found:")
            for issue in issues:
                print(f"    {issue}")
        else:
            print("\n[+] All validated labels OK")


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
  python tools/03_crossdomain/build_international_dataset.py

  # Dry run: count images and annotations without copying
  python tools/03_crossdomain/build_international_dataset.py --dry-run

  # Build and validate random labels
  python tools/03_crossdomain/build_international_dataset.py --validate

  # Custom config and output
  python tools/03_crossdomain/build_international_dataset.py \\
      --config configs/crossdomain/international_databases.yaml \\
      --output /media/malezainia2/E/ProcessingData/ambel-international-v1

  # Verbose mode for debugging
  python tools/03_crossdomain/build_international_dataset.py --dry-run --verbose
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
