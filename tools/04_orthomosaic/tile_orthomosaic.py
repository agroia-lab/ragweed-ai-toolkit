#!/usr/bin/env python3
"""
tile_orthomosaic.py - Tile a large GeoTIFF orthomosaic into JPEG tiles for YOLO inference.

Designed for Pix4D orthomosaics that are too large to load into RAM
(e.g., 2.4 GB, 73137x40738 px). Uses rasterio windowed reading to process
the raster tile-by-tile without ever loading the full image.

Handles 4-band (RGB+Alpha) rasters by using the alpha channel to skip
tiles with insufficient valid pixels (e.g., nodata edges of the orthomosaic).

Outputs:
    - JPEG tiles at the specified size (default 1024x1024)
    - tile_manifest.csv with UTM coordinates for each tile
    - tiling_summary.json with processing metadata

Usage:
    python tools/04_orthomosaic/tile_orthomosaic.py \\
        --raster /path/to/Ortomosaico.rgb.tif \\
        --output /path/to/ortho_tiles_1024/ \\
        --tile-size 1024 \\
        --min-valid 0.9 \\
        --quality 95 \\
        --overlap 0
"""

import argparse
import sys
from pathlib import Path

import rasterio

from ragweed_toolkit.orthomosaic import tile_orthomosaic


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tile a large GeoTIFF orthomosaic into JPEG tiles for YOLO inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic tiling at 1024px
  python tools/04_orthomosaic/tile_orthomosaic.py \\
      --raster /path/to/Ortomosaico.rgb.tif \\
      --output /path/to/ortho_tiles_1024/

  # Custom tile size with overlap
  python tools/04_orthomosaic/tile_orthomosaic.py \\
      --raster /path/to/Ortomosaico.rgb.tif \\
      --output /path/to/ortho_tiles_640/ \\
      --tile-size 640 --overlap 64 --min-valid 0.8

  # Lower quality for faster processing
  python tools/04_orthomosaic/tile_orthomosaic.py \\
      --raster /path/to/Ortomosaico.rgb.tif \\
      --output /path/to/ortho_tiles_draft/ \\
      --quality 80
        """,
    )
    parser.add_argument(
        "--raster",
        type=str,
        required=True,
        help="Path to input GeoTIFF orthomosaic",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for JPEG tiles",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile size in pixels (default: 1024)",
    )
    parser.add_argument(
        "--min-valid",
        type=float,
        default=0.9,
        help="Minimum fraction of valid (non-alpha) pixels to keep a tile, 0-1 (default: 0.9)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality, 1-100 (default: 95)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Overlap in pixels between adjacent tiles (default: 0)",
    )
    return parser.parse_args()


def format_size(size_bytes):
    """Format byte count as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.2f} GB"


def compute_disk_usage(output_dir):
    """Compute total disk usage of JPEG tiles in the output directory."""
    total = 0
    count = 0
    for p in Path(output_dir).glob("tile_r*.jpg"):
        total += p.stat().st_size
        count += 1
    return total, count


def main():
    args = parse_args()

    # Validate inputs
    raster_path = Path(args.raster)
    if not raster_path.exists():
        print(f"ERROR: Raster file not found: {raster_path}")
        sys.exit(1)

    if args.tile_size < 32:
        print(f"ERROR: Tile size must be >= 32, got {args.tile_size}")
        sys.exit(1)

    if args.overlap < 0:
        print(f"ERROR: Overlap must be >= 0, got {args.overlap}")
        sys.exit(1)

    if args.overlap >= args.tile_size:
        print(f"ERROR: Overlap ({args.overlap}) must be less than tile size ({args.tile_size})")
        sys.exit(1)

    if not (0.0 <= args.min_valid <= 1.0):
        print(f"ERROR: min-valid must be between 0 and 1, got {args.min_valid}")
        sys.exit(1)

    if not (1 <= args.quality <= 100):
        print(f"ERROR: JPEG quality must be between 1 and 100, got {args.quality}")
        sys.exit(1)

    output_dir = Path(args.output)

    # -----------------------------------------------------------------------
    # Banner
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  TILE ORTHOMOSAIC - GeoTIFF to JPEG tiles for YOLO inference")
    print("=" * 70)
    print()

    # Print raster info before tiling
    print("-" * 70)
    print("  Raster Information")
    print("-" * 70)

    file_size = raster_path.stat().st_size
    with rasterio.open(raster_path) as src:
        has_alpha = src.count >= 4
        step = args.tile_size - args.overlap

        print(f"  File:       {raster_path}")
        print(f"  File size:  {format_size(file_size)}")
        print(f"  Dimensions: {src.width} x {src.height} px")
        print(f"  Bands:      {src.count}")
        print(f"  Data type:  {src.dtypes[0]}")
        print(f"  CRS:        {src.crs}")
        print(f"  Resolution: {abs(src.transform.a):.4f} x {abs(src.transform.e):.4f} m/px")
        bounds = src.bounds
        print(f"  Bounds:     ({bounds.left:.2f}, {bounds.bottom:.2f}) - ({bounds.right:.2f}, {bounds.top:.2f})")
        print(f"  NoData:     {src.nodata}")
        print(f"  Alpha band: {'Yes (band 4)' if has_alpha else 'No'}")
    print()

    # Print tiling plan
    print("-" * 70)
    print("  Tiling Plan")
    print("-" * 70)
    print(f"  Tile size:    {args.tile_size} x {args.tile_size} px")
    print(f"  Overlap:      {args.overlap} px")
    print(f"  Step size:    {step} px")
    print(f"  Min valid:    {args.min_valid:.0%}")
    print(f"  JPEG quality: {args.quality}")
    print(f"  Output dir:   {args.output}")
    print()

    # -----------------------------------------------------------------------
    # Tile using library function
    # -----------------------------------------------------------------------
    print("-" * 70)
    print("  Processing Tiles")
    print("-" * 70)

    summary = tile_orthomosaic(
        raster_path=str(raster_path),
        output_dir=str(output_dir),
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_valid=args.min_valid,
        quality=args.quality,
    )

    saved = summary["saved_tiles"]
    skipped = summary["skipped_tiles"]
    total = summary["total_tiles"]
    elapsed = summary["elapsed_seconds"]
    n_cols = summary["n_cols"]
    n_rows = summary["n_rows"]

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    disk_bytes, disk_count = compute_disk_usage(output_dir)

    print()
    print("=" * 70)
    print("  TILING COMPLETE")
    print("=" * 70)
    print(f"  Saved tiles:   {saved:,}")
    print(f"  Skipped tiles: {skipped:,} (below {args.min_valid:.0%} valid pixels)")
    print(f"  Total grid:    {total:,} ({n_cols} cols x {n_rows} rows)")
    print(f"  Disk usage:    {format_size(disk_bytes)} ({disk_count} files)")
    if saved > 0:
        avg_size = disk_bytes / saved
        print(f"  Avg tile size: {format_size(int(avg_size))}")
    print(f"  Elapsed time:  {elapsed:.1f} s")
    print(f"  Output dir:    {output_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
