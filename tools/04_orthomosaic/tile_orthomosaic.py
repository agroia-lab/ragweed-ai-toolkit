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
    python scripts/cli/tile_orthomosaic.py \\
        --raster /path/to/Ortomosaico.rgb.tif \\
        --output /path/to/ortho_tiles_1024/ \\
        --tile-size 1024 \\
        --min-valid 0.9 \\
        --quality 95 \\
        --overlap 0
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from tqdm import tqdm


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tile a large GeoTIFF orthomosaic into JPEG tiles for YOLO inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic tiling at 1024px
  python scripts/cli/tile_orthomosaic.py \\
      --raster /path/to/Ortomosaico.rgb.tif \\
      --output /path/to/ortho_tiles_1024/

  # Custom tile size with overlap
  python scripts/cli/tile_orthomosaic.py \\
      --raster /path/to/Ortomosaico.rgb.tif \\
      --output /path/to/ortho_tiles_640/ \\
      --tile-size 640 --overlap 64 --min-valid 0.8

  # Lower quality for faster processing
  python scripts/cli/tile_orthomosaic.py \\
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


def print_banner():
    """Print script banner."""
    print("=" * 70)
    print("  TILE ORTHOMOSAIC - GeoTIFF to JPEG tiles for YOLO inference")
    print("=" * 70)
    print()


def print_section(title):
    """Print a section header."""
    print("-" * 70)
    print(f"  {title}")
    print("-" * 70)


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


def print_raster_info(src, raster_path):
    """Print raster metadata."""
    print_section("Raster Information")

    file_size = Path(raster_path).stat().st_size

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

    has_alpha = src.count >= 4
    print(f"  Alpha band: {'Yes (band 4)' if has_alpha else 'No'}")
    print()


def print_tiling_plan(args, n_rows, n_cols, total, step):
    """Print tiling parameters before processing."""
    print_section("Tiling Plan")
    print(f"  Tile size:    {args.tile_size} x {args.tile_size} px")
    print(f"  Overlap:      {args.overlap} px")
    print(f"  Step size:    {step} px")
    print(f"  Grid:         {n_cols} cols x {n_rows} rows")
    print(f"  Total tiles:  {total}")
    print(f"  Min valid:    {args.min_valid:.0%}")
    print(f"  JPEG quality: {args.quality}")
    print(f"  Output dir:   {args.output}")
    print()


def write_manifest(manifest, output_dir):
    """Write tile manifest CSV."""
    manifest_path = output_dir / "tile_manifest.csv"
    fieldnames = [
        "filename",
        "row",
        "col",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "n_valid_pixels",
        "frac_valid",
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)
    return manifest_path


def write_summary(summary, output_dir):
    """Write tiling summary JSON."""
    summary_path = output_dir / "tiling_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary_path


def compute_disk_usage(output_dir):
    """Compute total disk usage of JPEG tiles in the output directory."""
    total = 0
    count = 0
    for p in output_dir.glob("tile_r*.jpg"):
        total += p.stat().st_size
        count += 1
    return total, count


def main():
    args = parse_args()

    print_banner()

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

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open raster and process
    start_time = datetime.now()

    with rasterio.open(raster_path) as src:
        width = src.width
        height = src.height
        transform = src.transform
        crs = str(src.crs)
        n_bands = src.count
        has_alpha = n_bands >= 4

        # Print raster info
        print_raster_info(src, raster_path)

        # Calculate grid
        step = args.tile_size - args.overlap
        n_cols = math.ceil(width / step)
        n_rows = math.ceil(height / step)
        total = n_rows * n_cols

        # Print tiling plan
        print_tiling_plan(args, n_rows, n_cols, total, step)

        # Tile the raster
        print_section("Processing Tiles")

        manifest = []
        skipped = 0
        saved = 0

        pbar = tqdm(total=total, desc="Tiling", unit="tile")

        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                # Window position
                col_off = col_idx * step
                row_off = row_idx * step

                # Clamp window to raster bounds
                win_width = min(args.tile_size, width - col_off)
                win_height = min(args.tile_size, height - row_off)

                window = Window(col_off, row_off, win_width, win_height)

                # Check alpha for valid pixel fraction
                if has_alpha:
                    alpha = src.read(4, window=window)
                    n_valid = int(np.count_nonzero(alpha > 0))
                    frac_valid = n_valid / (args.tile_size * args.tile_size)
                    if frac_valid < args.min_valid:
                        skipped += 1
                        pbar.update(1)
                        pbar.set_postfix(saved=saved, skipped=skipped)
                        continue
                else:
                    frac_valid = 1.0
                    n_valid = win_width * win_height

                # Read RGB bands (windowed - never loads full raster)
                rgb = src.read([1, 2, 3], window=window)  # shape: (3, H, W)

                # Pad edge tiles to full tile_size if needed
                if win_width < args.tile_size or win_height < args.tile_size:
                    padded = np.zeros(
                        (3, args.tile_size, args.tile_size), dtype=rgb.dtype
                    )
                    padded[:, :win_height, :win_width] = rgb
                    rgb = padded

                # Transpose to HWC for PIL: (3, H, W) -> (H, W, 3)
                img_array = np.transpose(rgb, (1, 2, 0))
                img = Image.fromarray(img_array)

                # Save tile as JPEG
                fname = f"tile_r{row_idx:04d}_c{col_idx:04d}.jpg"
                img.save(output_dir / fname, "JPEG", quality=args.quality)

                # Compute UTM bounds from affine transform
                x_min = transform.c + col_off * transform.a
                y_max = transform.f + row_off * transform.e  # e is negative
                x_max = x_min + args.tile_size * transform.a
                y_min = y_max + args.tile_size * transform.e

                manifest.append(
                    {
                        "filename": fname,
                        "row": row_idx,
                        "col": col_idx,
                        "x_min": round(x_min, 4),
                        "y_min": round(y_min, 4),
                        "x_max": round(x_max, 4),
                        "y_max": round(y_max, 4),
                        "n_valid_pixels": n_valid,
                        "frac_valid": round(frac_valid, 4),
                    }
                )

                saved += 1
                pbar.update(1)
                pbar.set_postfix(saved=saved, skipped=skipped)

        pbar.close()

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    # Write tile manifest CSV
    print()
    print_section("Writing Outputs")

    manifest_path = write_manifest(manifest, output_dir)
    print(f"  Manifest: {manifest_path} ({len(manifest)} rows)")

    # Compute raster bounds for summary
    with rasterio.open(raster_path) as src:
        bounds = src.bounds

    # Build summary
    summary = {
        "input_path": str(raster_path.resolve()),
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "step": step,
        "min_valid": args.min_valid,
        "quality": args.quality,
        "crs": crs,
        "bounds": {
            "left": round(bounds.left, 4),
            "bottom": round(bounds.bottom, 4),
            "right": round(bounds.right, 4),
            "top": round(bounds.top, 4),
        },
        "raster_width": width,
        "raster_height": height,
        "n_bands": n_bands,
        "has_alpha": has_alpha,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "total_tiles": total,
        "skipped_tiles": skipped,
        "saved_tiles": saved,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(output_dir.resolve()),
    }

    summary_path = write_summary(summary, output_dir)
    print(f"  Summary:  {summary_path}")

    # Compute disk usage of saved tiles
    disk_bytes, disk_count = compute_disk_usage(output_dir)

    # Final summary
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
