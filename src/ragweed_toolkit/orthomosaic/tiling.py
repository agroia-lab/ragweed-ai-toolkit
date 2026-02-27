"""
Orthomosaic Tiling
===================

Split a georeferenced orthomosaic (GeoTIFF) into JPEG tiles for YOLO
inference. Uses rasterio windowed reading to process rasters that are too
large for RAM. The alpha channel (band 4) is used to skip tiles with
insufficient valid pixels (nodata edges).

Functions
---------
tile_orthomosaic
    Tile a GeoTIFF into JPEG tiles with a tile manifest.
"""

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from tqdm import tqdm


def tile_orthomosaic(
    raster_path: str,
    output_dir: str,
    *,
    tile_size: int = 1024,
    overlap: int = 0,
    min_valid: float = 0.9,
    quality: int = 95,
) -> Dict:
    """Tile a GeoTIFF orthomosaic into JPEG tiles for YOLO inference.

    Handles 4-band (RGB+Alpha) rasters by using the alpha channel to
    skip tiles with insufficient valid pixels. Edge tiles are zero-padded
    to full ``tile_size``.

    Outputs in ``output_dir``:
        - ``tile_r{row:04d}_c{col:04d}.jpg`` tiles
        - ``tile_manifest.csv`` with UTM coordinates per tile
        - ``tiling_summary.json`` with processing metadata

    Args:
        raster_path: Path to input GeoTIFF orthomosaic.
        output_dir: Output directory for tiles and manifest.
        tile_size: Tile width and height in pixels.
        overlap: Overlap in pixels between adjacent tiles.
        min_valid: Minimum fraction of valid (non-alpha) pixels to keep a
            tile, in range [0, 1].
        quality: JPEG quality (1-100).

    Returns:
        Summary dict with ``saved_tiles``, ``skipped_tiles``, ``crs``, etc.
    """
    raster = Path(raster_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()

    with rasterio.open(raster) as src:
        width = src.width
        height = src.height
        transform = src.transform
        crs = str(src.crs)
        n_bands = src.count
        has_alpha = n_bands >= 4

        step = tile_size - overlap
        n_cols = math.ceil(width / step)
        n_rows = math.ceil(height / step)
        total = n_rows * n_cols

        manifest: List[Dict] = []
        skipped = 0
        saved = 0

        pbar = tqdm(total=total, desc="Tiling", unit="tile")

        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                col_off = col_idx * step
                row_off = row_idx * step
                win_w = min(tile_size, width - col_off)
                win_h = min(tile_size, height - row_off)
                window = Window(col_off, row_off, win_w, win_h)

                # Check alpha channel
                if has_alpha:
                    alpha = src.read(4, window=window)
                    n_valid = int(np.count_nonzero(alpha > 0))
                    frac_valid = n_valid / (tile_size * tile_size)
                    if frac_valid < min_valid:
                        skipped += 1
                        pbar.update(1)
                        continue
                else:
                    frac_valid = 1.0
                    n_valid = win_w * win_h

                rgb = src.read([1, 2, 3], window=window)

                # Pad edge tiles
                if win_w < tile_size or win_h < tile_size:
                    padded = np.zeros((3, tile_size, tile_size), dtype=rgb.dtype)
                    padded[:, :win_h, :win_w] = rgb
                    rgb = padded

                img_array = np.transpose(rgb, (1, 2, 0))
                img = Image.fromarray(img_array)

                fname = f"tile_r{row_idx:04d}_c{col_idx:04d}.jpg"
                img.save(out / fname, "JPEG", quality=quality)

                # UTM bounds
                x_min = transform.c + col_off * transform.a
                y_max = transform.f + row_off * transform.e
                x_max = x_min + tile_size * transform.a
                y_min = y_max + tile_size * transform.e

                manifest.append({
                    "filename": fname,
                    "row": row_idx,
                    "col": col_idx,
                    "x_min": round(x_min, 4),
                    "y_min": round(y_min, 4),
                    "x_max": round(x_max, 4),
                    "y_max": round(y_max, 4),
                    "n_valid_pixels": n_valid,
                    "frac_valid": round(frac_valid, 4),
                })
                saved += 1
                pbar.update(1)

        pbar.close()

    elapsed = (datetime.now() - start_time).total_seconds()

    # Write manifest CSV
    manifest_path = out / "tile_manifest.csv"
    fieldnames = [
        "filename", "row", "col", "x_min", "y_min", "x_max", "y_max",
        "n_valid_pixels", "frac_valid",
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)

    # Re-read bounds for summary
    with rasterio.open(raster) as src:
        bounds = src.bounds

    summary = {
        "input_path": str(raster.resolve()),
        "tile_size": tile_size,
        "overlap": overlap,
        "step": step,
        "min_valid": min_valid,
        "quality": quality,
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
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(out.resolve()),
    }

    (out / "tiling_summary.json").write_text(json.dumps(summary, indent=2))

    return summary


def main():
    """CLI entry point for orthomosaic tiling."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Tile a GeoTIFF orthomosaic into JPEG tiles for YOLO inference."
    )
    parser.add_argument("--input", required=True, help="Path to input GeoTIFF orthomosaic")
    parser.add_argument("--output", required=True, help="Output directory for tiles")
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size in pixels (default: 1024)")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap in pixels (default: 0)")
    parser.add_argument("--min-valid", type=float, default=0.9,
                        help="Min fraction of valid pixels to keep tile (default: 0.9)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality 1-100 (default: 95)")
    args = parser.parse_args()

    summary = tile_orthomosaic(
        args.input,
        args.output,
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_valid=args.min_valid,
        quality=args.quality,
    )

    print(f"\nTiling complete:")
    print(f"  Input: {summary['input_path']}")
    print(f"  CRS: {summary['crs']}")
    print(f"  Raster: {summary['raster_width']} x {summary['raster_height']} px")
    print(f"  Grid: {summary['n_rows']} rows x {summary['n_cols']} cols")
    print(f"  Saved: {summary['saved_tiles']} tiles, Skipped: {summary['skipped_tiles']}")
    print(f"  Output: {summary['output_dir']}")
    print(f"  Time: {summary['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
