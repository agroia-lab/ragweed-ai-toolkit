#!/usr/bin/env python3
"""
Detections to GeoPackage Converter
====================================

Converts SAHI detection results + tile_manifest.csv into a GeoPackage with
UTM point geometries. Each detection becomes a georeferenced point on the map.

Supports two SAHI output formats:
    1. detections.json (--save-detections mode): Per-detection bounding boxes
       with pixel coordinates. Each detection is converted to a UTM point at
       the center of its bounding box.
    2. summary.json: Per-image detection counts (no individual bounding boxes).
       One point is created per tile at the tile centroid, with the detection
       count as an attribute.

The pixel-to-UTM conversion uses the tile_manifest.csv (from tile_orthomosaic.py)
which contains the UTM bounds of each tile.

Usage:
    # From detections.json (per-detection bounding boxes)
    python tools/04_orthomosaic/detections_to_geopackage.py \\
        --detections /path/to/sahi_output/detections.json \\
        --manifest /path/to/ortho_tiles_1024/tile_manifest.csv \\
        --output /path/to/ambel_ortho_detections.gpkg \\
        --crs EPSG:32719

    # From summary.json (per-image counts, centroid points)
    python tools/04_orthomosaic/detections_to_geopackage.py \\
        --detections /path/to/sahi_output/summary.json \\
        --manifest /path/to/ortho_tiles_1024/tile_manifest.csv \\
        --output /path/to/ambel_ortho_detections.gpkg

    # With confidence threshold filter
    python tools/04_orthomosaic/detections_to_geopackage.py \\
        --detections /path/to/detections.json \\
        --manifest /path/to/tile_manifest.csv \\
        --output /path/to/filtered_detections.gpkg \\
        --conf-threshold 0.35

Output:
    output.gpkg                - GeoPackage with point geometries
    detection_summary.json     - Statistics: counts by class, confidence distribution
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from ragweed_toolkit.orthomosaic import (
    detections_to_geopackage,
    load_tile_manifest,
)

# ============================================================================
# BANNER AND OUTPUT HELPERS
# ============================================================================


def print_banner():
    """Print script banner."""
    print("=" * 70)
    print("  DETECTIONS TO GEOPACKAGE - SAHI results to georeferenced points")
    print("=" * 70)
    print()


def print_section(title):
    """Print a section header."""
    print()
    print("-" * 70)
    print(f"  {title}")
    print("-" * 70)


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Convert SAHI detection results + tile manifest to GeoPackage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # From detections.json (per-detection bounding boxes)
    python tools/04_orthomosaic/detections_to_geopackage.py \\
        --detections /path/to/sahi_output/detections.json \\
        --manifest /path/to/ortho_tiles_1024/tile_manifest.csv \\
        --output /path/to/ambel_ortho_detections.gpkg \\
        --crs EPSG:32719

    # From summary.json (per-image counts only)
    python tools/04_orthomosaic/detections_to_geopackage.py \\
        --detections /path/to/sahi_output/summary.json \\
        --manifest /path/to/tile_manifest.csv \\
        --output /path/to/detections.gpkg

    # With confidence filter and custom tile size
    python tools/04_orthomosaic/detections_to_geopackage.py \\
        --detections /path/to/detections.json \\
        --manifest /path/to/tile_manifest.csv \\
        --output /path/to/filtered.gpkg \\
        --conf-threshold 0.35 --tile-size 640
        """,
    )

    parser.add_argument(
        "--detections",
        type=str,
        required=True,
        help="Path to SAHI detections.json or summary.json",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to tile_manifest.csv (from tile_orthomosaic.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output GeoPackage path (.gpkg)",
    )
    parser.add_argument(
        "--crs",
        type=str,
        default="EPSG:32719",
        help="Coordinate reference system (default: EPSG:32719 = UTM 19S)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.0,
        help="Minimum confidence to include (default: 0.0 = all)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile size in pixels for coordinate transform (default: 1024)",
    )

    args = parser.parse_args()

    print_banner()

    # Validate inputs
    detections_path = Path(args.detections)
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    if not detections_path.exists():
        print(f"ERROR: Detections file not found: {detections_path}")
        sys.exit(1)

    if not manifest_path.exists():
        print(f"ERROR: Manifest file not found: {manifest_path}")
        sys.exit(1)

    # Ensure output has .gpkg extension
    if output_path.suffix.lower() != ".gpkg":
        output_path = output_path.with_suffix(".gpkg")

    start_time = datetime.now()

    print(f"  Detections:      {detections_path}")
    print(f"  Manifest:        {manifest_path}")
    print(f"  Output:          {output_path}")
    print(f"  CRS:             {args.crs}")
    print(f"  Conf threshold:  {args.conf_threshold}")
    print(f"  Tile size:       {args.tile_size} px")

    # ====================================================================
    # STEP 1: Load tile manifest (for extent display)
    # ====================================================================
    print_section("[1/3] LOADING TILE MANIFEST")

    manifest = load_tile_manifest(str(manifest_path))
    print(f"  Loaded {len(manifest)} tile entries")

    all_x_min = min(t["x_min"] for t in manifest.values())
    all_y_min = min(t["y_min"] for t in manifest.values())
    all_x_max = max(t["x_max"] for t in manifest.values())
    all_y_max = max(t["y_max"] for t in manifest.values())
    print(f"  Extent: ({all_x_min:.2f}, {all_y_min:.2f}) - ({all_x_max:.2f}, {all_y_max:.2f})")

    # ====================================================================
    # STEP 2: Create GeoPackage via library
    # ====================================================================
    print_section("[2/3] CREATING GEOPACKAGE")

    gpkg_path = detections_to_geopackage(
        detections_path=str(detections_path),
        manifest_path=str(manifest_path),
        output_path=str(output_path),
        crs=args.crs,
        conf_threshold=args.conf_threshold,
        tile_size=args.tile_size,
    )

    if gpkg_path is None:
        print("ERROR: Failed to create GeoPackage (no valid records found)")
        print("  Check that tile filenames match between detections JSON and manifest CSV.")
        sys.exit(1)

    # Read back and show info
    try:
        import geopandas as gpd
        gdf = gpd.read_file(str(gpkg_path))
        print(f"  Created: {gpkg_path}")
        print(f"  Features: {len(gdf)}")
        print(f"  CRS: {gdf.crs}")
        print(f"  Columns: {list(gdf.columns)}")
    except ImportError:
        print(f"  Created: {gpkg_path}")

    # ====================================================================
    # STEP 3: Summary statistics
    # ====================================================================
    print_section("[3/3] DETECTION SUMMARY")

    # Determine format and print stats from the GeoPackage
    with open(detections_path) as f:
        det_data = json.load(f)

    source_format = "summary"
    if "images" in det_data:
        for val in det_data["images"].values():
            if isinstance(val, dict) and "detections" in val:
                source_format = "detections"
                break

    try:
        gdf = gpd.read_file(str(gpkg_path))
        n_features = len(gdf)

        if source_format == "detections":
            print(f"  Total detections: {n_features:,}")
            if "confidence" in gdf.columns:
                confs = gdf["confidence"].dropna()
                if len(confs) > 0:
                    print("  Confidence stats:")
                    print(f"    Min:    {confs.min():.4f}")
                    print(f"    Median: {confs.median():.4f}")
                    print(f"    Mean:   {confs.mean():.4f}")
                    print(f"    Max:    {confs.max():.4f}")
            if "class_name" in gdf.columns:
                print("  By class:")
                for cls, count in gdf["class_name"].value_counts().items():
                    print(f"    {cls:20s}: {count:,}")
        elif source_format == "summary":
            print(f"  Total tiles: {n_features}")
            if "total_detections" in gdf.columns:
                total_dets = int(gdf["total_detections"].sum())
                print(f"  Total detections: {total_dets:,}")
            # Print per-class totals from nr_* columns
            nr_cols = [c for c in gdf.columns if c.startswith("nr_")]
            if nr_cols:
                print("  By class:")
                for col in nr_cols:
                    total = int(gdf[col].sum())
                    print(f"    {col:20s}: {total:,}")
    except Exception:
        pass

    # Save summary JSON
    elapsed = (datetime.now() - start_time).total_seconds()
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "detections_path": str(detections_path),
        "manifest_path": str(manifest_path),
        "output_path": str(gpkg_path),
        "crs": args.crs,
        "conf_threshold": args.conf_threshold,
        "tile_size": args.tile_size,
        "source_format": source_format,
        "elapsed_seconds": round(elapsed, 1),
    }
    summary_path = output_path.parent / "detection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary saved: {summary_path}")

    # ====================================================================
    # FINAL SUMMARY
    # ====================================================================
    print()
    print("=" * 70)
    print("  GEOPACKAGE EXPORT COMPLETE")
    print("=" * 70)
    print(f"  GeoPackage:    {gpkg_path}")
    print(f"  CRS:           {args.crs}")
    print(f"  Format:        {source_format}")
    print(f"  Elapsed:       {elapsed:.1f} s")

    file_size = gpkg_path.stat().st_size
    if file_size < 1024:
        size_str = f"{file_size} B"
    elif file_size < 1024 ** 2:
        size_str = f"{file_size / 1024:.1f} KB"
    else:
        size_str = f"{file_size / 1024 ** 2:.1f} MB"
    print(f"  File size:     {size_str}")
    print("=" * 70)


if __name__ == "__main__":
    main()
