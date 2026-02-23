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
    python scripts/cli/detections_to_geopackage.py \\
        --detections /path/to/sahi_output/detections.json \\
        --manifest /path/to/ortho_tiles_1024/tile_manifest.csv \\
        --output /path/to/ambel_ortho_detections.gpkg \\
        --crs EPSG:32719

    # From summary.json (per-image counts, centroid points)
    python scripts/cli/detections_to_geopackage.py \\
        --detections /path/to/sahi_output/summary.json \\
        --manifest /path/to/ortho_tiles_1024/tile_manifest.csv \\
        --output /path/to/ambel_ortho_detections.gpkg

    # With confidence threshold filter
    python scripts/cli/detections_to_geopackage.py \\
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================

try:
    import geopandas as gpd
    from shapely.geometry import Point

    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


# ============================================================================
# BANNER AND OUTPUT HELPERS
# ============================================================================


def print_banner():
    """Print script banner."""
    print("=" * 70)
    print("  DETECTIONS TO GEOPACKAGE - SAHI results to georeferenced points")
    print("=" * 70)
    print()


def print_section(title: str):
    """Print a section header."""
    print()
    print("-" * 70)
    print(f"  {title}")
    print("-" * 70)


# ============================================================================
# MANIFEST LOADING
# ============================================================================


def load_manifest(manifest_path: Path) -> Dict[str, Dict]:
    """
    Load tile_manifest.csv into a lookup dictionary.

    The manifest CSV (from tile_orthomosaic.py) has columns:
        filename, row, col, x_min, y_min, x_max, y_max, n_valid_pixels, frac_valid

    Returns:
        Dict mapping filename -> {x_min, y_min, x_max, y_max, row, col}
    """
    df = pd.read_csv(manifest_path)

    required_cols = {"filename", "x_min", "y_min", "x_max", "y_max", "row", "col"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: Manifest CSV missing columns: {missing}")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)

    manifest = {}
    for _, row in df.iterrows():
        manifest[row["filename"]] = {
            "x_min": float(row["x_min"]),
            "y_min": float(row["y_min"]),
            "x_max": float(row["x_max"]),
            "y_max": float(row["y_max"]),
            "row": int(row["row"]),
            "col": int(row["col"]),
        }

    return manifest


# ============================================================================
# DETECTION LOADING
# ============================================================================


def detect_format(data: dict) -> str:
    """
    Detect whether the JSON is detections.json or summary.json format.

    detections.json has:
        { "images": { "filename.jpg": { "detections": [...], "width": ..., "height": ... } } }

    summary.json has:
        { "per_image_results": [...] } or { "per_image": {...} }

    Returns:
        "detections" or "summary"
    """
    if "images" in data:
        # Check if the images dict contains detection lists
        for key, value in data["images"].items():
            if isinstance(value, dict) and "detections" in value:
                return "detections"
        # If we checked all images and none had "detections" key
        return "summary"

    if "per_image_results" in data or "per_image" in data:
        return "summary"

    # Fallback: if it has "images" with nested data, treat as detections
    if "images" in data:
        return "detections"

    print("WARNING: Could not determine JSON format. Attempting detections format.")
    return "detections"


def load_detections_json(
    data: dict, manifest: Dict[str, Dict], tile_size: int, conf_threshold: float
) -> List[Dict]:
    """
    Load detections from detections.json format (per-detection bounding boxes).

    Each detection has: class_id, class_name, bbox [x1, y1, x2, y2], confidence

    For each detection:
        1. Look up the tile in the manifest to get UTM bounds
        2. Convert the bbox center pixel coordinates to UTM coordinates
        3. Compute bbox dimensions in meters

    Args:
        data: Parsed JSON data
        manifest: Tile manifest dictionary
        tile_size: Tile size in pixels (for coordinate conversion)
        conf_threshold: Minimum confidence to include

    Returns:
        List of record dicts with geometry info
    """
    records = []
    images_data = data.get("images", {})

    n_skipped_no_tile = 0
    n_skipped_conf = 0

    for image_name, image_info in images_data.items():
        detections = image_info.get("detections", [])
        img_width = image_info.get("width", tile_size)
        img_height = image_info.get("height", tile_size)

        # Look up tile in manifest
        tile_info = manifest.get(image_name)
        if tile_info is None:
            # Try without extension
            stem = Path(image_name).stem
            for fname, info in manifest.items():
                if Path(fname).stem == stem:
                    tile_info = info
                    print(f"  WARNING: Matched '{image_name}' via stem (expected exact filename match)")
                    break

        if tile_info is None:
            n_skipped_no_tile += 1
            continue

        tile_x_min = tile_info["x_min"]
        tile_y_min = tile_info["y_min"]
        tile_x_max = tile_info["x_max"]
        tile_y_max = tile_info["y_max"]
        tile_row = tile_info["row"]
        tile_col = tile_info["col"]

        # UTM extent of the tile
        tile_width_m = tile_x_max - tile_x_min
        tile_height_m = tile_y_max - tile_y_min

        for det in detections:
            confidence = float(det.get("confidence", 0))
            if confidence < conf_threshold:
                n_skipped_conf += 1
                continue

            # Bounding box in pixel coordinates [x1, y1, x2, y2]
            bbox = det.get("bbox", [0, 0, 0, 0])
            x1_px, y1_px, x2_px, y2_px = [float(c) for c in bbox]

            # Bbox center in pixels
            center_x_px = (x1_px + x2_px) / 2
            center_y_px = (y1_px + y2_px) / 2

            # Convert pixel center to UTM coordinates
            # x increases left to right, y increases top to bottom in pixels
            # In UTM: x increases left to right, y increases bottom to top
            utm_x = tile_x_min + (center_x_px / img_width) * tile_width_m
            utm_y = tile_y_max - (center_y_px / img_height) * tile_height_m

            # Bbox dimensions in meters
            bbox_width_px = x2_px - x1_px
            bbox_height_px = y2_px - y1_px
            bbox_width_m = (bbox_width_px / img_width) * tile_width_m
            bbox_height_m = (bbox_height_px / img_height) * tile_height_m

            records.append(
                {
                    "utm_x": utm_x,
                    "utm_y": utm_y,
                    "confidence": round(confidence, 4),
                    "class_id": int(det.get("class_id", 0)),
                    "class_name": det.get("class_name", "unknown"),
                    "tile_filename": image_name,
                    "tile_row": tile_row,
                    "tile_col": tile_col,
                    "bbox_width_m": round(bbox_width_m, 4),
                    "bbox_height_m": round(bbox_height_m, 4),
                    "bbox_x1_px": round(x1_px, 1),
                    "bbox_y1_px": round(y1_px, 1),
                    "bbox_x2_px": round(x2_px, 1),
                    "bbox_y2_px": round(y2_px, 1),
                    "source_format": "detections",
                }
            )

    if n_skipped_no_tile > 0:
        print(f"  WARNING: {n_skipped_no_tile} images not found in manifest (skipped)")
    if n_skipped_conf > 0:
        print(f"  Filtered: {n_skipped_conf} detections below confidence {conf_threshold}")

    return records


def load_summary_json(
    data: dict, manifest: Dict[str, Dict]
) -> List[Dict]:
    """
    Load detections from summary.json format (per-image counts only).

    Since summary.json only has counts (not individual bbox positions),
    one point is created per tile at the tile centroid.

    Handles both formats:
        - Old: {"per_image_results": [{"image": ..., "sahi_by_class": {...}, ...}]}
        - New: {"per_image": {"image.jpg": {"CLASS": count, ...}}}

    Args:
        data: Parsed JSON data
        manifest: Tile manifest dictionary

    Returns:
        List of record dicts with geometry info
    """
    records = []
    n_skipped = 0

    # Parse per-image results
    per_image = {}

    if "per_image_results" in data:
        # Old format: list of dicts
        for item in data["per_image_results"]:
            img_name = item.get("image", "")
            counts = item.get("sahi_by_class", {})
            total = item.get("sahi_total", sum(counts.values()))
            per_image[img_name] = {"counts": counts, "total": total}

    elif "per_image" in data:
        # New format: dict of dicts
        for img_name, counts in data["per_image"].items():
            total = sum(counts.values())
            per_image[img_name] = {"counts": counts, "total": total}

    else:
        print("ERROR: summary.json has no 'per_image_results' or 'per_image' key")
        return records

    for image_name, info in per_image.items():
        # Look up tile in manifest
        tile_info = manifest.get(image_name)
        if tile_info is None:
            stem = Path(image_name).stem
            for fname, minfo in manifest.items():
                if Path(fname).stem == stem:
                    tile_info = minfo
                    print(f"  WARNING: Matched '{image_name}' via stem (expected exact filename match)")
                    break

        if tile_info is None:
            n_skipped += 1
            continue

        # Centroid of the tile
        utm_x = (tile_info["x_min"] + tile_info["x_max"]) / 2
        utm_y = (tile_info["y_min"] + tile_info["y_max"]) / 2

        counts = info["counts"]
        total = info["total"]

        # Build class count columns
        record = {
            "utm_x": utm_x,
            "utm_y": utm_y,
            "confidence": None,
            "class_name": "summary",
            "class_id": -1,
            "tile_filename": image_name,
            "tile_row": tile_info["row"],
            "tile_col": tile_info["col"],
            "bbox_width_m": 0,
            "bbox_height_m": 0,
            "total_detections": total,
            "source_format": "summary",
        }

        # Add per-class counts as columns
        for cls_name, cls_count in counts.items():
            col_name = f"nr_{cls_name.lower().replace('-', '_')}"
            record[col_name] = int(cls_count)

        records.append(record)

    if n_skipped > 0:
        print(f"  WARNING: {n_skipped} images not found in manifest (skipped)")

    return records


# ============================================================================
# GEOPACKAGE CREATION
# ============================================================================


def create_geopackage(
    records: List[Dict], output_path: Path, crs: str
) -> Optional[Path]:
    """
    Create a GeoPackage from detection records with UTM point geometries.

    Args:
        records: List of record dicts with 'utm_x' and 'utm_y' keys
        output_path: Path for the output .gpkg file
        crs: Coordinate reference system (e.g., 'EPSG:32719')

    Returns:
        Path to created GeoPackage, or None on failure
    """
    if not records:
        print("ERROR: No records to export")
        return None

    if not HAS_GEOPANDAS:
        print("ERROR: geopandas is required but not installed")
        print("  Install with: pip install geopandas")
        return None

    # Create Point geometries
    geometries = [Point(rec["utm_x"], rec["utm_y"]) for rec in records]

    # Remove raw coordinate columns (geometry handles this)
    clean_records = []
    for rec in records:
        r = dict(rec)
        r.pop("utm_x", None)
        r.pop("utm_y", None)
        clean_records.append(r)

    # Build GeoDataFrame
    gdf = gpd.GeoDataFrame(clean_records, geometry=geometries, crs=crs)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as GeoPackage
    gdf.to_file(str(output_path), driver="GPKG")

    return output_path


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================


def compute_summary_statistics(
    records: List[Dict], source_format: str
) -> Dict:
    """
    Compute summary statistics for the detection records.

    Args:
        records: List of detection record dicts
        source_format: "detections" or "summary"

    Returns:
        Dict with summary statistics
    """
    stats = {
        "total_records": len(records),
        "source_format": source_format,
    }

    if not records:
        return stats

    if source_format == "detections":
        # Per-class counts
        class_counts = {}
        confidences = []
        for rec in records:
            cls = rec.get("class_name", "unknown")
            class_counts[cls] = class_counts.get(cls, 0) + 1
            conf = rec.get("confidence")
            if conf is not None:
                confidences.append(conf)

        stats["by_class"] = class_counts

        if confidences:
            confs = np.array(confidences)
            stats["confidence"] = {
                "min": round(float(confs.min()), 4),
                "max": round(float(confs.max()), 4),
                "mean": round(float(confs.mean()), 4),
                "median": round(float(np.median(confs)), 4),
                "std": round(float(confs.std()), 4),
            }

        # Unique tiles
        unique_tiles = set(rec.get("tile_filename", "") for rec in records)
        stats["unique_tiles"] = len(unique_tiles)

        # Bbox size statistics (in meters)
        widths = [rec.get("bbox_width_m", 0) for rec in records if rec.get("bbox_width_m", 0) > 0]
        heights = [rec.get("bbox_height_m", 0) for rec in records if rec.get("bbox_height_m", 0) > 0]
        if widths:
            stats["bbox_width_m"] = {
                "min": round(min(widths), 4),
                "max": round(max(widths), 4),
                "mean": round(float(np.mean(widths)), 4),
            }
        if heights:
            stats["bbox_height_m"] = {
                "min": round(min(heights), 4),
                "max": round(max(heights), 4),
                "mean": round(float(np.mean(heights)), 4),
            }

    elif source_format == "summary":
        # Summary format: per-tile counts
        totals = [rec.get("total_detections", 0) for rec in records]
        stats["total_detections"] = int(sum(totals))
        stats["tiles_with_detections"] = sum(1 for t in totals if t > 0)
        stats["tiles_without_detections"] = sum(1 for t in totals if t == 0)
        if totals:
            stats["detections_per_tile"] = {
                "min": int(min(totals)),
                "max": int(max(totals)),
                "mean": round(float(np.mean(totals)), 2),
                "median": round(float(np.median(totals)), 2),
            }

        # Aggregate class counts (from nr_* columns)
        class_totals = {}
        for rec in records:
            for key, val in rec.items():
                if key.startswith("nr_") and isinstance(val, (int, float)):
                    class_totals[key] = class_totals.get(key, 0) + int(val)
        if class_totals:
            stats["by_class"] = class_totals

    return stats


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
    python scripts/cli/detections_to_geopackage.py \\
        --detections /path/to/sahi_output/detections.json \\
        --manifest /path/to/ortho_tiles_1024/tile_manifest.csv \\
        --output /path/to/ambel_ortho_detections.gpkg \\
        --crs EPSG:32719

    # From summary.json (per-image counts only)
    python scripts/cli/detections_to_geopackage.py \\
        --detections /path/to/sahi_output/summary.json \\
        --manifest /path/to/tile_manifest.csv \\
        --output /path/to/detections.gpkg

    # With confidence filter and custom tile size
    python scripts/cli/detections_to_geopackage.py \\
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

    # Check dependencies
    if not HAS_GEOPANDAS:
        print("ERROR: geopandas is required but not installed")
        print("  Install with: pip install geopandas")
        sys.exit(1)

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
    # STEP 1: Load tile manifest
    # ====================================================================
    print_section("[1/4] LOADING TILE MANIFEST")

    manifest = load_manifest(manifest_path)
    print(f"  Loaded {len(manifest)} tile entries")

    # Show tile extent
    all_x_min = min(t["x_min"] for t in manifest.values())
    all_y_min = min(t["y_min"] for t in manifest.values())
    all_x_max = max(t["x_max"] for t in manifest.values())
    all_y_max = max(t["y_max"] for t in manifest.values())
    print(f"  Extent: ({all_x_min:.2f}, {all_y_min:.2f}) - ({all_x_max:.2f}, {all_y_max:.2f})")

    # ====================================================================
    # STEP 2: Load detections
    # ====================================================================
    print_section("[2/4] LOADING DETECTIONS")

    with open(detections_path) as f:
        det_data = json.load(f)

    source_format = detect_format(det_data)
    print(f"  Detected format: {source_format}")

    if source_format == "detections":
        records = load_detections_json(
            det_data, manifest, args.tile_size, args.conf_threshold
        )
    elif source_format == "summary":
        records = load_summary_json(det_data, manifest)
    else:
        print(f"ERROR: Unknown detection format")
        sys.exit(1)

    print(f"  Loaded {len(records)} records")

    if not records:
        print("ERROR: No valid records found. Check that tile filenames match between")
        print("       detections JSON and manifest CSV.")
        sys.exit(1)

    # ====================================================================
    # STEP 3: Create GeoPackage
    # ====================================================================
    print_section("[3/4] CREATING GEOPACKAGE")

    gpkg_path = create_geopackage(records, output_path, args.crs)

    if gpkg_path is None:
        print("ERROR: Failed to create GeoPackage")
        sys.exit(1)

    # Read back and show info
    gdf = gpd.read_file(str(gpkg_path))
    print(f"  Created: {gpkg_path}")
    print(f"  Features: {len(gdf)}")
    print(f"  CRS: {gdf.crs}")
    print(f"  Columns: {list(gdf.columns)}")

    # ====================================================================
    # STEP 4: Summary statistics
    # ====================================================================
    print_section("[4/4] DETECTION SUMMARY")

    stats = compute_summary_statistics(records, source_format)

    if source_format == "detections":
        print(f"  Total detections: {stats['total_records']:,}")
        print(f"  Unique tiles:     {stats.get('unique_tiles', 'N/A')}")

        if "by_class" in stats:
            print(f"  By class:")
            for cls, count in sorted(stats["by_class"].items(), key=lambda x: -x[1]):
                print(f"    {cls:20s}: {count:,}")

        if "confidence" in stats:
            conf = stats["confidence"]
            print(f"  Confidence stats:")
            print(f"    Min:    {conf['min']:.4f}")
            print(f"    Median: {conf['median']:.4f}")
            print(f"    Mean:   {conf['mean']:.4f}")
            print(f"    Max:    {conf['max']:.4f}")

        if "bbox_width_m" in stats:
            bw = stats["bbox_width_m"]
            bh = stats["bbox_height_m"]
            print(f"  Bbox size (meters):")
            print(f"    Width:  {bw['min']:.3f} - {bw['max']:.3f} (mean {bw['mean']:.3f})")
            print(f"    Height: {bh['min']:.3f} - {bh['max']:.3f} (mean {bh['mean']:.3f})")

    elif source_format == "summary":
        print(f"  Total tiles:       {stats['total_records']}")
        print(f"  Total detections:  {stats.get('total_detections', 0):,}")
        print(f"  Tiles with dets:   {stats.get('tiles_with_detections', 0)}")
        print(f"  Tiles without:     {stats.get('tiles_without_detections', 0)}")

        if "detections_per_tile" in stats:
            dpt = stats["detections_per_tile"]
            print(f"  Per-tile stats:")
            print(f"    Min:    {dpt['min']}")
            print(f"    Median: {dpt['median']}")
            print(f"    Mean:   {dpt['mean']:.2f}")
            print(f"    Max:    {dpt['max']}")

        if "by_class" in stats:
            print(f"  By class:")
            for cls, count in sorted(stats["by_class"].items(), key=lambda x: -x[1]):
                print(f"    {cls:20s}: {count:,}")

    # Save summary JSON
    summary_path = output_path.parent / "detection_summary.json"
    stats["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stats["detections_path"] = str(detections_path)
    stats["manifest_path"] = str(manifest_path)
    stats["output_path"] = str(gpkg_path)
    stats["crs"] = args.crs
    stats["conf_threshold"] = args.conf_threshold
    stats["tile_size"] = args.tile_size

    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")

    # ====================================================================
    # FINAL SUMMARY
    # ====================================================================
    elapsed = (datetime.now() - start_time).total_seconds()

    print()
    print("=" * 70)
    print("  GEOPACKAGE EXPORT COMPLETE")
    print("=" * 70)
    print(f"  GeoPackage:    {gpkg_path}")
    print(f"  Features:      {len(gdf):,}")
    print(f"  CRS:           {args.crs}")
    print(f"  Format:        {source_format}")
    if source_format == "detections" and "by_class" in stats:
        total = sum(stats["by_class"].values())
        print(f"  Detections:    {total:,}")
    elif source_format == "summary":
        print(f"  Total dets:    {stats.get('total_detections', 0):,}")
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
