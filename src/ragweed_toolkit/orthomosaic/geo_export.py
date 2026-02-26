"""
Geo Export — Detections to GeoPackage
======================================

Convert pixel-coordinate YOLO/SAHI detections back to geographic
coordinates using tile affine transforms from tile_manifest.csv.
Output is a GeoPackage point layer suitable for QGIS or GIS analysis.

Functions
---------
load_tile_manifest
    Load tile_manifest.csv into a lookup dictionary.
detections_to_geopackage
    Convert SAHI detection results to a georeferenced GeoPackage.
compute_detection_summary
    Compute summary statistics for detection records.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def load_tile_manifest(manifest_path: str) -> Dict[str, Dict]:
    """Load tile_manifest.csv into a lookup dictionary.

    Args:
        manifest_path: Path to ``tile_manifest.csv`` from :func:`tile_orthomosaic`.

    Returns:
        Dict mapping filename to ``{x_min, y_min, x_max, y_max, row, col}``.
    """
    df = pd.read_csv(manifest_path)
    required = {"filename", "x_min", "y_min", "x_max", "y_max", "row", "col"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")

    manifest: Dict[str, Dict] = {}
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


def _detect_json_format(data: dict) -> str:
    """Detect whether JSON is detections.json or summary.json format."""
    if "images" in data:
        for val in data["images"].values():
            if isinstance(val, dict) and "detections" in val:
                return "detections"
    if "per_image_results" in data or "per_image" in data:
        return "summary"
    return "detections"


def _load_detections_records(
    data: dict,
    manifest: Dict[str, Dict],
    tile_size: int,
    conf_threshold: float,
) -> List[Dict]:
    """Build records from detections.json format."""
    records: List[Dict] = []
    for image_name, image_info in data.get("images", {}).items():
        tile_info = manifest.get(image_name)
        if tile_info is None:
            continue

        img_w = image_info.get("width", tile_size)
        img_h = image_info.get("height", tile_size)
        tile_w_m = tile_info["x_max"] - tile_info["x_min"]
        tile_h_m = tile_info["y_max"] - tile_info["y_min"]

        for det in image_info.get("detections", []):
            confidence = float(det.get("confidence", 0))
            if confidence < conf_threshold:
                continue

            bbox = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = (float(c) for c in bbox)
            cx_px = (x1 + x2) / 2
            cy_px = (y1 + y2) / 2

            utm_x = tile_info["x_min"] + (cx_px / img_w) * tile_w_m
            utm_y = tile_info["y_max"] - (cy_px / img_h) * tile_h_m

            records.append({
                "utm_x": utm_x,
                "utm_y": utm_y,
                "confidence": round(confidence, 4),
                "class_id": int(det.get("class_id", 0)),
                "class_name": det.get("class_name", "unknown"),
                "tile_filename": image_name,
                "tile_row": tile_info["row"],
                "tile_col": tile_info["col"],
                "bbox_width_m": round((x2 - x1) / img_w * tile_w_m, 4),
                "bbox_height_m": round((y2 - y1) / img_h * tile_h_m, 4),
                "source_format": "detections",
            })
    return records


def _load_summary_records(
    data: dict,
    manifest: Dict[str, Dict],
) -> List[Dict]:
    """Build records from summary.json format."""
    per_image: Dict[str, Dict] = {}

    if "per_image_results" in data:
        for item in data["per_image_results"]:
            name = item.get("image", "")
            counts = item.get("sahi_by_class", {})
            per_image[name] = {"counts": counts, "total": item.get("sahi_total", sum(counts.values()))}
    elif "per_image" in data:
        for name, counts in data["per_image"].items():
            per_image[name] = {"counts": counts, "total": sum(counts.values())}

    records: List[Dict] = []
    for image_name, info in per_image.items():
        tile_info = manifest.get(image_name)
        if tile_info is None:
            continue

        utm_x = (tile_info["x_min"] + tile_info["x_max"]) / 2
        utm_y = (tile_info["y_min"] + tile_info["y_max"]) / 2

        record = {
            "utm_x": utm_x,
            "utm_y": utm_y,
            "confidence": None,
            "class_name": "summary",
            "class_id": -1,
            "tile_filename": image_name,
            "tile_row": tile_info["row"],
            "tile_col": tile_info["col"],
            "total_detections": info["total"],
            "source_format": "summary",
        }
        for cls_name, cls_count in info["counts"].items():
            col_name = f"nr_{cls_name.lower().replace('-', '_')}"
            record[col_name] = int(cls_count)
        records.append(record)

    return records


def detections_to_geopackage(
    detections_path: str,
    manifest_path: str,
    output_path: str,
    *,
    crs: str = "EPSG:32719",
    conf_threshold: float = 0.0,
    tile_size: int = 1024,
) -> Optional[Path]:
    """Convert SAHI detections + tile manifest to a GeoPackage.

    Supports two SAHI output formats:
      - ``detections.json``: Per-detection bboxes with pixel coords.
        Each detection becomes a UTM point at the bbox center.
      - ``summary.json``: Per-image counts. One point per tile centroid.

    Args:
        detections_path: Path to ``detections.json`` or ``summary.json``.
        manifest_path: Path to ``tile_manifest.csv``.
        output_path: Output ``.gpkg`` file path.
        crs: Coordinate reference system (default UTM 19S).
        conf_threshold: Minimum confidence to include (detections format).
        tile_size: Tile size in pixels for coordinate transform.

    Returns:
        Path to created GeoPackage or ``None`` on failure.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        raise ImportError("geopandas is required: pip install geopandas")

    manifest = load_tile_manifest(manifest_path)

    with open(detections_path) as f:
        det_data = json.load(f)

    fmt = _detect_json_format(det_data)

    if fmt == "detections":
        records = _load_detections_records(det_data, manifest, tile_size, conf_threshold)
    else:
        records = _load_summary_records(det_data, manifest)

    if not records:
        return None

    geometries = [Point(r["utm_x"], r["utm_y"]) for r in records]
    clean = [{k: v for k, v in r.items() if k not in ("utm_x", "utm_y")} for r in records]

    gdf = gpd.GeoDataFrame(clean, geometry=geometries, crs=crs)
    out = Path(output_path)
    if out.suffix.lower() != ".gpkg":
        out = out.with_suffix(".gpkg")
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(out), driver="GPKG")

    return out


def compute_detection_summary(records: List[Dict]) -> Dict:
    """Compute summary statistics for detection records.

    Args:
        records: List of detection record dicts.

    Returns:
        Summary dict with counts, confidence stats, and class breakdown.
    """
    if not records:
        return {"total_records": 0}

    fmt = records[0].get("source_format", "detections")
    stats: Dict = {"total_records": len(records), "source_format": fmt}

    if fmt == "detections":
        class_counts: Dict[str, int] = {}
        confidences: List[float] = []
        for r in records:
            cls = r.get("class_name", "unknown")
            class_counts[cls] = class_counts.get(cls, 0) + 1
            c = r.get("confidence")
            if c is not None:
                confidences.append(c)
        stats["by_class"] = class_counts
        if confidences:
            arr = np.array(confidences)
            stats["confidence"] = {
                "min": round(float(arr.min()), 4),
                "max": round(float(arr.max()), 4),
                "mean": round(float(arr.mean()), 4),
                "median": round(float(np.median(arr)), 4),
            }
        stats["unique_tiles"] = len({r.get("tile_filename") for r in records})
    else:
        totals = [r.get("total_detections", 0) for r in records]
        stats["total_detections"] = int(sum(totals))
        stats["tiles_with_detections"] = sum(1 for t in totals if t > 0)

    return stats
