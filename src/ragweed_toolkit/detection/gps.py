"""
GPS Extraction from EXIF Metadata
==================================

Reads GPS coordinates (latitude, longitude, altitude) from image EXIF tags
and returns the results as a GeoDataFrame for spatial analysis.

Used throughout the chapter wherever field images need to be georeferenced
(Sections 3.3, 4.1, 4.3).

Usage::

    from ragweed_toolkit.detection.gps import extract_gps, extract_gps_geodataframe

    # Single image
    coords = extract_gps("field_photo.jpg")
    # {'latitude': -34.12, 'longitude': -70.98, 'altitude': 312.0}

    # Batch -> GeoDataFrame
    gdf = extract_gps_geodataframe(["img1.jpg", "img2.jpg"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS


def _dms_to_decimal(dms: tuple, ref: str) -> float:
    """Convert (degrees, minutes, seconds) + hemisphere reference to decimal degrees."""
    d, m, s = (float(x) for x in dms)
    decimal = d + m / 60.0 + s / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def extract_gps(image_path: Union[str, Path]) -> Optional[Dict[str, float]]:
    """Extract GPS coordinates from a single image's EXIF metadata.

    Uses Pillow to read EXIF tags. Handles images without GPS data
    gracefully by returning ``None``.

    Args:
        image_path: Path to a JPEG/TIFF/HEIC image file.

    Returns:
        Dictionary with ``"latitude"``, ``"longitude"``, and optionally
        ``"altitude"`` (metres above sea level), or ``None`` if the
        image has no GPS EXIF data.
    """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data is None:
            return None

        gps_info: Dict[str, Any] = {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if tag_name == "GPSInfo":
                for gps_tag_id, gps_value in value.items():
                    gps_tag_name = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag_name] = gps_value

        if not gps_info:
            return None

        result: Dict[str, float] = {}

        if "GPSLatitude" in gps_info and "GPSLatitudeRef" in gps_info:
            result["latitude"] = _dms_to_decimal(
                gps_info["GPSLatitude"], gps_info["GPSLatitudeRef"]
            )
        if "GPSLongitude" in gps_info and "GPSLongitudeRef" in gps_info:
            result["longitude"] = _dms_to_decimal(
                gps_info["GPSLongitude"], gps_info["GPSLongitudeRef"]
            )
        if "GPSAltitude" in gps_info:
            result["altitude"] = float(gps_info["GPSAltitude"])

        return result if result else None

    except Exception:
        return None


def extract_gps_geodataframe(
    image_paths: Sequence[Union[str, Path]],
    crs: str = "EPSG:4326",
) -> "geopandas.GeoDataFrame":
    """Extract GPS from multiple images and return a GeoDataFrame.

    Images without GPS data are included with ``geometry=None`` so the
    caller can decide how to handle them (drop, impute, etc.).

    Args:
        image_paths: Sequence of image file paths.
        crs: Coordinate reference system (default WGS 84).

    Returns:
        A :class:`geopandas.GeoDataFrame` with columns:

        - ``filename`` -- Image filename (stem + suffix).
        - ``filepath`` -- Full resolved path.
        - ``latitude``, ``longitude``, ``altitude`` -- Extracted coords
          (NaN when missing).
        - ``has_gps`` -- Boolean flag.
        - ``geometry`` -- Shapely Point or ``None``.
    """
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    records: List[Dict[str, Any]] = []
    for img_path in image_paths:
        img_path = Path(img_path)
        gps = extract_gps(img_path)
        records.append(
            {
                "filename": img_path.name,
                "filepath": str(img_path.resolve()),
                "latitude": gps["latitude"] if gps and "latitude" in gps else None,
                "longitude": gps["longitude"] if gps and "longitude" in gps else None,
                "altitude": gps.get("altitude") if gps else None,
                "has_gps": gps is not None and "latitude" in gps,
            }
        )

    df = pd.DataFrame(records)

    # Build geometry column
    geometry = [
        Point(row["longitude"], row["latitude"]) if row["has_gps"] else None
        for _, row in df.iterrows()
    ]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    return gdf
