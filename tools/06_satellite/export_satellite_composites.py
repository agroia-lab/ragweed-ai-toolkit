#!/usr/bin/env python3
"""
Export Satellite Composites for Paddocks
=========================================

Creates GeoTIFF raster images from Sentinel-2:
1. Max NDVI composite (peak vegetation)
2. RGB true color composite (median)

Usage:
    conda activate yolov8_custom
    python scripts/satellite/export_satellite_composites.py --season 25-26

Output:
    ${INIA_EXTERNAL_DRIVE}/outputs_sugal_25-26/satellite_embeddings/composites/
    ├── ndvi_max_25-26_Apalta.tif
    ├── ndvi_max_25-26_La_Capilla.tif
    ├── rgb_median_25-26_Apalta.tif
    └── ...
"""

import argparse
import sys
from pathlib import Path

import ee
import geopandas as gpd

# Project paths - use portable path resolution
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

from scripts.utils.paths import get_paths

from ragweed_toolkit.satellite.composites import (
    SEASON_WINDOWS,
)
from ragweed_toolkit.satellite.composites import (
    export_ndvi_max as _lib_export_ndvi_max,
)
from ragweed_toolkit.satellite.composites import (
    export_rgb_median as _lib_export_rgb_median,
)

# --- Library imports (replacing internal duplicates) ---
from ragweed_toolkit.satellite.composites import (
    get_sentinel2_collection as _lib_get_collection,
)

# Earth Engine project
EE_PROJECT = "gen-lang-client-0195046178"

# Initialize Earth Engine
try:
    ee.Initialize(project=EE_PROJECT)
    print(f"Earth Engine initialized (project: {EE_PROJECT})")
except Exception:
    print("Authenticating Earth Engine...")
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)

# Get paths from config
_paths = get_paths()
_satellite_embeddings = _paths.get('external', {}).get('satellite_embeddings', '/media/malezainia1/LORENZO2/outputs_sugal_25-26/satellite_embeddings')

# Configuration
BASE_PATH = _satellite_embeddings
BOUNDARIES_FILE = f"{BASE_PATH}/paddock_boundaries.gpkg"


def get_paddock_geometry(paddock_name, boundaries_file):
    """Load paddock geometry from GeoPackage."""
    from shapely.geometry import MultiPolygon
    from shapely.ops import unary_union

    gdf = gpd.read_file(boundaries_file)

    # Find all rows matching this paddock (may have multiple polygons)
    matching_rows = []
    for idx, row in gdf.iterrows():
        name = row.get('paddock', row.get('name', ''))
        if paddock_name.lower().replace('_', ' ') in name.lower().replace('_', ' '):
            matching_rows.append(row.geometry)

    if not matching_rows:
        print(f"  WARNING: Paddock '{paddock_name}' not found")
        return None

    # Merge all geometries for this paddock
    merged = unary_union(matching_rows)

    # Convert to WGS84 for Earth Engine
    geom = gpd.GeoSeries([merged], crs=gdf.crs).to_crs("EPSG:4326").iloc[0]

    # Handle MultiPolygon by taking the largest polygon
    if isinstance(geom, MultiPolygon):
        # Get the largest polygon by area
        largest = max(geom.geoms, key=lambda p: p.area)
        geom = largest

    # Convert coordinates to list of [lon, lat] pairs for Earth Engine (ignore Z if present)
    coords = [[coord[0], coord[1]] for coord in geom.exterior.coords]

    try:
        ee_geom = ee.Geometry.Polygon([coords])
        return ee_geom
    except Exception as e:
        print(f"  ERROR creating geometry: {e}")
        # Try bounding box as fallback
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        bbox_coords = [
            [bounds[0], bounds[1]],
            [bounds[2], bounds[1]],
            [bounds[2], bounds[3]],
            [bounds[0], bounds[3]],
            [bounds[0], bounds[1]]
        ]
        ee_geom = ee.Geometry.Polygon([bbox_coords])
        print("  Using bounding box instead")
        return ee_geom


# --- Thin CLI wrappers that add logging around the library functions ---
# The library functions have the signature (geometry, start_date, end_date, paddock_name)
# and do not print progress. These wrappers preserve the original CLI output.

def export_ndvi_max(geometry, start_date, end_date, output_path, paddock_name):
    """Export max NDVI composite as GeoTIFF (wrapper around library function)."""
    print(f"  Exporting Max NDVI for {paddock_name}...")

    # Count images for user feedback (library does this internally but doesn't print)
    collection = _lib_get_collection(geometry, start_date, end_date)
    count = collection.size().getInfo()
    print(f"    Found {count} Sentinel-2 images")

    if count == 0:
        print("    WARNING: No images found!")
        return None

    task = _lib_export_ndvi_max(geometry, start_date, end_date, paddock_name)
    if task:
        print(f"    Export task started: {task.id}")
    return task


def export_rgb_median(geometry, start_date, end_date, output_path, paddock_name):
    """Export median RGB composite as GeoTIFF (wrapper around library function)."""
    print(f"  Exporting RGB Median for {paddock_name}...")

    collection = _lib_get_collection(geometry, start_date, end_date)
    count = collection.size().getInfo()

    if count == 0:
        print("    WARNING: No images found!")
        return None

    task = _lib_export_rgb_median(geometry, start_date, end_date, paddock_name)
    if task:
        print(f"    Export task started: {task.id}")
    return task


def list_paddocks(boundaries_file):
    """List available paddocks (unique names only)."""
    gdf = gpd.read_file(boundaries_file)
    col = 'paddock' if 'paddock' in gdf.columns else 'name'
    paddocks = gdf[col].unique().tolist()
    return paddocks


def main():
    parser = argparse.ArgumentParser(description="Export satellite composites for paddocks")
    parser.add_argument("--season", type=str, default="25-26",
                        choices=list(SEASON_WINDOWS.keys()),
                        help="Season to process")
    parser.add_argument("--paddock", type=str, default="all",
                        help="Paddock name or 'all'")
    parser.add_argument("--boundaries", type=str, default=BOUNDARIES_FILE,
                        help="Path to boundaries GeoPackage")
    parser.add_argument("--type", type=str, default="both",
                        choices=["ndvi", "rgb", "both"],
                        help="Type of composite to export")
    parser.add_argument("--list", action="store_true",
                        help="List available paddocks and exit")

    args = parser.parse_args()

    # List paddocks if requested
    if args.list:
        paddocks = list_paddocks(args.boundaries)
        print("Available paddocks:")
        for p in paddocks:
            print(f"  - {p}")
        return

    # Get date window
    start_date, end_date = SEASON_WINDOWS[args.season]
    print(f"\n{'='*60}")
    print("SATELLITE COMPOSITES EXPORT")
    print(f"{'='*60}")
    print(f"Season: {args.season}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Type: {args.type}")
    print()

    # Get paddocks to process
    if args.paddock.lower() == "all":
        paddocks = list_paddocks(args.boundaries)
    else:
        paddocks = [args.paddock]

    print(f"Paddocks to process: {len(paddocks)}")

    # Output directory
    output_dir = Path(BASE_PATH) / "composites" / args.season
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    print()

    # Process each paddock
    tasks = []
    for paddock in paddocks:
        print(f"\n--- {paddock} ---")

        # Get geometry
        geometry = get_paddock_geometry(paddock, args.boundaries)
        if geometry is None:
            continue

        # Export NDVI
        if args.type in ["ndvi", "both"]:
            task = export_ndvi_max(geometry, start_date, end_date, output_dir, paddock)
            if task:
                tasks.append(task)

        # Export RGB
        if args.type in ["rgb", "both"]:
            task = export_rgb_median(geometry, start_date, end_date, output_dir, paddock)
            if task:
                tasks.append(task)

    print(f"\n{'='*60}")
    print("EXPORT TASKS SUBMITTED")
    print(f"{'='*60}")
    print(f"Total tasks: {len(tasks)}")
    print()
    print("Files will be exported to Google Drive folder: 'satellite_exports'")
    print()
    print("To check task status:")
    print("  https://code.earthengine.google.com/tasks")
    print()
    print("After completion, download from Google Drive and move to:")
    print(f"  {output_dir}")


if __name__ == "__main__":
    main()
