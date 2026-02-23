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

import ee
import geopandas as gpd
import argparse
import os
import sys
from pathlib import Path
import time

# Project paths - use portable path resolution
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root))

from scripts.utils.paths import get_paths

# Earth Engine project
EE_PROJECT = "gen-lang-client-0195046178"

# Initialize Earth Engine
try:
    ee.Initialize(project=EE_PROJECT)
    print(f"Earth Engine initialized (project: {EE_PROJECT})")
except:
    print("Authenticating Earth Engine...")
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)

# Get paths from config
_paths = get_paths()
_satellite_embeddings = _paths.get('external', {}).get('satellite_embeddings', '/media/malezainia1/LORENZO2/outputs_sugal_25-26/satellite_embeddings')

# Configuration
BASE_PATH = _satellite_embeddings
BOUNDARIES_FILE = f"{BASE_PATH}/paddock_boundaries.gpkg"

# Season date windows (peak tomato season: Dec 15 - Jan 15)
SEASON_WINDOWS = {
    "25-26": ("2024-12-15", "2025-01-15"),
    "24-25": ("2023-12-15", "2024-01-15"),
    "22-23": ("2022-12-15", "2023-01-15"),
    "21-22": ("2021-12-15", "2022-01-15"),
    "20-21": ("2020-12-15", "2021-01-15"),
}


def get_paddock_geometry(paddock_name, boundaries_file):
    """Load paddock geometry from GeoPackage."""
    from shapely.geometry import MultiPolygon, Polygon
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
        print(f"  Using bounding box instead")
        return ee_geom


def get_sentinel2_collection(start_date, end_date, geometry):
    """Get cloud-filtered Sentinel-2 collection."""
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    )
    return collection


def compute_ndvi(image):
    """Compute NDVI for Sentinel-2 image."""
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi)


def export_ndvi_max(geometry, start_date, end_date, output_path, paddock_name):
    """Export max NDVI composite as GeoTIFF."""
    print(f"  Exporting Max NDVI for {paddock_name}...")

    collection = get_sentinel2_collection(start_date, end_date, geometry)
    count = collection.size().getInfo()
    print(f"    Found {count} Sentinel-2 images")

    if count == 0:
        print(f"    WARNING: No images found!")
        return None

    # Compute NDVI and get max
    ndvi_collection = collection.map(compute_ndvi)
    ndvi_max = ndvi_collection.select("NDVI").max()

    # Export to Drive first, then download
    task = ee.batch.Export.image.toDrive(
        image=ndvi_max,
        description=f"ndvi_max_{paddock_name}",
        folder="satellite_exports",
        fileNamePrefix=f"ndvi_max_{paddock_name}",
        region=geometry,
        scale=10,
        crs="EPSG:32719",
        maxPixels=1e9
    )

    task.start()
    print(f"    Export task started: {task.id}")
    return task


def export_rgb_median(geometry, start_date, end_date, output_path, paddock_name):
    """Export median RGB composite as GeoTIFF."""
    print(f"  Exporting RGB Median for {paddock_name}...")

    collection = get_sentinel2_collection(start_date, end_date, geometry)
    count = collection.size().getInfo()

    if count == 0:
        print(f"    WARNING: No images found!")
        return None

    # Get median RGB (B4=Red, B3=Green, B2=Blue)
    rgb_median = collection.select(["B4", "B3", "B2"]).median()

    # Scale to 0-255 for visualization (divide by 10000, multiply by 255, cap at 255)
    rgb_vis = rgb_median.divide(3000).multiply(255).clamp(0, 255).toUint8()

    task = ee.batch.Export.image.toDrive(
        image=rgb_vis,
        description=f"rgb_median_{paddock_name}",
        folder="satellite_exports",
        fileNamePrefix=f"rgb_median_{paddock_name}",
        region=geometry,
        scale=10,
        crs="EPSG:32719",
        maxPixels=1e9
    )

    task.start()
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
    print(f"SATELLITE COMPOSITES EXPORT")
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
    print(f"EXPORT TASKS SUBMITTED")
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
