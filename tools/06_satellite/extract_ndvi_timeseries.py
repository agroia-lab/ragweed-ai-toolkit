#!/usr/bin/env python3
"""
Extract NDVI Time Series Statistics for Orobanche Zone Management.

This script extracts Sentinel-2 NDVI time series statistics for crop-season analysis,
supporting site-specific orobanche management via fertigation zones.

Key features:
- Crop season window (Oct-Mar) instead of calendar year
- Multiple statistics: min, max, mean, std, slope, etc.
- Cloud masking using SCL band
- Clustering for zone delineation
- Ready for correlation with YOLO detection data

Data source: Sentinel-2 SR Harmonized (COPERNICUS/S2_SR_HARMONIZED)
- Consistent sensor calibration
- 10m resolution (B4, B8)
- 5-day revisit time
- Atmospherically corrected

Usage:
    # Season 24-25 (complete data available)
    python scripts/satellite/extract_ndvi_timeseries.py --season 24-25 --paddock all

    # Season 25-26 (partial - up to current date)
    python scripts/satellite/extract_ndvi_timeseries.py --season 25-26 --paddock Apalta

    # Custom date range
    python scripts/satellite/extract_ndvi_timeseries.py --start 2024-10-01 --end 2025-02-28 --paddock all

    # With clustering
    python scripts/satellite/extract_ndvi_timeseries.py --season 24-25 --paddock all --cluster 3 5 7

Output:
    ${INIA_EXTERNAL_DRIVE}/outputs_sugal_25-26/satellite_rasters/ndvi_stats/
    ├── ndvi_stats_24-25_Apalta.tif          (multi-band GeoTIFF from Earth Engine)
    └── ...

    ${INIA_EXTERNAL_DRIVE}/outputs_sugal_25-26/satellite_embeddings/
    ├── ndvi_timeseries_24-25_all.gpkg       (grid with statistics + clusters)
    └── ...

Statistics calculated per pixel:
    Basic: min, max, mean, median, std, range, count, cv
    Temporal: date_max, slope, early, peak, late, decline
"""

import argparse
import importlib.util
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union

RASTERIO_AVAILABLE = importlib.util.find_spec("rasterio") is not None
SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None

# Project paths
_script_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _script_dir.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.utils.paths import get_external_drive, get_paths

# --- Library imports (replacing internal duplicates) ---
from ragweed_toolkit.satellite.ndvi import (
    CROP_SEASON_WINDOWS,
    create_risk_zones,
    extract_raster_to_grid,
)
from ragweed_toolkit.satellite.ndvi import (
    export_ndvi_stats as _lib_export_ndvi_stats,
)

# Get paths from config
_paths = get_paths()
_external_base = Path(get_external_drive())

# Output directories
RASTER_OUTPUT_DIR = _external_base / "outputs_sugal_25-26" / "satellite_rasters" / "ndvi_stats"
VECTOR_OUTPUT_DIR = _external_base / "outputs_sugal_25-26" / "satellite_embeddings"
BOUNDARIES_PATH = _external_base / "outputs_sugal_25-26" / "satellite_embeddings" / "paddock_boundaries.gpkg"

# Earth Engine project
EE_PROJECT = "gen-lang-client-0195046178"

# Grid settings
GRID_SIZE = 10  # meters (matches Sentinel-2 resolution)

# Use library season windows
SEASON_WINDOWS = CROP_SEASON_WINDOWS

# Sub-season periods for temporal statistics
SUBSEASON_PERIODS = {
    "early": (10, 11),      # Oct-Nov: Establishment
    "peak": (12, 1),        # Dec-Jan: Peak growth
    "late": (2, 3),         # Feb-Mar: Senescence
}


def initialize_ee() -> bool:
    """Initialize Earth Engine with project."""
    try:
        ee.Initialize(project=EE_PROJECT)
        print(f"Earth Engine initialized (project: {EE_PROJECT})")
        return True
    except Exception as e:
        print(f"Earth Engine initialization failed: {e}")
        print("Run: earthengine authenticate")
        return False


def load_paddock_boundaries(
    paddock_name: str = "all",
    boundaries_path: Path = None
) -> gpd.GeoDataFrame:
    """Load paddock boundaries from GeoPackage."""
    bpath = boundaries_path or BOUNDARIES_PATH

    if not bpath.exists():
        print(f"Error: Boundaries file not found: {bpath}")
        sys.exit(1)

    print(f"Loading boundaries from: {bpath}")
    gdf = gpd.read_file(bpath)

    # Detect paddock field
    paddock_field = None
    for field in ['paddock', 'Nombre Predio', 'Subparcela', 'name']:
        if field in gdf.columns:
            paddock_field = field
            break

    if paddock_field and paddock_field != 'paddock':
        gdf['paddock'] = gdf[paddock_field]

    if paddock_name.lower() != "all":
        gdf = gdf[gdf['paddock'].str.lower().str.contains(paddock_name.lower())]
        if len(gdf) == 0:
            available = gpd.read_file(bpath)
            if 'paddock' in available.columns:
                print(f"Error: Paddock '{paddock_name}' not found.")
                print(f"Available: {', '.join(available['paddock'].unique())}")
            sys.exit(1)

    # Ensure UTM 19S
    if gdf.crs is None or gdf.crs.to_epsg() != 32719:
        gdf = gdf.to_crs(epsg=32719)

    print(f"Loaded {len(gdf)} polygon(s) for: {', '.join(gdf['paddock'].unique())}")
    return gdf


def get_paddock_geometry_ee(gdf: gpd.GeoDataFrame) -> ee.Geometry:
    """Convert GeoDataFrame to Earth Engine geometry."""
    # Convert to WGS84 for Earth Engine
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    merged = unary_union(gdf_wgs84.geometry)

    # Handle MultiPolygon
    if merged.geom_type == 'MultiPolygon':
        # Use convex hull for simplicity, or largest polygon
        merged = merged.convex_hull

    coords = [[c[0], c[1]] for c in merged.exterior.coords]
    return ee.Geometry.Polygon([coords])


# --- Thin CLI wrapper that adds logging around the library export function ---
def export_ndvi_stats_to_drive(
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    paddock_name: str,
    season_label: str,
    folder: str = "ndvi_stats_exports"
) -> ee.batch.Task:
    """
    Export NDVI statistics to Google Drive (wrapper around library function).

    Returns the export task.
    """
    print(f"\n{'='*60}")
    print(f"Exporting NDVI statistics: {paddock_name}")
    print(f"{'='*60}")
    print(f"  Period: {start_date} to {end_date}")

    print("  Computing statistics, slope, and date of maximum...")
    task = _lib_export_ndvi_stats(
        geometry, start_date, end_date, paddock_name,
        drive_folder=folder,
        season_label=season_label,
    )

    safe_name = paddock_name.replace(" ", "_").replace(",", "")
    filename = f"ndvi_stats_{season_label}_{safe_name}"
    print(f"  Export task started: {filename}")
    print(f"  Task ID: {task.id}")
    print(f"  Status: {task.status()['state']}")

    return task


# TODO: Library has ragweed_toolkit.satellite.ndvi._generate_grid (private).
# The generate_grid function below is script-specific but equivalent.
def generate_grid(geometry, cell_size: int = 10) -> gpd.GeoDataFrame:
    """Generate a grid of cells over a geometry."""
    minx, miny, maxx, maxy = geometry.bounds

    cells = []
    cell_ids = []
    cell_id = 0

    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cell = box(x, y, x + cell_size, y + cell_size)
            if cell.intersects(geometry):
                clipped = cell.intersection(geometry)
                if not clipped.is_empty:
                    cells.append(clipped)
                    cell_ids.append(cell_id)
                    cell_id += 1
            y += cell_size
        x += cell_size

    return gpd.GeoDataFrame({
        'cell_id': cell_ids,
        'geometry': cells
    }, crs="EPSG:32719")


def apply_clustering(
    gdf: gpd.GeoDataFrame,
    n_clusters: List[int] = [3, 5, 7],
    features: List[str] = None
) -> gpd.GeoDataFrame:
    """
    Apply KMeans clustering on NDVI statistics.

    Args:
        gdf: GeoDataFrame with NDVI statistic columns
        n_clusters: List of cluster counts to compute
        features: Columns to use for clustering (default: key statistics)

    Returns:
        GeoDataFrame with added cluster columns
    """
    if not SKLEARN_AVAILABLE:
        print("Warning: scikit-learn not available, skipping clustering")
        return gdf

    # Default features for clustering
    if features is None:
        features = ['ndvi_max', 'ndvi_std', 'ndvi_slope', 'ndvi_decline']

    # Filter to available features
    available_features = [f for f in features if f in gdf.columns]
    if not available_features:
        print("Warning: No clustering features available")
        return gdf

    print(f"\nApplying clustering on: {', '.join(available_features)}")

    # Get valid rows
    valid_mask = gdf[available_features[0]].notna()
    for f in available_features[1:]:
        valid_mask = valid_mask & gdf[f].notna()

    if valid_mask.sum() < max(n_clusters):
        print(f"Warning: Not enough valid cells ({valid_mask.sum()}) for clustering")
        return gdf

    # Extract features
    X = gdf.loc[valid_mask, available_features].values

    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cluster for each k
    from sklearn.cluster import KMeans
    for k in n_clusters:
        col_name = f'cluster_{k}'
        gdf[col_name] = np.nan

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        gdf.loc[valid_mask, col_name] = labels.astype(int)
        print(f"  Added cluster_{k} ({k} zones)")

    return gdf


def list_available_tasks() -> None:
    """List pending/running Earth Engine tasks."""
    tasks = ee.batch.Task.list()
    active = [t for t in tasks if t.status()['state'] in ['READY', 'RUNNING']]

    if active:
        print("\nActive Earth Engine tasks:")
        for t in active:
            status = t.status()
            print(f"  {status['description']}: {status['state']}")
    else:
        print("\nNo active Earth Engine tasks")


def main():
    parser = argparse.ArgumentParser(
        description="Extract NDVI time series statistics for orobanche zone management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export statistics for season 24-25 (complete data)
  python extract_ndvi_timeseries.py --season 24-25 --paddock all --export

  # Custom date range
  python extract_ndvi_timeseries.py --start 2024-10-01 --end 2025-01-31 --paddock Apalta --export

  # Process existing raster (after downloading from Drive)
  python extract_ndvi_timeseries.py --raster ndvi_stats_24-25_Apalta.tif --paddock Apalta --cluster 3 5 7

  # Full pipeline: export, wait, then process
  python extract_ndvi_timeseries.py --season 24-25 --paddock all --export --wait --cluster 3 5 7

Available seasons:
  25-26: 2025-10-01 to 2026-03-31 (current, partial data)
  24-25: 2024-10-01 to 2025-03-31 (complete)
  23-24, 22-23, 21-22, 20-21 (historical)
"""
    )

    # Input options
    parser.add_argument("--season", type=str, choices=list(SEASON_WINDOWS.keys()),
                        help="Crop season (Oct-Mar)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--paddock", type=str, default="all",
                        help="Paddock name or 'all'")
    parser.add_argument("--boundaries", type=str, help="Custom boundaries file")

    # Processing options
    parser.add_argument("--export", action="store_true",
                        help="Export to Google Drive (Earth Engine)")
    parser.add_argument("--wait", action="store_true",
                        help="Wait for export tasks to complete")
    parser.add_argument("--raster", type=str,
                        help="Process existing raster file (skip export)")
    parser.add_argument("--raster-dir", type=str,
                        help="Directory with downloaded rasters")

    # Analysis options
    parser.add_argument("--cluster", type=int, nargs='*', default=None,
                        help="Apply clustering (e.g., --cluster 3 5 7)")
    parser.add_argument("--risk-zones", action="store_true",
                        help="Create orobanche risk zone classification")
    parser.add_argument("--output", type=str,
                        help="Output GeoPackage path")

    # Utility options
    parser.add_argument("--list-tasks", action="store_true",
                        help="List active Earth Engine tasks and exit")
    parser.add_argument("--cloud-pct", type=int, default=30,
                        help="Maximum cloud percentage (default: 30)")

    args = parser.parse_args()

    # Initialize Earth Engine
    if not initialize_ee():
        sys.exit(1)

    # List tasks mode
    if args.list_tasks:
        list_available_tasks()
        sys.exit(0)

    # Determine date range
    if args.start and args.end:
        start_date = args.start
        end_date = args.end
        season_label = f"{start_date[:4]}-{end_date[2:4]}"
    elif args.season:
        start_date, end_date = SEASON_WINDOWS[args.season]
        season_label = args.season
    elif args.raster:
        # Extract from filename if possible
        start_date, end_date = "unknown", "unknown"
        season_label = "custom"
    else:
        parser.print_help()
        print("\nError: Specify --season, --start/--end, or --raster")
        sys.exit(1)

    # Check if end date is in the future
    today = datetime.now().strftime("%Y-%m-%d")
    if end_date > today:
        print(f"Note: End date {end_date} is in the future. Using {today} instead.")
        end_date = today

    print("=" * 60)
    print("NDVI TIME SERIES EXTRACTION")
    print("=" * 60)
    print(f"Season: {season_label}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Paddock: {args.paddock}")
    print(f"Cloud filter: <{args.cloud_pct}%")
    if args.cluster:
        print(f"Clustering: K={', '.join(map(str, args.cluster))}")
    if args.risk_zones:
        print("Risk zones: enabled")
    print()

    # Load boundaries
    boundaries = load_paddock_boundaries(
        args.paddock,
        Path(args.boundaries) if args.boundaries else None
    )

    # Export mode
    if args.export:
        print("\n" + "=" * 60)
        print("EXPORTING TO GOOGLE DRIVE")
        print("=" * 60)

        tasks = []
        for paddock_name in boundaries['paddock'].unique():
            paddock_gdf = boundaries[boundaries['paddock'] == paddock_name]
            geometry = get_paddock_geometry_ee(paddock_gdf)

            task = export_ndvi_stats_to_drive(
                geometry=geometry,
                start_date=start_date,
                end_date=end_date,
                paddock_name=paddock_name,
                season_label=season_label
            )
            tasks.append((paddock_name, task))

        print(f"\n{len(tasks)} export task(s) started")
        print("Files will be exported to Google Drive folder: 'ndvi_stats_exports'")
        print("\nTo check task status:")
        print("  https://code.earthengine.google.com/tasks")
        print("  python extract_ndvi_timeseries.py --list-tasks")

        if args.wait:
            print("\nWaiting for tasks to complete...")
            for paddock_name, task in tasks:
                while task.status()['state'] in ['READY', 'RUNNING']:
                    print(f"  {paddock_name}: {task.status()['state']}")
                    time.sleep(30)
                print(f"  {paddock_name}: {task.status()['state']}")

        if not args.raster and not args.raster_dir:
            print("\nAfter downloading, run again with --raster-dir to process:")
            print(f"  python extract_ndvi_timeseries.py --season {season_label} "
                  f"--paddock {args.paddock} --raster-dir /path/to/downloads --cluster 3 5 7")
            sys.exit(0)

    # Process rasters
    all_results = []

    if args.raster:
        # Single raster file
        raster_path = Path(args.raster)
        if not raster_path.exists():
            print(f"Error: Raster not found: {raster_path}")
            sys.exit(1)

        for paddock_name in boundaries['paddock'].unique():
            paddock_gdf = boundaries[boundaries['paddock'] == paddock_name]
            result = extract_raster_to_grid(
                str(raster_path), paddock_gdf, paddock_name,
                grid_size=GRID_SIZE,
            )
            if len(result) > 0:
                all_results.append(result)

    elif args.raster_dir:
        # Directory with multiple rasters
        raster_dir = Path(args.raster_dir)
        if not raster_dir.exists():
            print(f"Error: Directory not found: {raster_dir}")
            sys.exit(1)

        for paddock_name in boundaries['paddock'].unique():
            safe_name = paddock_name.replace(" ", "_").replace(",", "")
            pattern = f"ndvi_stats_{season_label}_{safe_name}.tif"
            raster_files = list(raster_dir.glob(pattern))

            if not raster_files:
                # Try without season label
                raster_files = list(raster_dir.glob(f"*{safe_name}*.tif"))

            if raster_files:
                raster_path = raster_files[0]
                print(f"\nProcessing: {raster_path.name}")
                paddock_gdf = boundaries[boundaries['paddock'] == paddock_name]
                result = extract_raster_to_grid(
                    str(raster_path), paddock_gdf, paddock_name,
                    grid_size=GRID_SIZE,
                )
                if len(result) > 0:
                    all_results.append(result)
            else:
                print(f"\nWarning: No raster found for {paddock_name}")

    if not all_results:
        if not args.export:
            print("\nNo rasters to process. Use --export to generate them first.")
        sys.exit(0)

    # Combine results
    combined = pd.concat(all_results, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, crs="EPSG:32719")

    # Add metadata
    combined['season'] = season_label
    combined['period'] = f"{start_date} to {end_date}"
    combined['extraction_date'] = datetime.now().isoformat()

    # Apply clustering
    if args.cluster:
        clusters = args.cluster if args.cluster else [3, 5, 7]
        combined = apply_clustering(combined, clusters)

    # Create risk zones (using library function)
    if args.risk_zones:
        print("\nCreating orobanche risk zones...")
        combined = create_risk_zones(combined)
        # Print summary
        for zone in ['Low', 'Medium', 'High']:
            count = (combined['risk_zone'] == zone).sum()
            print(f"  {zone} risk: {count} cells")

    # Add X, Y coordinates (centroid of each cell)
    combined['x'] = combined.geometry.centroid.x
    combined['y'] = combined.geometry.centroid.y

    # Reorder columns to put x, y, paddock at the front (after cell_id)
    priority_cols = ['cell_id', 'paddock', 'x', 'y']
    other_cols = [c for c in combined.columns if c not in priority_cols and c != 'geometry']
    combined = combined[priority_cols + other_cols + ['geometry']]

    # Save output
    if args.output:
        output_path = Path(args.output)
    else:
        VECTOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = VECTOR_OUTPUT_DIR / f"ndvi_timeseries_{season_label}_{args.paddock.lower()}.gpkg"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_file(output_path, driver="GPKG")

    # Summary
    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Total cells: {len(combined)}")
    print(f"Paddocks: {combined['paddock'].nunique()}")

    # Statistics summary
    print()
    print("Statistics summary:")
    stat_cols = [c for c in combined.columns if c.startswith('ndvi_')]
    for col in stat_cols[:8]:  # First 8 stats
        if col in combined.columns:
            vals = combined[col].dropna()
            if len(vals) > 0:
                print(f"  {col}: min={vals.min():.3f}, max={vals.max():.3f}, mean={vals.mean():.3f}")

    # Cluster summary
    cluster_cols = [c for c in combined.columns if c.startswith('cluster_')]
    if cluster_cols:
        print()
        print("Cluster distribution:")
        for col in cluster_cols:
            counts = combined[col].value_counts().sort_index()
            print(f"  {col}: {dict(counts)}")

    print()
    print("To visualize in QGIS:")
    print(f"  1. Add layer: {output_path}")
    print("  2. Style by: ndvi_max (graduated), cluster_5 (categorized)")
    print("  3. Compare with YOLO detection points")

    return combined


if __name__ == "__main__":
    main()
