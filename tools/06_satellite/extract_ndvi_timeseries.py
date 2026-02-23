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
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict
import json
import time

import ee
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box, mapping
from shapely.ops import unary_union
from tqdm import tqdm

# Optional imports
try:
    import rasterio
    from rasterio.mask import mask as rio_mask
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Project paths
_script_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _script_dir.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.paths import get_paths, get_external_drive

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

# Season windows (crop season: Oct 1 - Mar 31)
SEASON_WINDOWS = {
    "25-26": ("2025-10-01", "2026-03-31"),
    "24-25": ("2024-10-01", "2025-03-31"),
    "23-24": ("2023-10-01", "2024-03-31"),
    "22-23": ("2022-10-01", "2023-03-31"),
    "21-22": ("2021-10-01", "2022-03-31"),
    "20-21": ("2020-10-01", "2021-03-31"),
}

# Sub-season periods for temporal statistics
SUBSEASON_PERIODS = {
    "early": (10, 11),      # Oct-Nov: Establishment
    "peak": (12, 1),        # Dec-Jan: Peak growth
    "late": (2, 3),         # Feb-Mar: Senescence
}

# Cloud mask values from SCL band
# 0=No data, 1=Saturated, 2=Dark, 3=Shadow, 4=Vegetation, 5=Bare soil,
# 6=Water, 7=Unclassified, 8=Cloud medium, 9=Cloud high, 10=Cirrus, 11=Snow
SCL_CLEAR_VALUES = [4, 5, 6, 7, 11]  # Vegetation, bare soil, water, unclassified, snow


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


def get_sentinel2_collection(
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    cloud_pct: int = 30
) -> ee.ImageCollection:
    """Get cloud-filtered Sentinel-2 collection."""
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    )
    return collection


def add_ndvi(image: ee.Image) -> ee.Image:
    """Add NDVI band to image."""
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi)


def mask_clouds(image: ee.Image) -> ee.Image:
    """Mask clouds using SCL band."""
    scl = image.select("SCL")
    clear_mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11))
    return image.updateMask(clear_mask)


def add_date_band(image: ee.Image) -> ee.Image:
    """Add date band (days since epoch) for temporal analysis."""
    date = ee.Date(image.get("system:time_start"))
    days = date.difference(ee.Date("1970-01-01"), "day")
    date_band = ee.Image.constant(days).rename("date").toFloat()
    return image.addBands(date_band)


def add_doy_band(image: ee.Image) -> ee.Image:
    """Add day-of-year band."""
    date = ee.Date(image.get("system:time_start"))
    doy = date.getRelative("day", "year")
    doy_band = ee.Image.constant(doy).rename("doy").toFloat()
    return image.addBands(doy_band)


def add_month_band(image: ee.Image) -> ee.Image:
    """Add month band for sub-season filtering."""
    date = ee.Date(image.get("system:time_start"))
    month = date.get("month")
    month_band = ee.Image.constant(month).rename("month").toInt()
    return image.addBands(month_band)


def compute_ndvi_statistics(
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    cloud_pct: int = 30
) -> ee.Image:
    """
    Compute NDVI time series statistics for a region.

    Returns a multi-band image with:
    - ndvi_min, ndvi_max, ndvi_mean, ndvi_median, ndvi_std
    - ndvi_range, ndvi_count, ndvi_cv
    - ndvi_p10, ndvi_p90 (percentiles)
    - ndvi_early, ndvi_peak, ndvi_late (sub-season means)
    """
    # Get collection
    collection = get_sentinel2_collection(geometry, start_date, end_date, cloud_pct)

    # Process: mask clouds, add NDVI
    processed = (
        collection
        .map(mask_clouds)
        .map(add_ndvi)
        .map(add_month_band)
        .select(["NDVI", "month"])
    )

    # Basic statistics (all cast to Float for consistent export)
    ndvi_min = processed.select("NDVI").reduce(ee.Reducer.min()).toFloat().rename("ndvi_min")
    ndvi_max = processed.select("NDVI").reduce(ee.Reducer.max()).toFloat().rename("ndvi_max")
    ndvi_mean = processed.select("NDVI").reduce(ee.Reducer.mean()).toFloat().rename("ndvi_mean")
    ndvi_median = processed.select("NDVI").reduce(ee.Reducer.median()).toFloat().rename("ndvi_median")
    ndvi_std = processed.select("NDVI").reduce(ee.Reducer.stdDev()).toFloat().rename("ndvi_std")
    ndvi_count = processed.select("NDVI").reduce(ee.Reducer.count()).toFloat().rename("ndvi_count")

    # Percentiles (cast to Float for consistent export)
    ndvi_p10 = processed.select("NDVI").reduce(ee.Reducer.percentile([10])).toFloat().rename("ndvi_p10")
    ndvi_p90 = processed.select("NDVI").reduce(ee.Reducer.percentile([90])).toFloat().rename("ndvi_p90")

    # Derived statistics (already Float from Float operations)
    ndvi_range = ndvi_max.subtract(ndvi_min).toFloat().rename("ndvi_range")
    ndvi_cv = ndvi_std.divide(ndvi_mean).toFloat().rename("ndvi_cv")

    # Sub-season means (Oct-Nov, Dec-Jan, Feb-Mar) - cast to Float
    # Early season: Oct (10), Nov (11)
    early_filter = processed.filter(
        ee.Filter.Or(ee.Filter.eq("month", 10), ee.Filter.eq("month", 11))
    )
    ndvi_early = early_filter.select("NDVI").reduce(ee.Reducer.mean()).toFloat().rename("ndvi_early")

    # Peak season: Dec (12), Jan (1)
    peak_filter = processed.filter(
        ee.Filter.Or(ee.Filter.eq("month", 12), ee.Filter.eq("month", 1))
    )
    ndvi_peak = peak_filter.select("NDVI").reduce(ee.Reducer.mean()).toFloat().rename("ndvi_peak")

    # Late season: Feb (2), Mar (3)
    late_filter = processed.filter(
        ee.Filter.Or(ee.Filter.eq("month", 2), ee.Filter.eq("month", 3))
    )
    ndvi_late = late_filter.select("NDVI").reduce(ee.Reducer.mean()).toFloat().rename("ndvi_late")

    # Decline rate: (peak - late) / peak
    ndvi_decline = ndvi_peak.subtract(ndvi_late).divide(ndvi_peak).toFloat().rename("ndvi_decline")

    # Combine all bands
    result = (
        ndvi_min
        .addBands(ndvi_max)
        .addBands(ndvi_mean)
        .addBands(ndvi_median)
        .addBands(ndvi_std)
        .addBands(ndvi_count)
        .addBands(ndvi_range)
        .addBands(ndvi_cv)
        .addBands(ndvi_p10)
        .addBands(ndvi_p90)
        .addBands(ndvi_early)
        .addBands(ndvi_peak)
        .addBands(ndvi_late)
        .addBands(ndvi_decline)
    )

    return result


def compute_ndvi_slope(
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    cloud_pct: int = 30
) -> ee.Image:
    """
    Compute linear trend (slope) of NDVI over time.

    Uses linear regression: NDVI = slope * time + intercept
    Positive slope = improving, Negative slope = declining
    """
    collection = get_sentinel2_collection(geometry, start_date, end_date, cloud_pct)

    processed = (
        collection
        .map(mask_clouds)
        .map(add_ndvi)
        .map(add_date_band)
        .select(["NDVI", "date"])
    )

    # Linear fit: dependent=NDVI, independent=date
    linear_fit = processed.reduce(ee.Reducer.linearFit())

    # Scale slope to "per month" (multiply by ~30 days) - cast to Float for consistent export
    slope = linear_fit.select("scale").multiply(30).toFloat().rename("ndvi_slope")
    intercept = linear_fit.select("offset").toFloat().rename("ndvi_intercept")

    return slope.addBands(intercept)


def compute_date_of_max(
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    cloud_pct: int = 30
) -> ee.Image:
    """
    Compute the day-of-year when maximum NDVI occurred.

    Early peak (DOY < 365 in Oct-Dec) may indicate stunted growth.
    """
    collection = get_sentinel2_collection(geometry, start_date, end_date, cloud_pct)

    processed = (
        collection
        .map(mask_clouds)
        .map(add_ndvi)
        .map(add_doy_band)
    )

    # Get max NDVI and corresponding DOY
    def add_ndvi_doy(image):
        return image.addBands(
            image.select("doy").multiply(image.select("NDVI").gt(-9999))
        )

    with_doy = processed.map(add_ndvi_doy)

    # Find image with max NDVI using qualityMosaic
    max_image = with_doy.qualityMosaic("NDVI")
    date_of_max = max_image.select("doy").toFloat().rename("ndvi_date_max")

    return date_of_max


def export_ndvi_stats_to_drive(
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    paddock_name: str,
    season_label: str,
    folder: str = "ndvi_stats_exports"
) -> ee.batch.Task:
    """
    Export NDVI statistics to Google Drive.

    Returns the export task.
    """
    print(f"\n{'='*60}")
    print(f"Exporting NDVI statistics: {paddock_name}")
    print(f"{'='*60}")
    print(f"  Period: {start_date} to {end_date}")

    # Compute all statistics
    print("  Computing basic statistics...")
    stats = compute_ndvi_statistics(geometry, start_date, end_date)

    print("  Computing linear trend...")
    slope = compute_ndvi_slope(geometry, start_date, end_date)

    print("  Computing date of maximum...")
    date_max = compute_date_of_max(geometry, start_date, end_date)

    # Combine all
    result = stats.addBands(slope).addBands(date_max)

    # File name
    safe_name = paddock_name.replace(" ", "_").replace(",", "")
    filename = f"ndvi_stats_{season_label}_{safe_name}"

    print(f"  Starting export task: {filename}")

    # Export to Drive
    task = ee.batch.Export.image.toDrive(
        image=result,
        description=filename,
        folder=folder,
        fileNamePrefix=filename,
        region=geometry,
        scale=10,
        crs="EPSG:32719",
        maxPixels=1e9,
        fileFormat="GeoTIFF"
    )

    task.start()
    print(f"  Task ID: {task.id}")
    print(f"  Status: {task.status()['state']}")

    return task


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


def extract_raster_to_grid(
    raster_path: Path,
    paddock_gdf: gpd.GeoDataFrame,
    paddock_name: str
) -> gpd.GeoDataFrame:
    """
    Extract raster values to 10m grid cells.

    Args:
        raster_path: Path to multi-band GeoTIFF with NDVI statistics
        paddock_gdf: Paddock boundary GeoDataFrame
        paddock_name: Name of paddock

    Returns:
        GeoDataFrame with cell geometries and statistic columns
    """
    if not RASTERIO_AVAILABLE:
        print("Error: rasterio not available")
        return gpd.GeoDataFrame()

    print(f"\n  Extracting raster to grid for {paddock_name}...")

    with rasterio.open(raster_path) as src:
        # Get band names from descriptions or use defaults
        band_names = list(src.descriptions) if src.descriptions[0] else [
            "ndvi_min", "ndvi_max", "ndvi_mean", "ndvi_median", "ndvi_std",
            "ndvi_count", "ndvi_range", "ndvi_cv", "ndvi_p10", "ndvi_p90",
            "ndvi_early", "ndvi_peak", "ndvi_late", "ndvi_decline",
            "ndvi_slope", "ndvi_intercept", "ndvi_date_max"
        ]

        print(f"    Raster bands: {src.count}")
        print(f"    Raster CRS: {src.crs}")

        # Ensure paddock is in raster CRS
        if paddock_gdf.crs != src.crs:
            paddock_gdf = paddock_gdf.to_crs(src.crs)

        paddock_geom = paddock_gdf.unary_union

        # Generate grid
        grid = generate_grid(paddock_geom, GRID_SIZE)
        print(f"    Generated {len(grid)} grid cells")

        # Get centroids for sampling
        centroids = grid.geometry.centroid

        # Extract values at centroids
        coords = [(p.x, p.y) for p in centroids]
        values = list(src.sample(coords))

        # Build DataFrame
        data = {'cell_id': grid['cell_id'].values}
        for i, band_name in enumerate(band_names[:src.count]):
            data[band_name] = [v[i] if not np.isnan(v[i]) and v[i] != src.nodata else np.nan
                               for v in values]

        # Create result GeoDataFrame
        result = grid.copy()
        for col, vals in data.items():
            if col != 'cell_id':
                result[col] = vals

        # Add metadata
        result['paddock'] = paddock_name

        valid_count = result['ndvi_mean'].notna().sum() if 'ndvi_mean' in result.columns else 0
        print(f"    Valid cells: {valid_count}/{len(result)}")

        return result


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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cluster for each k
    for k in n_clusters:
        col_name = f'cluster_{k}'
        gdf[col_name] = np.nan

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        gdf.loc[valid_mask, col_name] = labels.astype(int)
        print(f"  Added cluster_{k} ({k} zones)")

    return gdf


def create_orobanche_risk_zones(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Create orobanche risk classification based on NDVI patterns.

    Risk levels based on expected signatures:
    - High risk: Low ndvi_max, high ndvi_std, negative slope, high decline
    - Medium risk: Moderate values
    - Low risk: High ndvi_max, low ndvi_std, stable/positive slope
    """
    if 'ndvi_max' not in gdf.columns:
        return gdf

    print("\nCreating orobanche risk zones...")

    # Initialize risk score
    gdf['risk_score'] = 0.0
    valid_mask = gdf['ndvi_max'].notna()

    if valid_mask.sum() == 0:
        return gdf

    # Score components (normalized 0-1, higher = more risk)
    for col, weight, invert in [
        ('ndvi_max', 0.3, True),       # Lower max = higher risk
        ('ndvi_std', 0.2, False),      # Higher std = higher risk
        ('ndvi_slope', 0.25, True),    # Negative slope = higher risk
        ('ndvi_decline', 0.25, False), # Higher decline = higher risk
    ]:
        if col in gdf.columns:
            vals = gdf.loc[valid_mask, col].values
            # Normalize to 0-1
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            if vmax > vmin:
                normalized = (vals - vmin) / (vmax - vmin)
                if invert:
                    normalized = 1 - normalized
                gdf.loc[valid_mask, 'risk_score'] += normalized * weight

    # Classify into risk zones
    gdf['risk_zone'] = 'Unknown'
    risk_vals = gdf.loc[valid_mask, 'risk_score']

    # Quantile-based classification
    q33 = risk_vals.quantile(0.33)
    q66 = risk_vals.quantile(0.66)

    gdf.loc[valid_mask & (gdf['risk_score'] <= q33), 'risk_zone'] = 'Low'
    gdf.loc[valid_mask & (gdf['risk_score'] > q33) & (gdf['risk_score'] <= q66), 'risk_zone'] = 'Medium'
    gdf.loc[valid_mask & (gdf['risk_score'] > q66), 'risk_zone'] = 'High'

    # Summary
    for zone in ['Low', 'Medium', 'High']:
        count = (gdf['risk_zone'] == zone).sum()
        print(f"  {zone} risk: {count} cells")

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
            result = extract_raster_to_grid(raster_path, paddock_gdf, paddock_name)
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
                result = extract_raster_to_grid(raster_path, paddock_gdf, paddock_name)
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

    # Create risk zones
    if args.risk_zones:
        combined = create_orobanche_risk_zones(combined)

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
