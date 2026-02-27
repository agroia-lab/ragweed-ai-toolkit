#!/usr/bin/env python3
"""
Extract Presto embeddings with CUSTOM TEMPORAL WINDOWS (1-24 months).

This script extends extract_presto_embeddings.py to support flexible temporal
windows for Orobanche detection. Instead of requiring exactly 12 monthly
timesteps, it supports 1-24 month windows with predefined agricultural seasons.

IMPORTANT: Requires the modified WorldCereal code that accepts 1-24 timesteps.
The modification is in:
  worldcereal-classification_malezas/src/worldcereal/openeo/feature_extractor.py
  worldcereal-classification_malezas/src/worldcereal/openeo/inference.py

Predefined agricultural windows for Orobanche in tomato (Chile, ~34S):
  planting:        Sep-Nov (3 months)  Early crop establishment
  growth:          Dec-Feb (3 months)  Peak growth + orobanche emergence
  peak:            Oct-Jan (4 months)  Full orobanche symptom window
  full:            Apr-Mar (12 months) Complete annual cycle
  pre_season:      Jun-Aug (3 months)  Bare soil before planting
  post_harvest:    Mar-May (3 months)  After harvest
  extended_growth: Sep-Mar (7 months)  Planting through harvest

Requirements:
    conda activate worldcereal
    # Same as extract_presto_embeddings.py

Examples:
    # List available windows
    python extract_presto_custom_windows.py --list-windows

    # Extract planting season (Sep-Nov, 3 months)
    python extract_presto_custom_windows.py --season 25-26 --windows planting --paddock all

    # Extract multiple windows at once
    python extract_presto_custom_windows.py --season 25-26 --windows planting growth --paddock all

    # Custom date range (any 1-24 month span)
    python extract_presto_custom_windows.py --start 2025-09-01 --end 2025-11-30 --paddock Parcela_5

    # With TIME pooling (per-month embeddings)
    python extract_presto_custom_windows.py --season 24-25 --windows full --time-pooling --paddock all

    # With clustering and PCA
    python extract_presto_custom_windows.py --season 24-25 --windows growth --paddock all --cluster 20 --pca --local
"""

import argparse
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from shapely.geometry import box, mapping

RASTERIO_AVAILABLE = importlib.util.find_spec("rasterio") is not None
SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None

# ---- Portable path resolution ----
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

# --- Library imports (replacing sibling-script imports) ---
from ragweed_toolkit.satellite.presto import (
    PRESTO_DIMENSIONS,
    PRESTO_NODATA,
    PRESTO_OFFSET,
    PRESTO_SCALE,
)
from ragweed_toolkit.satellite.presto import (
    apply_clustering as _lib_apply_clustering,
)
from ragweed_toolkit.satellite.presto import (
    apply_pca as _lib_apply_pca,
)

# --- Import shared helpers from the library-backed sibling script ---
# These functions were already kept as script-specific in extract_presto_embeddings.py
try:
    from extract_presto_embeddings import (
        authenticate_openeo,
        check_dependencies,
        get_paddock_extent_utm,
        load_paddock_boundaries,
    )
    ORIGINAL_AVAILABLE = True
except ImportError:
    ORIGINAL_AVAILABLE = False

OPENEO_URL = "https://openeo.dataspace.copernicus.eu"

# Output directories
OUTPUT_DIR = Path("/media/malezainia1/LORENZO/outputs_sugal_25-26/satellite_embeddings/presto")
OUTPUT_DIR_MAIN = Path("/media/malezainia1/LORENZO/outputs_sugal_25-26/satellite_embeddings")

# ============================================================================
# AGRICULTURAL WINDOWS for Orobanche detection in tomato (Chile, ~34S)
# ============================================================================
# Based on literature review:
#   - Diaz et al. (2006): O. ramosa emergence at 550 GDD in Chile
#   - arXiv:2509.10804 (2025): Sentinel-2 broomrape detection in tomato
#   - Cochavi et al. (2017): Spectral changes from parasitism at 31-38 DAP
#
# Tomato calendar (O'Higgins region):
#   Transplanting: Sep-Oct | Growth: Nov-Jan | Harvest: Feb-Mar
# ============================================================================

AGRICULTURAL_WINDOWS = {
    "planting": {
        "months": (9, 11),  # Sep-Nov
        "n_timesteps": 3,
        "description": "Early crop establishment + root colonization baseline",
    },
    "growth": {
        "months": (12, 2),  # Dec-Feb (crosses year boundary)
        "n_timesteps": 3,
        "description": "Peak growth + orobanche emergence (strongest signal)",
    },
    "peak": {
        "months": (10, 1),  # Oct-Jan
        "n_timesteps": 4,
        "description": "Full orobanche symptom window (establishment to peak damage)",
    },
    "full": {
        "months": (4, 3),  # Apr-Mar (12 months)
        "n_timesteps": 12,
        "description": "Complete annual cycle (standard Presto window)",
    },
    "pre_season": {
        "months": (6, 8),  # Jun-Aug
        "n_timesteps": 3,
        "description": "Bare soil before planting (baseline conditions)",
    },
    "post_harvest": {
        "months": (3, 5),  # Mar-May
        "n_timesteps": 3,
        "description": "After harvest (residual stress patterns)",
    },
    "extended_growth": {
        "months": (9, 3),  # Sep-Mar
        "n_timesteps": 7,
        "description": "Planting through harvest (full crop season)",
    },
}


def count_months(start_date: str, end_date: str) -> int:
    """Count the number of monthly timesteps between two dates."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    months = (end.year - start.year) * 12 + (end.month - start.month) + 1
    return max(1, months)


def compute_window_dates(window_name: str, season: str) -> Tuple[str, str, str]:
    """
    Compute actual dates for a named agricultural window and season.

    Args:
        window_name: Name from AGRICULTURAL_WINDOWS
        season: Growing season like '25-26' or '24-25'

    Returns:
        Tuple of (start_date, end_date, label)
    """
    if window_name not in AGRICULTURAL_WINDOWS:
        raise ValueError(f"Unknown window '{window_name}'. Available: {list(AGRICULTURAL_WINDOWS.keys())}")

    window = AGRICULTURAL_WINDOWS[window_name]
    start_month, end_month = window["months"]

    # Parse season years
    parts = season.split("-")
    if len(parts) != 2:
        raise ValueError(f"Season must be in format 'YY-YY', got: {season}")

    year1 = 2000 + int(parts[0])
    year2 = 2000 + int(parts[1])

    # Determine start year based on which half of season
    if start_month >= 4:  # Apr-Dec: first year of season
        start_year = year1
    else:  # Jan-Mar: second year of season
        start_year = year2

    # Determine end year
    if end_month >= start_month and start_month >= 4:
        end_year = start_year
    elif end_month < start_month:  # crosses year boundary
        end_year = start_year + 1
    else:
        end_year = year2

    # Compute last day of end month
    end_date_obj = datetime(end_year, end_month, 1) + relativedelta(months=1) - relativedelta(days=1)

    start_date = f"{start_year}-{start_month:02d}-01"
    end_date = end_date_obj.strftime("%Y-%m-%d")
    label = f"{season}_{window_name}"

    return start_date, end_date, label


def validate_window(start_date: str, end_date: str, label: str) -> bool:
    """Validate a temporal window is within Presto's 1-24 timestep range."""
    n_months = count_months(start_date, end_date)

    if n_months < 1 or n_months > 24:
        print(f"  ERROR: Window '{label}' has {n_months} months (Presto supports 1-24)")
        return False

    status = "standard" if n_months == 12 else "custom"
    print(f"  {label}: {start_date} to {end_date} ({n_months} months, {status})")
    return True


def extract_presto_embeddings_custom(
    extent: dict,
    start_date: str,
    end_date: str,
    output_path: Path,
    paddock_name: str = "paddock",
    time_pooling: bool = False,
) -> Optional[Path]:
    """
    Extract Presto embeddings via openEO with custom temporal window.

    What this does:
        Sends a request to the Copernicus openEO platform to extract
        Sentinel-1/2 data for the specified area and time period, then
        runs the Presto model to produce 128-dimensional embeddings.

    Args:
        extent: Bounding box with west, south, east, north, epsg
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_path: Where to save the GeoTIFF
        paddock_name: Name for the openEO job title
        time_pooling: If True, request per-timestep embeddings (N*128 bands)

    Returns:
        Path to downloaded GeoTIFF or None if failed
    """
    try:
        from openeo_gfmap.spatial import BoundingBoxExtent
        from openeo_gfmap.temporal import TemporalContext
        from worldcereal.job import INFERENCE_JOB_OPTIONS, create_embeddings_process_graph
        from worldcereal.parameters import EmbeddingsParameters
    except ImportError as e:
        print(f"Error: WorldCereal package not available: {e}")
        print("Install with: pip install worldcereal openeo openeo-gfmap")
        return None

    n_months = count_months(start_date, end_date)

    print(f"\n{'='*60}")
    print(f"OPENEO JOB: {paddock_name}")
    print(f"{'='*60}")
    print(f"  Extent: {extent['west']:.0f}, {extent['south']:.0f} to {extent['east']:.0f}, {extent['north']:.0f}")
    print(f"  Period: {start_date} to {end_date} ({n_months} monthly timesteps)")
    if time_pooling:
        print(f"  TIME POOLING: ON - output will have {n_months} x {PRESTO_DIMENSIONS} = {n_months * PRESTO_DIMENSIONS} bands")
    else:
        print(f"  Output: {PRESTO_DIMENSIONS} bands (mean-pooled over time)")

    try:
        embedding_params = EmbeddingsParameters()

        bbox_extent = BoundingBoxExtent(
            west=extent['west'],
            south=extent['south'],
            east=extent['east'],
            north=extent['north'],
            epsg=extent['epsg']
        )

        temporal_extent = TemporalContext(
            start_date=start_date,
            end_date=end_date
        )

        print("  Building process graph...")
        print(f"  (Using {n_months}-month window - Presto will use {n_months} timesteps)")

        inference_result = create_embeddings_process_graph(
            spatial_extent=bbox_extent,
            temporal_extent=temporal_extent,
            embeddings_parameters=embedding_params,
            scale_uint16=True,
        )

        print("  Creating openEO job...")
        job = inference_result.create_job(
            title=f"Presto {paddock_name} {start_date} to {end_date} ({n_months}mo)",
            job_options=INFERENCE_JOB_OPTIONS,
        )

        print("  Starting job (this may take 10-60 minutes depending on area and window size)...")
        print("  Shorter windows generally process faster than 12-month windows.")
        job.start_and_wait()

        print("  Job finished! Downloading results...")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        for asset in job.get_results().get_assets():
            if asset.metadata.get("type", "").startswith("image/tiff"):
                asset.download(str(output_path))
                break
        else:
            raise RuntimeError("No GeoTIFF asset found in job results.")

        if not output_path.exists():
            raise FileNotFoundError(f"Download failed: {output_path}")

        print(f"  Downloaded: {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        return output_path

    except Exception as e:
        print(f"  Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return None


def raster_to_grid_embeddings_custom(
    raster_path: Path,
    paddock_gdf: gpd.GeoDataFrame,
    paddock_name: str,
    time_pooling: bool = False,
    n_timesteps: int = None,
) -> gpd.GeoDataFrame:
    """
    Convert Presto embedding raster to 10m grid cells.

    Handles both standard (128-band) and time-pooled (N*128-band) rasters.
    This function is kept script-specific because of the time-pooling logic
    which has no library equivalent.

    Args:
        raster_path: Path to GeoTIFF with embedding bands
        paddock_gdf: Paddock boundary GeoDataFrame
        paddock_name: Name of paddock
        time_pooling: If True, interpret bands as N timesteps x 128 dims
        n_timesteps: Number of timesteps (required if time_pooling=True)

    Returns:
        GeoDataFrame with cell geometries and embedding columns
    """
    if not RASTERIO_AVAILABLE:
        print("Error: rasterio not available for raster processing")
        return gpd.GeoDataFrame()

    print(f"\n  Converting raster to grid for {paddock_name}...")

    import rasterio
    from rasterio.mask import mask as rio_mask

    with rasterio.open(raster_path) as src:
        n_bands = src.count
        print(f"    Raster CRS: {src.crs}")
        print(f"    Raster shape: {n_bands} bands x {src.height} x {src.width}")

        # Determine embedding structure
        if time_pooling and n_timesteps and n_bands == n_timesteps * PRESTO_DIMENSIONS:
            dims_per_timestep = PRESTO_DIMENSIONS
            print(f"    TIME POOLING mode: {n_timesteps} timesteps x {dims_per_timestep} dims = {n_bands} bands")
        elif n_bands == PRESTO_DIMENSIONS:
            time_pooling = False
            dims_per_timestep = PRESTO_DIMENSIONS
            print(f"    Standard mode: {PRESTO_DIMENSIONS} embedding dimensions")
        elif n_bands % PRESTO_DIMENSIONS == 0:
            n_timesteps = n_bands // PRESTO_DIMENSIONS
            dims_per_timestep = PRESTO_DIMENSIONS
            time_pooling = n_timesteps > 1
            print(f"    Detected {n_timesteps} timesteps x {dims_per_timestep} dims = {n_bands} bands")
        else:
            print(f"    WARNING: Unexpected band count ({n_bands}). Using as-is.")
            time_pooling = False
            dims_per_timestep = n_bands

        # Ensure paddock is in raster CRS
        if paddock_gdf.crs != src.crs:
            paddock_gdf_reproj = paddock_gdf.to_crs(src.crs)
        else:
            paddock_gdf_reproj = paddock_gdf

        paddock_geom = paddock_gdf_reproj.unary_union

        # Read masked data
        try:
            out_image, out_transform = rio_mask(
                src, [mapping(paddock_geom)], crop=True, nodata=PRESTO_NODATA
            )
        except Exception as e:
            print(f"    Warning: Mask failed, reading full raster: {e}")
            out_image = src.read()
            out_transform = src.transform

        actual_bands, height, width = out_image.shape
        print(f"    Masked shape: {actual_bands} bands x {height} x {width}")

        # Convert UInt16 to float
        nodata_mask = out_image[0] == PRESTO_NODATA
        out_image = out_image.astype(np.float32) * PRESTO_SCALE + PRESTO_OFFSET
        out_image[:, nodata_mask] = np.nan

        # Create grid cells
        cells = []
        embeddings_list = []
        cell_id = 0

        for row in range(height):
            for col in range(width):
                x = out_transform.c + col * out_transform.a
                y = out_transform.f + row * out_transform.e
                cell_geom = box(x, y, x + abs(out_transform.a), y - abs(out_transform.e))
                pixel_emb = out_image[:, row, col]

                if np.isnan(pixel_emb[0]):
                    continue

                if cell_geom.intersects(paddock_geom):
                    clipped_geom = cell_geom.intersection(paddock_geom)
                    if not clipped_geom.is_empty:
                        cells.append({'cell_id': cell_id, 'geometry': clipped_geom})
                        embeddings_list.append(pixel_emb)
                        cell_id += 1

        print(f"    Generated {len(cells)} grid cells with embeddings")

    if not cells:
        print(f"    Warning: No valid cells for {paddock_name}")
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame(cells, crs=src.crs)
    if gdf.crs.to_epsg() != 32719:
        gdf = gdf.to_crs(epsg=32719)

    emb_array = np.array(embeddings_list)

    if time_pooling and n_timesteps and n_timesteps > 1:
        # Time-pooled: T01_A00, T01_A01, ..., T02_A00, etc.
        for t in range(n_timesteps):
            for d in range(dims_per_timestep):
                band_idx = t * dims_per_timestep + d
                if band_idx < emb_array.shape[1]:
                    gdf[f'T{t+1:02d}_A{d:02d}'] = emb_array[:, band_idx]
        print(f"    Added {n_timesteps * dims_per_timestep} time-pooled columns")

        # Also add mean-pooled columns for convenience
        for d in range(dims_per_timestep):
            time_cols = [f'T{t+1:02d}_A{d:02d}' for t in range(n_timesteps)]
            gdf[f'A{d:02d}'] = gdf[time_cols].mean(axis=1)
        print(f"    Added mean-pooled summary columns A00-A{dims_per_timestep-1:02d}")
    else:
        for i in range(min(actual_bands, emb_array.shape[1])):
            gdf[f'A{i:02d}'] = emb_array[:, i]

    gdf['paddock'] = paddock_name
    return gdf


# --- Thin CLI wrappers around library clustering/PCA ---
# These add logging and delegate to the library functions.

def apply_global_clustering(gdf, n_clusters):
    """Global clustering wrapper around library function."""
    print(f"  Applying global clustering (K={n_clusters})...")
    return _lib_apply_clustering(gdf, n_clusters=n_clusters, scope="global")


def apply_local_clustering(gdf, n_clusters):
    """Per-paddock clustering wrapper around library function."""
    print(f"  Applying local (per-paddock) clustering (K={n_clusters})...")
    result = gdf.copy()
    for paddock in result['paddock'].unique():
        mask = result['paddock'] == paddock
        subset = result[mask].copy()
        subset = _lib_apply_clustering(subset, n_clusters=n_clusters, scope="local")
        # Copy local cluster column back
        col_name = f"cluster_local_{n_clusters}"
        if col_name in subset.columns:
            result.loc[mask, col_name] = subset[col_name].values
    return result


def apply_global_pca(gdf):
    """Global PCA wrapper around library function."""
    print("  Applying global PCA...")
    return _lib_apply_pca(gdf, n_components=3, scope="global")


def apply_local_pca(gdf):
    """Per-paddock PCA wrapper around library function."""
    print("  Applying local (per-paddock) PCA...")
    result = gdf.copy()
    for paddock in result['paddock'].unique():
        mask = result['paddock'] == paddock
        subset = result[mask].copy()
        subset = _lib_apply_pca(subset, n_components=3, scope="local")
        # Copy local PCA columns back
        for col in subset.columns:
            if col.startswith("pca_local_"):
                result.loc[mask, col] = subset[col].values
    return result


def print_available_windows():
    """Print all available agricultural windows with descriptions."""
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    print("\nAvailable agricultural windows:")
    print("-" * 70)
    print(f"  {'Name':<18} {'Months':<14} {'Timesteps':<10} Description")
    print("-" * 70)
    for name, info in AGRICULTURAL_WINDOWS.items():
        start_m, end_m = info["months"]
        months_str = f"{month_names[start_m]}-{month_names[end_m]}"
        print(f"  {name:<18} {months_str:<14} {info['n_timesteps']:<10} {info['description']}")
    print("-" * 70)
    print()
    print("  You can also use --start and --end for custom date ranges (1-24 months).")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Extract Presto embeddings with CUSTOM TEMPORAL WINDOWS (1-24 months)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WHAT'S NEW (compared to extract_presto_embeddings.py):
  - Supports 1-24 month windows (not just 12 months)
  - Predefined agricultural windows (planting, growth, peak, etc.)
  - Multiple windows in a single run (--windows planting growth)
  - Time pooling to preserve monthly temporal dimension

EXAMPLES:

  # Authenticate (first time only)
  python extract_presto_custom_windows.py --authenticate

  # List available agricultural windows
  python extract_presto_custom_windows.py --list-windows

  # Extract planting season (Sep-Nov, 3 months)
  python extract_presto_custom_windows.py --season 25-26 --windows planting --paddock all

  # Extract multiple windows at once
  python extract_presto_custom_windows.py --season 25-26 --windows planting growth --paddock all

  # Custom date range (any 1-24 month span)
  python extract_presto_custom_windows.py --start 2025-09-01 --end 2025-11-30 --paddock Parcela_5

  # Full year with time pooling (12 x 128 = 1536 columns per pixel)
  python extract_presto_custom_windows.py --season 24-25 --windows full --time-pooling --paddock all

  # With clustering and PCA
  python extract_presto_custom_windows.py --season 24-25 --windows growth --paddock all --cluster 20 --pca --local
"""
    )

    # Authentication
    parser.add_argument("--authenticate", action="store_true",
                        help="Authenticate with Copernicus CDSE")

    # Window selection
    parser.add_argument("--season", type=str,
                        help="Growing season (e.g., '25-26', '24-25')")
    parser.add_argument("--windows", type=str, nargs="+",
                        choices=list(AGRICULTURAL_WINDOWS.keys()),
                        help="Agricultural window(s) to extract (can specify multiple)")
    parser.add_argument("--start", type=str, help="Custom start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="Custom end date (YYYY-MM-DD)")
    parser.add_argument("--list-windows", action="store_true",
                        help="List all available agricultural windows")

    # Spatial selection
    parser.add_argument("--paddock", type=str, default="all",
                        help="Paddock name or 'all' (default: all)")
    parser.add_argument("--boundaries", type=str,
                        help="Custom paddock boundaries file (.gpkg/.shp)")

    # Output options
    parser.add_argument("--output", type=str,
                        help="Custom output file path (overrides auto-naming)")
    parser.add_argument("--time-pooling", action="store_true",
                        help="Preserve per-month temporal dimension (N*128 columns instead of 128)")

    # Analysis options
    parser.add_argument("--cluster", type=int, default=None,
                        help="Number of KMeans clusters (e.g., 20)")
    parser.add_argument("--pca", action="store_true",
                        help="Apply PCA dimensionality reduction")
    parser.add_argument("--local", action="store_true",
                        help="Also compute per-paddock (local) clustering/PCA")

    # Processing options
    parser.add_argument("--raster", type=str,
                        help="Use existing raster file (skip openEO extraction)")
    parser.add_argument("--check-deps", action="store_true",
                        help="Check if all dependencies are installed")

    args = parser.parse_args()

    # --- Special modes ---
    if args.list_windows:
        print_available_windows()
        sys.exit(0)

    if args.check_deps:
        if not ORIGINAL_AVAILABLE:
            print("ERROR: Cannot import from extract_presto_embeddings.py")
            sys.exit(1)
        ok, missing = check_dependencies()
        sys.exit(0 if ok else 1)

    if args.authenticate:
        if not ORIGINAL_AVAILABLE:
            print("ERROR: Cannot import from extract_presto_embeddings.py")
            sys.exit(1)
        ok, _ = check_dependencies()
        if not ok:
            sys.exit(1)
        success = authenticate_openeo()
        sys.exit(0 if success else 1)

    # --- Determine windows to extract ---
    windows_to_extract = []  # List of (start_date, end_date, label)

    if args.start and args.end:
        label = f"{args.season}_custom" if args.season else "custom"
        windows_to_extract.append((args.start, args.end, label))

    elif args.windows:
        if not args.season:
            print("ERROR: --season is required when using --windows")
            print("       Example: --season 25-26 --windows planting growth")
            sys.exit(1)

        for window_name in args.windows:
            try:
                start, end, label = compute_window_dates(window_name, args.season)
                windows_to_extract.append((start, end, label))
            except ValueError as e:
                print(f"ERROR: {e}")
                sys.exit(1)

    elif args.raster:
        windows_to_extract.append(("unknown", "unknown", "raster"))

    else:
        parser.print_help()
        print("\nERROR: Specify either --windows (with --season) or --start/--end")
        print_available_windows()
        sys.exit(1)

    # --- Header ---
    print("=" * 60)
    print("PRESTO CUSTOM WINDOW EMBEDDINGS")
    print("=" * 60)
    print()

    if len(windows_to_extract) > 1:
        print(f"Extracting {len(windows_to_extract)} temporal windows:")
    else:
        print("Extracting temporal window:")

    all_valid = True
    for start, end, label in windows_to_extract:
        if start == "unknown":
            print(f"  {label}: Using existing raster")
            continue
        if not validate_window(start, end, label):
            all_valid = False

    if not all_valid:
        print("\nAborting due to invalid window(s).")
        sys.exit(1)

    print()
    print(f"Paddock: {args.paddock}")
    if args.time_pooling:
        print("Time pooling: ON (per-month embeddings)")
    if args.cluster:
        print(f"Clustering: K={args.cluster}")
    if args.pca:
        print("PCA: enabled")
    if args.local:
        print("Local analysis: enabled")
    print()

    # --- Check dependencies ---
    if not args.raster:
        if not ORIGINAL_AVAILABLE:
            print("ERROR: Cannot import shared functions from extract_presto_embeddings.py")
            sys.exit(1)
        ok, _ = check_dependencies()
        if not ok:
            sys.exit(1)

    # --- Load boundaries ---
    boundaries = load_paddock_boundaries(args.paddock, args.boundaries)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Process each window ---
    for window_idx, (start_date, end_date, window_label) in enumerate(windows_to_extract):

        if len(windows_to_extract) > 1:
            print(f"\n{'#'*60}")
            print(f"# WINDOW {window_idx + 1}/{len(windows_to_extract)}: {window_label}")
            print(f"# Period: {start_date} to {end_date}")
            print(f"{'#'*60}")

        n_months = count_months(start_date, end_date) if start_date != "unknown" else None

        all_results = []

        for paddock_name in boundaries['paddock'].unique():
            print(f"\n{'='*60}")
            print(f"Processing: {paddock_name} [{window_label}]")
            print(f"{'='*60}")

            paddock_gdf = boundaries[boundaries['paddock'] == paddock_name]

            if args.raster:
                raster_path = Path(args.raster)
                if not raster_path.exists():
                    print(f"Error: Raster not found: {raster_path}")
                    continue
            else:
                raster_filename = f"presto_raw_{paddock_name}_{window_label}.tif"
                raster_path = OUTPUT_DIR / raster_filename

                if not raster_path.exists():
                    extent = get_paddock_extent_utm(paddock_gdf)
                    raster_path = extract_presto_embeddings_custom(
                        extent=extent,
                        start_date=start_date,
                        end_date=end_date,
                        output_path=raster_path,
                        paddock_name=paddock_name,
                        time_pooling=args.time_pooling,
                    )
                    if raster_path is None:
                        print(f"  Skipping {paddock_name} - extraction failed")
                        continue
                else:
                    print(f"  Using existing raster: {raster_path}")

            result = raster_to_grid_embeddings_custom(
                raster_path=raster_path,
                paddock_gdf=paddock_gdf,
                paddock_name=paddock_name,
                time_pooling=args.time_pooling,
                n_timesteps=n_months,
            )

            if len(result) > 0:
                all_results.append(result)

        if not all_results:
            print(f"\nWARNING: No valid embeddings for window '{window_label}'")
            if len(windows_to_extract) > 1:
                print("Continuing with next window...")
                continue
            else:
                sys.exit(1)

        # Combine results
        combined = pd.concat(all_results, ignore_index=True)
        combined = gpd.GeoDataFrame(combined, crs="EPSG:32719")

        # Add metadata
        combined['season'] = args.season if args.season else "custom"
        combined['window'] = window_label
        combined['temporal_window'] = f"{start_date} to {end_date}"
        combined['n_timesteps'] = n_months if n_months else 0
        combined['embedding_source'] = 'Presto'
        combined['time_pooled'] = args.time_pooling
        combined['extraction_date'] = datetime.now().isoformat()

        # Clustering and PCA
        if args.cluster:
            combined = apply_global_clustering(combined, args.cluster)
        if args.pca:
            combined = apply_global_pca(combined)
        if args.local:
            if args.cluster:
                combined = apply_local_clustering(combined, args.cluster)
            if args.pca:
                combined = apply_local_pca(combined)

        # Output path
        if args.output:
            output_path = Path(args.output)
            if len(windows_to_extract) > 1:
                stem = output_path.stem
                output_path = output_path.parent / f"{stem}_{window_label}{output_path.suffix}"
        else:
            season_str = args.season if args.season else "custom"
            suffix = "complete" if args.local else "global"
            if args.time_pooling:
                suffix += "_timepooled"
            window_part = window_label.split('_', 1)[-1] if '_' in window_label else window_label
            output_path = OUTPUT_DIR_MAIN / f"embeddings_{season_str}_presto_{window_part}_{suffix}.gpkg"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_file(output_path, driver="GPKG")

        # Summary
        print()
        print("=" * 60)
        print(f"WINDOW COMPLETE: {window_label}")
        print("=" * 60)
        print(f"  Output: {output_path}")
        print(f"  Total cells: {len(combined)}")
        print(f"  Paddocks: {combined['paddock'].nunique()}")
        print(f"  Period: {start_date} to {end_date}")
        if n_months:
            print(f"  Timesteps: {n_months} months")

        print()
        print("  Summary by paddock:")
        emb_col = 'A00' if 'A00' in combined.columns else 'cell_id'
        if emb_col in combined.columns:
            summary = combined.groupby('paddock').agg({
                'cell_id': 'count',
                emb_col: lambda x: x.notna().sum() if emb_col != 'cell_id' else len(x)
            }).rename(columns={'cell_id': 'total_cells', emb_col: 'valid_embeddings'})
            for line in summary.to_string().split('\n'):
                print(f"    {line}")

    # --- Final summary ---
    print()
    print("=" * 60)
    print("ALL EXTRACTIONS COMPLETE")
    print("=" * 60)
    print(f"  Windows processed: {len(windows_to_extract)}")
    for start, end, label in windows_to_extract:
        print(f"    - {label}: {start} to {end}")
    print()
    print("To visualize in QGIS:")
    print("  1. Layer > Add Layer > Add Vector Layer")
    print(f"  2. Navigate to: {OUTPUT_DIR_MAIN}")
    print("  3. Select the .gpkg file(s)")
    if args.cluster:
        print(f"  4. Style by: cluster_global_{args.cluster} (categorized)")
    if args.pca:
        print("  5. PCA visualization: pca_global_1 (graduated)")
    if args.time_pooling:
        print("  6. Time-pooled columns: T01_A00, T01_A01, ..., T02_A00, etc.")
        print("     Mean-pooled summary: A00, A01, ..., A127")


if __name__ == "__main__":
    main()
