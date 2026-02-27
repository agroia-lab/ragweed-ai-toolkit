#!/usr/bin/env python3
"""
Extract WorldCereal/Presto embeddings for paddock zones.

This script uses the WorldCereal Presto model to extract 128-dimensional embeddings
from Sentinel-1/2 time series data via openEO.

IMPORTANT: Presto requires exactly 12 monthly timesteps (a full year of data).
Shorter temporal windows will fail with: "Can only run Presto on 12 timesteps, got: N"

Key differences from Google embeddings:
- 128 dimensions (vs 64)
- REQUIRES 12-month windows (cannot use partial seasons directly)
- Based on Sentinel-1 + Sentinel-2 fusion (not annual composites)
- Uses NASA Harvest Presto model finetuned on WorldCereal data

For crop-season-specific analysis, extract embeddings for a 12-month window
ending at harvest time, then analyze the temporal patterns within that year.

Requirements:
    conda create -n worldcereal python=3.11 -y
    conda activate worldcereal
    cd /home/malezainia1/dev/worldcereal-classification_malezas
    pip install -e . --no-deps
    pip install openeo==0.35.0 openeo-gfmap==0.4.7 geopandas rasterio scikit-learn tqdm

Authentication:
    python extract_presto_embeddings.py --authenticate
    # Opens browser for Copernicus CDSE login (lleon@inia.cl)

Usage (IMPORTANT: Presto requires 12-month windows):
    # Extract 24-25 season (historical data available)
    python extract_presto_embeddings.py --season 24-25 --paddock all --cluster 20 --pca --local

    # Extract single paddock for testing
    python extract_presto_embeddings.py --season 24-25 --paddock Parcela_5

    # Custom 12-month window
    python extract_presto_embeddings.py --start 2024-04-01 --end 2025-03-31 --paddock all
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import tempfile
import time

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box, mapping
from tqdm import tqdm

# Optional imports for raster processing
try:
    import rasterio
    from rasterio.mask import mask as rio_mask
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

# Optional imports for clustering/PCA
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA, IncrementalPCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# --- Library imports (replacing internal duplicates) ---
from ragweed_toolkit.satellite.presto import (
    extract_presto_embeddings as _lib_extract_presto,
    raster_to_grid as _lib_raster_to_grid,
    apply_clustering as _lib_apply_clustering,
    apply_pca as _lib_apply_pca,
    PRESTO_DIMENSIONS,
    PRESTO_SCALE,
    PRESTO_OFFSET,
    PRESTO_NODATA,
    SEASON_WINDOWS_12M,
)

# Output directories
OUTPUT_DIR = Path("/media/malezainia1/LORENZO/outputs_sugal_25-26/satellite_embeddings/presto")
OUTPUT_DIR_MAIN = Path("/media/malezainia1/LORENZO/outputs_sugal_25-26/satellite_embeddings")
BOUNDARIES_PATH = Path("/media/malezainia1/LORENZO/outputs_sugal_25-26/satellite_embeddings/paddock_boundaries.gpkg")

# Grid size
GRID_SIZE = 10  # meters (10x10m grid)

# Use library season windows
SEASON_WINDOWS = SEASON_WINDOWS_12M

# NOTE: Partial season windows DON'T WORK with Presto
# Presto requires exactly 12 timesteps (monthly composites for a full year)
# The error "Can only run Presto on 12 timesteps, got: N" means the window is too short
PARTIAL_SEASON_WINDOWS = {
    # These are kept for reference but WILL NOT WORK with Presto
    "25-26": ("2025-10-01", "2026-01-31"),  # 4 months - WON'T WORK
    "24-25": ("2024-10-01", "2025-01-31"),  # 4 months - WON'T WORK
    "22-23": ("2022-10-01", "2023-01-31"),  # 4 months - WON'T WORK
    "21-22": ("2021-10-01", "2022-01-31"),  # 4 months - WON'T WORK
    "20-21": ("2020-10-01", "2021-01-31"),  # 4 months - WON'T WORK
}

# OpenEO connection
OPENEO_URL = "https://openeo.dataspace.copernicus.eu"


def check_worldcereal_available() -> bool:
    """Check if WorldCereal package is available."""
    try:
        from worldcereal.job import create_embeddings_process_graph
        from worldcereal.parameters import EmbeddingsParameters
        return True
    except ImportError:
        return False


def check_dependencies() -> Tuple[bool, list]:
    """Check if required packages are installed."""
    missing = []

    try:
        import openeo
    except ImportError:
        missing.append("openeo")

    try:
        import worldcereal
    except ImportError:
        missing.append("worldcereal")

    try:
        import rasterio
    except ImportError:
        missing.append("rasterio")

    if missing:
        print("=" * 60)
        print("MISSING DEPENDENCIES")
        print("=" * 60)
        print()
        print("Required packages not installed:")
        for pkg in missing:
            print(f"  - {pkg}")
        print()
        print("To install, create a dedicated environment:")
        print()
        print("  conda create -n worldcereal python=3.10 -y")
        print("  conda activate worldcereal")
        print("  pip install worldcereal openeo geopandas rasterio scikit-learn tqdm")
        print()
        return False, missing

    return True, []


def authenticate_openeo() -> bool:
    """Authenticate with Copernicus Data Space Ecosystem."""
    try:
        import openeo

        print("=" * 60)
        print("COPERNICUS CDSE AUTHENTICATION")
        print("=" * 60)
        print()
        print(f"Connecting to: {OPENEO_URL}")
        print("A browser window will open for authentication.")
        print("Log in with your Copernicus CDSE account (lleon@inia.cl)")
        print()

        connection = openeo.connect(OPENEO_URL)
        connection.authenticate_oidc()

        print()
        print("Authentication successful!")
        print("Credentials cached for future use.")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"Authentication failed: {e}")
        return False


def get_openeo_connection():
    """Get authenticated OpenEO connection."""
    import openeo

    connection = openeo.connect(OPENEO_URL)

    try:
        connection.authenticate_oidc()
        return connection
    except Exception as e:
        print(f"Error: Please authenticate first:")
        print(f"  python extract_presto_embeddings.py --authenticate")
        return None


def load_paddock_boundaries(
    paddock_name: str = "all",
    boundaries_path: str = None
) -> gpd.GeoDataFrame:
    """Load paddock boundaries from GeoPackage."""
    bpath = Path(boundaries_path) if boundaries_path else BOUNDARIES_PATH

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
        gdf = gdf[gdf['paddock'] == paddock_name]
        if len(gdf) == 0:
            available = gpd.read_file(bpath)['paddock'].unique()
            print(f"Error: Paddock '{paddock_name}' not found.")
            print(f"Available: {', '.join(available)}")
            sys.exit(1)

    # Ensure UTM 19S
    if gdf.crs.to_epsg() != 32719:
        gdf = gdf.to_crs(epsg=32719)

    print(f"Loaded {len(gdf)} polygon(s) for: {', '.join(gdf['paddock'].unique())}")
    return gdf


def get_paddock_extent_utm(gdf: gpd.GeoDataFrame, buffer: float = 50) -> dict:
    """Get bounding box extent in UTM coordinates."""
    if gdf.crs.to_epsg() != 32719:
        gdf = gdf.to_crs(epsg=32719)

    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]

    return {
        'west': bounds[0] - buffer,
        'south': bounds[1] - buffer,
        'east': bounds[2] + buffer,
        'north': bounds[3] + buffer,
        'epsg': 32719
    }


# --- Thin CLI wrapper around library extraction function with progress logging ---
def extract_presto_embeddings_worldcereal(
    extent: dict,
    start_date: str,
    end_date: str,
    output_path: Path,
    paddock_name: str = "paddock"
) -> Optional[Path]:
    """
    Extract Presto embeddings using WorldCereal API via openEO.
    Wrapper around ragweed_toolkit.satellite.presto.extract_presto_embeddings
    with CLI-specific logging.
    """
    print(f"\n{'='*60}")
    print(f"OPENEO JOB: {paddock_name}")
    print(f"{'='*60}")
    print(f"  Extent: {extent['west']:.0f}, {extent['south']:.0f} to {extent['east']:.0f}, {extent['north']:.0f}")
    print(f"  Period: {start_date} to {end_date}")
    print("  Starting job (this may take 20-60 minutes)...")

    result = _lib_extract_presto(
        extent=extent,
        start_date=start_date,
        end_date=end_date,
        output_path=str(output_path),
        paddock_name=paddock_name,
        openeo_url=OPENEO_URL,
    )

    if result is not None:
        print(f"  Downloaded: {result}")
        print(f"  Size: {result.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print(f"  Error: Extraction failed for {paddock_name}")

    return result


# --- Thin CLI wrapper around library raster_to_grid with progress logging ---
def raster_to_grid_embeddings(
    raster_path: Path,
    paddock_gdf: gpd.GeoDataFrame,
    paddock_name: str
) -> gpd.GeoDataFrame:
    """
    Convert Presto embedding raster to 10m grid cells.
    Wrapper around ragweed_toolkit.satellite.presto.raster_to_grid
    with CLI-specific logging.
    """
    print(f"\n  Converting raster to grid for {paddock_name}...")

    result = _lib_raster_to_grid(
        raster_path=str(raster_path),
        paddock_gdf=paddock_gdf,
        paddock_name=paddock_name,
    )

    print(f"    Generated {len(result)} grid cells with embeddings")
    return result


# --- Wrappers around library clustering/PCA with CLI-specific logging ---

def apply_global_clustering(gdf, n_clusters=20):
    """Apply global KMeans clustering across all paddocks."""
    print(f"\nApplying GLOBAL KMeans (K={n_clusters})...")
    result = _lib_apply_clustering(gdf, n_clusters=n_clusters, scope="global")
    col_name = f'cluster_global_{n_clusters}'
    if col_name in result.columns:
        print(f"  Added column '{col_name}'")
    return result


def apply_local_clustering(gdf, n_clusters=20):
    """Apply local KMeans clustering within each paddock."""
    print(f"\nApplying LOCAL KMeans (K={n_clusters}) per paddock...")
    result = _lib_apply_clustering(gdf, n_clusters=n_clusters, scope="local")
    col_name = f'cluster_local_{n_clusters}'
    if col_name in result.columns:
        for paddock in result['paddock'].unique():
            mask = (result['paddock'] == paddock) & result[col_name].notna()
            print(f"  {paddock}: {mask.sum()} cells clustered")
    return result


def apply_global_pca(gdf, n_components=3):
    """Apply global PCA across all paddocks."""
    print(f"\nApplying GLOBAL PCA (n_components={n_components})...")
    result = _lib_apply_pca(gdf, n_components=n_components, scope="global")
    return result


def apply_local_pca(gdf, n_components=3):
    """Apply local PCA within each paddock."""
    print(f"\nApplying LOCAL PCA (n_components={n_components}) per paddock...")
    result = _lib_apply_pca(gdf, n_components=n_components, scope="local")
    for paddock in result['paddock'].unique():
        col = f'pca_local_1'
        if col in result.columns:
            mask = (result['paddock'] == paddock) & result[col].notna()
            print(f"  {paddock}: PCA computed ({mask.sum()} cells)")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract WorldCereal/Presto embeddings for paddock zones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT: Presto requires exactly 12 monthly timesteps (full year).
Partial season windows will NOT work.

Examples:
  # Authenticate with Copernicus CDSE (first time)
  python extract_presto_embeddings.py --authenticate

  # Extract 24-25 season (12-month window ending Mar 2025)
  python extract_presto_embeddings.py --season 24-25 --paddock all --cluster 20 --pca --local

  # Extract single paddock (faster for testing)
  python extract_presto_embeddings.py --season 24-25 --paddock Parcela_5

  # Custom 12-month window (MUST be exactly 12 months)
  python extract_presto_embeddings.py --start 2024-04-01 --end 2025-03-31 --paddock all

  # Process existing raster (skip openEO download)
  python extract_presto_embeddings.py --raster presto_raw.tif --paddock all --cluster 20

Available seasons (12-month windows ending at harvest):
  20-21: 2020-04-01 to 2021-03-31
  21-22: 2021-04-01 to 2022-03-31
  22-23: 2022-04-01 to 2023-03-31
  24-25: 2024-04-01 to 2025-03-31  <-- use this for historical data
  25-26: 2025-04-01 to 2026-03-31  <-- requires future satellite data!

NOTE: For 25-26 season, data is not yet available for Jan-Mar 2026.
      Use 24-25 season for testing.
"""
    )
    parser.add_argument("--authenticate", action="store_true",
                        help="Authenticate with Copernicus CDSE")
    parser.add_argument("--season", type=str, choices=list(SEASON_WINDOWS.keys()),
                        help="Tomato season (Sep-Mar window)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--paddock", type=str, default="all",
                        help="Paddock name or 'all'")
    parser.add_argument("--boundaries", type=str, help="Custom boundaries file")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--cluster", type=int, default=None,
                        help="Number of clusters (e.g., 20)")
    parser.add_argument("--pca", action="store_true", help="Apply PCA analysis")
    parser.add_argument("--local", action="store_true",
                        help="Also compute local (per-paddock) clustering/PCA")
    parser.add_argument("--raster", type=str,
                        help="Existing raster file (skip openEO)")
    parser.add_argument("--check-deps", action="store_true",
                        help="Check dependencies and exit")

    args = parser.parse_args()

    # Check dependencies
    if args.check_deps:
        ok, missing = check_dependencies()
        sys.exit(0 if ok else 1)

    # Authentication mode
    if args.authenticate:
        ok, _ = check_dependencies()
        if not ok:
            sys.exit(1)
        success = authenticate_openeo()
        sys.exit(0 if success else 1)

    # Determine temporal window
    if args.start and args.end:
        start_date = args.start
        end_date = args.end
        # Generate season label from dates
        start_year = start_date[:4]
        end_year = end_date[:4]
        if start_year != end_year:
            season_label = f"{start_year[-2:]}-{end_year[-2:]}"
        else:
            season_label = start_year
    elif args.season:
        start_date, end_date = SEASON_WINDOWS[args.season]
        season_label = args.season
    elif args.raster:
        # If only raster provided, use placeholder dates
        start_date, end_date = "unknown", "unknown"
        season_label = "raster"
    else:
        parser.print_help()
        print("\nError: Please specify --season or both --start and --end")
        sys.exit(1)

    print("=" * 60)
    print("PRESTO EMBEDDINGS EXTRACTION")
    print("=" * 60)
    print(f"Season: {season_label}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Paddock: {args.paddock}")
    print(f"Dimensions: {PRESTO_DIMENSIONS}")
    if args.cluster:
        print(f"Clustering: K={args.cluster}")
    if args.pca:
        print(f"PCA: enabled")
    if args.local:
        print(f"Local analysis: enabled")
    if args.raster:
        print(f"Raster: {args.raster}")
    print()

    # Check dependencies (unless using existing raster)
    if not args.raster:
        ok, _ = check_dependencies()
        if not ok:
            sys.exit(1)

    # Load boundaries
    boundaries = load_paddock_boundaries(args.paddock, args.boundaries)

    # Output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Process each paddock
    for paddock_name in boundaries['paddock'].unique():
        print(f"\n{'='*60}")
        print(f"Processing: {paddock_name}")
        print(f"{'='*60}")

        paddock_gdf = boundaries[boundaries['paddock'] == paddock_name]

        # Determine raster path
        if args.raster:
            raster_path = Path(args.raster)
            if not raster_path.exists():
                print(f"Error: Raster not found: {raster_path}")
                continue
        else:
            # Extract via openEO
            raster_path = OUTPUT_DIR / f"presto_raw_{paddock_name}_{season_label}.tif"

            if not raster_path.exists():
                extent = get_paddock_extent_utm(paddock_gdf)
                raster_path = extract_presto_embeddings_worldcereal(
                    extent=extent,
                    start_date=start_date,
                    end_date=end_date,
                    output_path=raster_path,
                    paddock_name=paddock_name
                )

                if raster_path is None:
                    print(f"  Skipping {paddock_name} - extraction failed")
                    continue
            else:
                print(f"  Using existing raster: {raster_path}")

        # Convert raster to grid embeddings
        result = raster_to_grid_embeddings(
            raster_path=raster_path,
            paddock_gdf=paddock_gdf,
            paddock_name=paddock_name
        )

        if len(result) > 0:
            all_results.append(result)

    if not all_results:
        print("\nError: No valid embeddings extracted")
        sys.exit(1)

    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, crs="EPSG:32719")

    # Add metadata
    combined['season'] = season_label
    combined['temporal_window'] = f"{start_date} to {end_date}"
    combined['embedding_source'] = 'Presto'
    combined['extraction_date'] = datetime.now().isoformat()

    # Apply global clustering
    if args.cluster:
        combined = apply_global_clustering(combined, args.cluster)

    # Apply global PCA
    if args.pca:
        combined = apply_global_pca(combined)

    # Apply local analysis
    if args.local:
        if args.cluster:
            combined = apply_local_clustering(combined, args.cluster)
        if args.pca:
            combined = apply_local_pca(combined)

    # Save output
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = "complete" if args.local else "global"
        output_path = OUTPUT_DIR_MAIN / f"embeddings_{season_label}_presto_{suffix}.gpkg"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_file(output_path, driver="GPKG")

    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Total cells: {len(combined)}")
    print(f"Paddocks: {combined['paddock'].nunique()}")
    print(f"Period: {start_date} to {end_date}")

    # Summary by paddock
    print()
    print("Summary by paddock:")
    emb_col = 'A00' if 'A00' in combined.columns else 'cell_id'
    if emb_col in combined.columns:
        summary = combined.groupby('paddock').agg({
            'cell_id': 'count',
            emb_col: lambda x: x.notna().sum() if emb_col != 'cell_id' else len(x)
        }).rename(columns={'cell_id': 'total_cells', emb_col: 'valid_embeddings'})
        print(summary.to_string())

    print()
    print("To visualize in QGIS:")
    print(f"  1. Add layer: {output_path}")
    if args.cluster:
        print(f"  2. Style by: cluster_global_{args.cluster} (categorized)")
        if args.local:
            print(f"     Or: cluster_local_{args.cluster} (per-paddock)")
    if args.pca:
        print(f"  3. PCA: pca_global_1 (graduated)")

    return combined


if __name__ == "__main__":
    main()
