#!/usr/bin/env python3
"""
Extract Google Satellite Embeddings for paddock zones.

This script:
1. Loads paddock boundaries from GeoPackage
2. Generates a 10x10m grid for each paddock
3. Extracts 64-dimensional embeddings from Google Satellite Embeddings
4. Optionally applies clustering (KMeans) and PCA analysis
5. Saves results to GeoPackage for QGIS visualization

Usage:
    python scripts/satellite/extract_embeddings.py --paddock all
    python scripts/satellite/extract_embeddings.py --paddock all --year 2025
    python scripts/satellite/extract_embeddings.py --paddock Apalta --year 2024 --cluster 10
    python scripts/satellite/extract_embeddings.py --boundaries custom.gpkg --years 2020 2021
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import ee
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
from tqdm import tqdm

# Optional imports for clustering/PCA
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Project paths - use portable path resolution
_script_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _script_dir.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.paths import get_paths, get_external_drive

# Get paths from config
_paths = get_paths()
_external_base = Path(_paths.get('external', {}).get('outputs_base', '/media'))
_satellite_embeddings = Path(_paths.get('external', {}).get('satellite_embeddings', _external_base / 'satellite_embeddings'))

# Default boundary files - prefer external, fallback to local
DEFAULT_BOUNDARIES = _satellite_embeddings / "paddock_boundaries.gpkg"
FALLBACK_BOUNDARIES = PROJECT_ROOT / "outputs/satellite_analysis/paddock_boundaries.gpkg"

# Output directory - external drive by default for large outputs
DEFAULT_OUTPUT_DIR = _satellite_embeddings
FALLBACK_OUTPUT_DIR = PROJECT_ROOT / "outputs/satellite_analysis"

# Earth Engine project
EE_PROJECT = "gen-lang-client-0195046178"

# Grid settings
GRID_SIZE = 10  # meters (10x10m grid)


def initialize_ee():
    """Initialize Earth Engine with project."""
    try:
        ee.Initialize(project=EE_PROJECT)
        print(f"✅ Earth Engine initialized (project: {EE_PROJECT})")
        return True
    except Exception as e:
        print(f"❌ Earth Engine initialization failed: {e}")
        print("Run: earthengine authenticate")
        return False


def get_boundaries_path(custom_path: str = None) -> Path:
    """Determine which boundaries file to use."""
    if custom_path:
        return Path(custom_path)
    if DEFAULT_BOUNDARIES.exists():
        return DEFAULT_BOUNDARIES
    return FALLBACK_BOUNDARIES


def get_output_dir() -> Path:
    """Determine output directory (prefer LORENZO2 if available)."""
    if DEFAULT_OUTPUT_DIR.exists():
        return DEFAULT_OUTPUT_DIR
    FALLBACK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return FALLBACK_OUTPUT_DIR


def load_paddock_boundaries(
    paddock_name: str = "all",
    boundaries_path: str = None
) -> gpd.GeoDataFrame:
    """Load paddock boundaries from GeoPackage.

    Args:
        paddock_name: Paddock name to filter, or 'all' for all paddocks
        boundaries_path: Custom boundaries file path (optional)
    """
    bpath = get_boundaries_path(boundaries_path)
    print(f"📂 Loading boundaries from: {bpath}")

    gdf = gpd.read_file(bpath)

    # Detect paddock field name (different sources use different names)
    paddock_field = None
    for field in ['paddock', 'Nombre Predio', 'Subparcela', 'name']:
        if field in gdf.columns:
            paddock_field = field
            break

    if paddock_field is None:
        print(f"⚠️ No recognized paddock field found. Columns: {list(gdf.columns)}")
        paddock_field = gdf.columns[0]  # Use first column

    # Standardize to 'paddock' column
    if paddock_field != 'paddock':
        gdf['paddock'] = gdf[paddock_field]

    if paddock_name.lower() != "all":
        gdf = gdf[gdf['paddock'] == paddock_name]
        if len(gdf) == 0:
            available = gpd.read_file(bpath)['paddock'].unique() if 'paddock' in gpd.read_file(bpath).columns else []
            print(f"❌ Paddock '{paddock_name}' not found.")
            if available:
                print(f"Available: {', '.join(available)}")
            sys.exit(1)

    print(f"📍 Loaded {len(gdf)} polygon(s) for: {', '.join(gdf['paddock'].unique())}")
    return gdf


def generate_grid(geometry, cell_size: int = 10) -> gpd.GeoDataFrame:
    """Generate a grid of cells over a geometry."""
    minx, miny, maxx, maxy = geometry.bounds

    # Create grid cells
    cells = []
    cell_ids = []

    x = minx
    cell_id = 0
    while x < maxx:
        y = miny
        while y < maxy:
            cell = box(x, y, x + cell_size, y + cell_size)
            # Only keep cells that intersect the paddock
            if cell.intersects(geometry):
                cells.append(cell.intersection(geometry))
                cell_ids.append(cell_id)
                cell_id += 1
            y += cell_size
        x += cell_size

    return gpd.GeoDataFrame({
        'cell_id': cell_ids,
        'geometry': cells
    }, crs="EPSG:32719")


def extract_embeddings_for_paddock(
    paddock_gdf: gpd.GeoDataFrame,
    paddock_name: str,
    year: int = 2024,
    batch_size: int = 500
) -> gpd.GeoDataFrame:
    """Extract satellite embeddings for a single paddock."""

    print(f"\n{'='*60}")
    print(f"Processing: {paddock_name}")
    print(f"{'='*60}")

    # Merge all polygons for this paddock
    paddock_union = paddock_gdf.unary_union

    # Generate grid
    print(f"Generating {GRID_SIZE}x{GRID_SIZE}m grid...")
    grid = generate_grid(paddock_union, GRID_SIZE)
    print(f"  → {len(grid)} grid cells")

    # Get centroids for sampling (convert to WGS84 for Earth Engine)
    grid_wgs84 = grid.to_crs(epsg=4326)
    centroids = grid_wgs84.geometry.centroid

    # Load embeddings collection
    embeddings = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
    emb_year = embeddings.filter(
        ee.Filter.date(f'{year}-01-01', f'{year}-12-31')
    ).mosaic()

    # Extract embeddings in batches
    print(f"Extracting embeddings for year {year}...")
    all_features = []

    # Process in batches to avoid EE limits
    for i in tqdm(range(0, len(centroids), batch_size), desc="Batches"):
        batch_centroids = centroids.iloc[i:i+batch_size]
        batch_ids = grid['cell_id'].iloc[i:i+batch_size].tolist()

        # Create EE feature collection for this batch
        ee_features = []
        for idx, (cell_id, centroid) in enumerate(zip(batch_ids, batch_centroids)):
            point = ee.Geometry.Point([centroid.x, centroid.y])
            ee_features.append(ee.Feature(point, {'cell_id': cell_id}))

        fc = ee.FeatureCollection(ee_features)

        # Sample embeddings at these points
        sampled = emb_year.sampleRegions(
            collection=fc,
            scale=10,
            geometries=False
        )

        # Get results
        try:
            results = sampled.getInfo()
            for feat in results['features']:
                props = feat['properties']
                all_features.append(props)
        except Exception as e:
            print(f"  ⚠️ Batch {i//batch_size} failed: {e}")
            # Add empty entries for failed batch
            for cell_id in batch_ids:
                all_features.append({'cell_id': cell_id})

    # Convert to DataFrame
    df = pd.DataFrame(all_features)

    # Merge with grid geometry
    grid_with_emb = grid.merge(df, on='cell_id', how='left')

    # Add metadata
    grid_with_emb['paddock'] = paddock_name
    grid_with_emb['year'] = year
    grid_with_emb['extraction_date'] = datetime.now().isoformat()

    # Calculate embedding statistics
    emb_cols = [c for c in grid_with_emb.columns if c.startswith('A')]
    valid_count = grid_with_emb[emb_cols[0]].notna().sum() if emb_cols else 0
    print(f"  → Extracted {valid_count}/{len(grid)} cells with valid embeddings")

    return grid_with_emb


def apply_clustering(
    gdf: gpd.GeoDataFrame,
    n_clusters: list = [10, 15, 20]
) -> gpd.GeoDataFrame:
    """Apply KMeans clustering to embeddings.

    Args:
        gdf: GeoDataFrame with embedding columns (A00-A63)
        n_clusters: List of cluster counts to compute

    Returns:
        GeoDataFrame with added cluster columns (cluster_10, cluster_15, cluster_20)
    """
    if not SKLEARN_AVAILABLE:
        print("⚠️ scikit-learn not available, skipping clustering")
        return gdf

    # Get embedding columns
    emb_cols = [c for c in gdf.columns if c.startswith('A') and c[1:].isdigit()]
    if len(emb_cols) == 0:
        print("⚠️ No embedding columns found for clustering")
        return gdf

    # Extract valid embeddings
    valid_mask = gdf[emb_cols[0]].notna()
    embeddings = gdf.loc[valid_mask, emb_cols].values

    if len(embeddings) < max(n_clusters):
        print(f"⚠️ Not enough valid embeddings ({len(embeddings)}) for clustering")
        return gdf

    print(f"\n📊 Applying KMeans clustering...")
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    for k in n_clusters:
        print(f"  → K={k} clusters...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(emb_scaled)

        col_name = f'cluster_{k}'
        gdf[col_name] = np.nan
        gdf.loc[valid_mask, col_name] = labels.astype(int)
        print(f"    ✓ Added column '{col_name}'")

    return gdf


def apply_pca(
    gdf: gpd.GeoDataFrame,
    n_components: int = 3
) -> gpd.GeoDataFrame:
    """Apply PCA to embeddings for visualization.

    Args:
        gdf: GeoDataFrame with embedding columns (A00-A63)
        n_components: Number of PCA components (default 3 for RGB visualization)

    Returns:
        GeoDataFrame with added PCA columns (pca_1, pca_2, pca_3)
    """
    if not SKLEARN_AVAILABLE:
        print("⚠️ scikit-learn not available, skipping PCA")
        return gdf

    # Get embedding columns
    emb_cols = [c for c in gdf.columns if c.startswith('A') and c[1:].isdigit()]
    if len(emb_cols) == 0:
        print("⚠️ No embedding columns found for PCA")
        return gdf

    # Extract valid embeddings
    valid_mask = gdf[emb_cols[0]].notna()
    embeddings = gdf.loc[valid_mask, emb_cols].values

    if len(embeddings) < n_components:
        print(f"⚠️ Not enough valid embeddings for PCA")
        return gdf

    print(f"\n📈 Applying PCA (n_components={n_components})...")
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(emb_scaled)

    # Add PCA columns
    for i in range(n_components):
        col_name = f'pca_{i+1}'
        gdf[col_name] = np.nan
        gdf.loc[valid_mask, col_name] = pca_result[:, i]

    # Normalize to 0-255 for RGB visualization
    for i in range(min(3, n_components)):
        col = f'pca_{i+1}'
        col_rgb = f'pca_{i+1}_rgb'
        valid_vals = gdf.loc[valid_mask, col]
        min_val, max_val = valid_vals.min(), valid_vals.max()
        gdf[col_rgb] = np.nan
        if max_val > min_val:
            gdf.loc[valid_mask, col_rgb] = ((valid_vals - min_val) / (max_val - min_val) * 255).astype(int)
        else:
            gdf.loc[valid_mask, col_rgb] = 128

    explained = pca.explained_variance_ratio_
    print(f"  ✓ Variance explained: {', '.join([f'PC{i+1}={v:.1%}' for i, v in enumerate(explained)])}")
    print(f"  ✓ Total: {sum(explained):.1%}")

    return gdf


def main():
    parser = argparse.ArgumentParser(
        description="Extract satellite embeddings for paddock zones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract 2025 embeddings for all paddocks
  python extract_embeddings.py --paddock all --year 2025

  # Extract with clustering and PCA
  python extract_embeddings.py --paddock all --year 2024 --cluster 10 15 20 --pca

  # Use custom boundaries file
  python extract_embeddings.py --boundaries historical.gpkg --year 2021

  # Extract multiple years
  python extract_embeddings.py --boundaries historical.gpkg --years 2020 2021
"""
    )
    parser.add_argument(
        "--paddock",
        type=str,
        default="all",
        help="Paddock name or 'all' (default: all)"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year for embeddings (default: 2024)"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs='+',
        help="Multiple years to extract (e.g., --years 2020 2021 2022)"
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=10,
        help="Grid cell size in meters (default: 10)"
    )
    parser.add_argument(
        "--boundaries",
        type=str,
        default=None,
        help="Custom boundaries file (GeoPackage/Shapefile)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated in LORENZO2)"
    )
    parser.add_argument(
        "--cluster",
        type=int,
        nargs='*',
        default=None,
        help="Apply KMeans clustering (e.g., --cluster 10 15 20)"
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        help="Apply PCA for RGB visualization"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for Earth Engine requests (default: 500)"
    )

    args = parser.parse_args()

    global GRID_SIZE
    GRID_SIZE = args.grid_size

    # Determine years to process
    if args.years:
        years = args.years
    elif args.year:
        years = [args.year]
    else:
        years = [2024]  # Default

    print("=" * 60)
    print("SATELLITE EMBEDDINGS EXTRACTION")
    print("=" * 60)
    print(f"Paddock: {args.paddock}")
    print(f"Year(s): {', '.join(map(str, years))}")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}m")
    if args.boundaries:
        print(f"Boundaries: {args.boundaries}")
    if args.cluster:
        print(f"Clustering: K={', '.join(map(str, args.cluster))}")
    if args.pca:
        print("PCA: enabled")
    print()

    # Initialize Earth Engine
    if not initialize_ee():
        sys.exit(1)

    # Load boundaries
    boundaries = load_paddock_boundaries(args.paddock, args.boundaries)

    # Get output directory
    output_dir = get_output_dir()

    # Process each year
    for year in years:
        print(f"\n{'#' * 60}")
        print(f"# YEAR: {year}")
        print(f"{'#' * 60}")

        # Process each paddock
        all_results = []

        for paddock_name in boundaries['paddock'].unique():
            paddock_gdf = boundaries[boundaries['paddock'] == paddock_name]

            result = extract_embeddings_for_paddock(
                paddock_gdf,
                paddock_name,
                year=year,
                batch_size=args.batch_size
            )
            all_results.append(result)

        # Combine all results
        combined = pd.concat(all_results, ignore_index=True)
        combined = gpd.GeoDataFrame(combined, crs="EPSG:32719")

        # Apply clustering if requested
        if args.cluster:
            clusters = args.cluster if args.cluster else [10, 15, 20]
            combined = apply_clustering(combined, clusters)

        # Apply PCA if requested
        if args.pca:
            combined = apply_pca(combined)

        # Save output
        if args.output and len(years) == 1:
            output_path = Path(args.output)
        else:
            paddock_suffix = args.paddock.lower().replace(" ", "_")
            output_path = output_dir / f"embeddings_{paddock_suffix}_{year}.gpkg"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_file(output_path, driver="GPKG")

        print()
        print("=" * 60)
        print(f"EXTRACTION COMPLETE - Year {year}")
        print("=" * 60)
        print(f"Output: {output_path}")
        print(f"Total cells: {len(combined)}")
        print(f"Paddocks: {combined['paddock'].nunique()}")

        # Summary by paddock
        print()
        print("Summary by paddock:")
        emb_col = 'A00' if 'A00' in combined.columns else combined.columns[0]
        summary = combined.groupby('paddock').agg({
            'cell_id': 'count',
            emb_col: lambda x: x.notna().sum()
        }).rename(columns={'cell_id': 'total_cells', emb_col: 'valid_embeddings'})
        print(summary.to_string())

    return combined


if __name__ == "__main__":
    main()
