#!/usr/bin/env python
"""
Bivariate LISA Analysis: Orobanche Counts vs Soil Indices
==========================================================

This script performs bivariate Local Moran's I analysis to identify spatial
correlations between Orobanche infestation levels and soil properties.

Scientific Basis (Anselin, 2019):
- High-High: High Orobanche near high soil index (positive spatial correlation)
- Low-Low: Low Orobanche near low soil index (positive correlation)
- High-Low: High Orobanche near low soil index (negative correlation)
- Low-High: Low Orobanche near high soil index (negative correlation)

Usage:
    python scripts/stats/bivariate_lisa_soil.py

Output:
    outputs/orobanche_model/bivariate_lisa_results.gpkg
    outputs/orobanche_model/bivariate_lisa_summary.csv

Author: INIA Deep Learning Project
Dependencies: pygeoda, geopandas, rasterio
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.errors import RasterioIOError
import pygeoda
from datetime import datetime

from ragweed_toolkit.spatial import LISA_COLORS, QUADRANT_MAP, lisa_summary

# Mapping from pygeoda cluster codes to ragweed_toolkit LISA keys
_PYGEODA_TO_LISA_KEY = {0: "NS", 1: "HH", 2: "LL", 3: "LH", 4: "HL"}


# =============================================================================
# Configuration
# =============================================================================

# Input LISA GeoPackage (with univariate LISA results)
LISA_INPUT = Path("outputs/geo_exports/sugal_campos_feb2026_batch_lisa.gpkg")

# Output directory
OUTPUT_DIR = Path("outputs/orobanche_model")

# Soil indices to analyze
SOIL_INDICES = [
    "Clay_Index",
    "S2WI",
    "Brightness",
    "BSI",
    "NBR2",
    "SWIR_diff"
]

# Base path for soil rasters
SOIL_RASTER_BASE = Path("/media/malezainia1/LORENZO/outputs_sugal_25-26/soil_characterization")

# Mapping from paddock prefix to soil raster directory
# (Some datasets may not have matching soil rasters)
PADDOCK_TO_RASTER = {
    "Apalta": "apalta",
    "Camarico": "camarico",
    "El Carmen": "el_carmen",
    "La Capilla": "la_capilla",
    "Las Pilastras": "las_pilastras",
    "Lo Castillo": "lo_castillo",
    "Los Galpones": "los_galpones",
    "Parcela 5": "parcela_5",
    "Santa Eugenia": "santa_eugenia",
    "Santa Ines": "santa_ines",
}

# Bivariate LISA cluster labels (derived from ragweed_toolkit.spatial.QUADRANT_MAP)
_LABEL_EXPANSION = {"HH": "High-High", "LL": "Low-Low", "LH": "Low-High",
                     "HL": "High-Low", "NS": "Not Significant"}
BIMORAN_LABELS = {code: _LABEL_EXPANSION.get(key, "Isolated")
                  for code, key in _PYGEODA_TO_LISA_KEY.items()}
BIMORAN_LABELS[5] = 'Isolated'


def get_paddock_prefix(paddock_name: str) -> str:
    """Extract paddock prefix for raster lookup."""
    # "Apalta Celular 20260108" -> "Apalta"
    # "Lo Castillo Celular 20260107" -> "Lo Castillo"
    # "Santa Ines Celular 20260106" -> "Santa Ines"
    # "Las Pilastras Celular 20260123" -> "Las Pilastras"

    # Known two-word paddock names
    two_word_prefixes = ["Lo Castillo", "Santa Ines", "Santa Eugenia",
                         "Las Pilastras", "El Carmen", "Los Galpones",
                         "La Capilla", "Parcela 5"]

    for prefix in two_word_prefixes:
        if paddock_name.startswith(prefix):
            return prefix

    # Single word prefix
    parts = paddock_name.split()
    if len(parts) >= 1:
        return parts[0]
    return paddock_name


def get_raster_path(paddock_prefix: str, soil_index: str) -> Path:
    """Construct the raster path for a given paddock and soil index."""
    raster_dir = PADDOCK_TO_RASTER.get(paddock_prefix)
    if not raster_dir:
        return None

    # Convert directory name to raster name format
    # e.g., "lo_castillo" -> "Lo_Castillo"
    name_parts = raster_dir.split("_")
    name_formatted = "_".join([p.capitalize() for p in name_parts])

    raster_path = SOIL_RASTER_BASE / raster_dir / "clipped_buffered" / f"{name_formatted}_{soil_index}_clipped.tif"

    return raster_path if raster_path.exists() else None


def sample_raster_at_points(gdf: gpd.GeoDataFrame, raster_path: Path) -> np.ndarray:
    """Sample raster values at point locations."""
    values = np.full(len(gdf), np.nan)

    try:
        with rasterio.open(raster_path) as src:
            # Reproject points to raster CRS if needed
            gdf_reproj = gdf.to_crs(src.crs)

            coords = [(geom.x, geom.y) for geom in gdf_reproj.geometry]

            # Sample raster
            sampled = list(src.sample(coords))
            values = np.array([v[0] if len(v) > 0 else np.nan for v in sampled])

            # Handle nodata
            if src.nodata is not None:
                values[values == src.nodata] = np.nan

    except RasterioIOError as e:
        print(f"    Warning: Could not read raster {raster_path}: {e}")

    return values


# TODO: Replace with ragweed_toolkit.spatial.compute_bivariate_lisa when it
# supports pygeoda KNN weights and NaN masking (currently esda + DistanceBand only)
def run_bivariate_lisa(gdf: gpd.GeoDataFrame, y_col: str, x_col: str, k: int = 8) -> dict:
    """
    Run Bivariate Local Moran's I analysis.

    Tests spatial correlation between y (Orobanche) at location i
    and x (soil index) at neighboring locations.

    Args:
        gdf: GeoDataFrame with point geometries
        y_col: Column name for dependent variable (e.g., 'nr_orara')
        x_col: Column name for independent variable (e.g., 'Clay_Index')
        k: Number of nearest neighbors

    Returns:
        Dictionary with cluster, p-value, and Moran I arrays
    """
    # Filter out rows with NaN values
    valid_mask = gdf[y_col].notna() & gdf[x_col].notna()
    gdf_valid = gdf[valid_mask].copy().reset_index(drop=True)

    if len(gdf_valid) < 20:
        return None

    # Create pygeoda object - reset index is required for pygeoda
    geoda_obj = pygeoda.open(gdf_valid)

    # Adjust k if needed
    actual_k = min(k, len(gdf_valid) - 1)

    # Create KNN weights
    w = pygeoda.knn_weights(geoda_obj, k=actual_k)

    # Run Bivariate Local Moran's I
    y_values = list(gdf_valid[y_col])
    x_values = list(gdf_valid[x_col])

    lisa = pygeoda.local_bimoran(w, y_values, x_values)

    # Map results back to original indices
    # Since we use .copy() for gdf_paddock in main(), the indices are preserved
    # We return arrays that can be assigned directly to the boolean mask
    clusters = np.full(len(gdf), np.nan)
    pvalues = np.full(len(gdf), np.nan)
    moran_i = np.full(len(gdf), np.nan)

    # Get original positions of valid rows
    valid_positions = np.where(valid_mask.values)[0]
    lisa_clusters = lisa.lisa_clusters()
    lisa_pvalues = lisa.lisa_pvalues()
    lisa_values = lisa.lisa_values()

    for i, pos in enumerate(valid_positions):
        clusters[pos] = lisa_clusters[i]
        pvalues[pos] = lisa_pvalues[i]
        moran_i[pos] = lisa_values[i]

    return {
        'clusters': clusters,
        'pvalues': pvalues,
        'moran_i': moran_i
    }


def print_bivariate_summary(gdf: gpd.GeoDataFrame, soil_index: str, cluster_col: str):
    """Print summary of bivariate LISA results."""
    valid = gdf[cluster_col].notna()
    gdf_valid = gdf[valid]

    if len(gdf_valid) == 0:
        return

    print(f"\n  {soil_index}:")
    print(f"    Valid points: {len(gdf_valid)} / {len(gdf)}")

    for code in [1, 2, 3, 4]:
        subset = gdf_valid[gdf_valid[cluster_col] == code]
        if len(subset) > 0:
            label = BIMORAN_LABELS.get(code, f'Type {code}')
            lisa_key = _PYGEODA_TO_LISA_KEY.get(code, "NS")
            color = LISA_COLORS.get(lisa_key, "#cccccc")
            pct = 100 * len(subset) / len(gdf_valid)
            mean_orara = subset['nr_orara'].mean()
            print(f"    {label} [{color}]: {len(subset)} pts ({pct:.1f}%), mean ORARA={mean_orara:.1f}")


def main():
    print("="*70)
    print("BIVARIATE LISA ANALYSIS")
    print("Orobanche Counts vs Soil Indices")
    print("="*70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load LISA GeoPackage
    print(f"\nLoading: {LISA_INPUT}")
    gdf = gpd.read_file(LISA_INPUT)
    print(f"  Loaded {len(gdf)} points")
    print(f"  CRS: {gdf.crs}")

    # Get unique paddocks
    paddocks = [p for p in gdf['paddock'].unique() if p is not None]
    print(f"  Paddocks: {len(paddocks)}")

    # Initialize result columns for each soil index
    for idx in SOIL_INDICES:
        gdf[f'{idx}'] = np.nan  # Sampled soil value
        gdf[f'lisa_{idx.lower()}'] = np.nan  # Bivariate cluster
        gdf[f'pval_{idx.lower()}'] = np.nan  # P-value
        gdf[f'moran_{idx.lower()}'] = np.nan  # Local Moran's I

    # Process each paddock
    print("\n" + "-"*70)
    print("SAMPLING SOIL INDICES FROM RASTERS")
    print("-"*70)

    for paddock in sorted(paddocks):
        prefix = get_paddock_prefix(paddock)
        paddock_mask = gdf['paddock'] == paddock
        n_points = paddock_mask.sum()

        print(f"\n{paddock} ({n_points} points)")

        for soil_idx in SOIL_INDICES:
            raster_path = get_raster_path(prefix, soil_idx)

            if raster_path is None:
                print(f"  {soil_idx}: No raster found")
                continue

            if not raster_path.exists():
                print(f"  {soil_idx}: File missing")
                continue

            # Sample raster at points
            gdf_paddock = gdf[paddock_mask].copy()
            values = sample_raster_at_points(gdf_paddock, raster_path)

            # Store sampled values
            gdf.loc[paddock_mask, soil_idx] = values

            valid_count = np.sum(~np.isnan(values))
            print(f"  {soil_idx}: {valid_count}/{n_points} valid samples")

    # Run bivariate LISA for each paddock and soil index
    print("\n" + "-"*70)
    print("RUNNING BIVARIATE LISA ANALYSIS")
    print("-"*70)

    summary_rows = []

    for paddock in sorted(paddocks):
        paddock_mask = gdf['paddock'] == paddock
        gdf_paddock = gdf[paddock_mask].copy()
        n_points = len(gdf_paddock)

        if n_points < 20:
            print(f"\n{paddock}: Skipping (only {n_points} points)")
            continue

        print(f"\n{paddock} ({n_points} points)")

        for soil_idx in SOIL_INDICES:
            col_name = soil_idx
            cluster_col = f'lisa_{soil_idx.lower()}'
            pval_col = f'pval_{soil_idx.lower()}'
            moran_col = f'moran_{soil_idx.lower()}'

            # Check if we have soil data
            valid_soil = gdf_paddock[col_name].notna().sum()
            if valid_soil < 20:
                print(f"  {soil_idx}: Insufficient data ({valid_soil} valid)")
                continue

            # Run bivariate LISA
            result = run_bivariate_lisa(gdf_paddock, 'nr_orara', col_name, k=8)

            if result is None:
                print(f"  {soil_idx}: Analysis failed")
                continue

            # Store results back to main GeoDataFrame
            gdf.loc[paddock_mask, cluster_col] = result['clusters']
            gdf.loc[paddock_mask, pval_col] = result['pvalues']
            gdf.loc[paddock_mask, moran_col] = result['moran_i']

            # Compute summary statistics
            valid_mask = ~np.isnan(result['clusters'])
            clusters = result['clusters'][valid_mask]
            moran_values = result['moran_i'][valid_mask]

            # Count clusters
            high_high = np.sum(clusters == 1)
            low_low = np.sum(clusters == 2)
            low_high = np.sum(clusters == 3)
            high_low = np.sum(clusters == 4)
            significant = high_high + low_low + low_high + high_low

            # Global Moran's I equivalent (mean of local values)
            mean_moran = np.nanmean(moran_values)

            summary_rows.append({
                'paddock': paddock,
                'soil_index': soil_idx,
                'n_valid': int(np.sum(valid_mask)),
                'mean_moran_i': round(mean_moran, 4),
                'high_high': int(high_high),
                'low_low': int(low_low),
                'low_high': int(low_high),
                'high_low': int(high_low),
                'pct_significant': round(100 * significant / np.sum(valid_mask), 1) if np.sum(valid_mask) > 0 else 0
            })

            print_bivariate_summary(gdf[paddock_mask], soil_idx, cluster_col)

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows)

    # Print overall summary by soil index
    print("\n" + "="*70)
    print("SOIL INDEX CORRELATION SUMMARY")
    print("="*70)
    print("\nMean Moran's I by Soil Index (positive = positive correlation):")

    idx_summary = summary_df.groupby('soil_index').agg({
        'mean_moran_i': 'mean',
        'high_high': 'sum',
        'low_low': 'sum',
        'low_high': 'sum',
        'high_low': 'sum',
        'pct_significant': 'mean'
    }).round(3)

    idx_summary['total_pos_corr'] = idx_summary['high_high'] + idx_summary['low_low']
    idx_summary['total_neg_corr'] = idx_summary['high_low'] + idx_summary['low_high']

    print("\n" + idx_summary.sort_values('mean_moran_i', ascending=False).to_string())

    # Interpretation
    print("\n" + "-"*70)
    print("INTERPRETATION")
    print("-"*70)
    print("""
High-High: Areas with HIGH Orobanche surrounded by HIGH soil index values
  -> Soil condition may PROMOTE infestation

High-Low: Areas with HIGH Orobanche surrounded by LOW soil index values
  -> Soil condition may NOT be the primary driver

Low-High: Areas with LOW Orobanche surrounded by HIGH soil index values
  -> Soil condition alone does not cause infestation

Low-Low: Areas with LOW Orobanche surrounded by LOW soil index values
  -> Both Orobanche and soil index are low together
""")

    # Save results
    output_gpkg = OUTPUT_DIR / "bivariate_lisa_results.gpkg"
    output_csv = OUTPUT_DIR / "bivariate_lisa_summary.csv"

    print("\n" + "-"*70)
    print("SAVING RESULTS")
    print("-"*70)

    gdf.to_file(output_gpkg, driver="GPKG")
    print(f"  GeoPackage: {output_gpkg}")

    summary_df.to_csv(output_csv, index=False)
    print(f"  Summary CSV: {output_csv}")

    # Also save the index summary
    idx_summary_path = OUTPUT_DIR / "bivariate_lisa_index_summary.csv"
    idx_summary.to_csv(idx_summary_path)
    print(f"  Index Summary: {idx_summary_path}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    main()
