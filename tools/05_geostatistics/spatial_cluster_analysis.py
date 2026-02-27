#!/usr/bin/env python
"""
Spatial Cluster Analysis (LISA) for Orobanche Detection Data
=============================================================

Perform Local Indicators of Spatial Association (LISA) analysis to identify
hot spots, cold spots, and spatial outliers in orobanche infestation data.

Usage:
    # Single paddock
    python scripts/stats/spatial_cluster_analysis.py \
        --input outputs/geo_exports/apalta_celular_20260108_linked.shp \
        --output apalta --mode single

    # Merged analysis (all files combined)
    python scripts/stats/spatial_cluster_analysis.py \
        --input "outputs/geo_exports/inia_sector*_linked.shp" \
        --output inia_rayentue --mode merged

    # Batch isolated (each paddock separately)
    python scripts/stats/spatial_cluster_analysis.py \
        --input "outputs/geo_exports/*_celular_*_linked.shp" \
        --output sugal_campos --mode batch

Author: INIA Deep Learning Project
Dependencies: pygeoda, geopandas, pandas
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import pandas as pd
import pygeoda

from ragweed_toolkit.spatial import LISA_COLORS

# Mapping from pygeoda cluster codes to ragweed_toolkit LISA_COLORS keys.
# pygeoda: 0=NS, 1=HH, 2=LL, 3=LH, 4=HL
_PYGEODA_TO_LISA_KEY = {0: "NS", 1: "HH", 2: "LL", 3: "LH", 4: "HL"}

# LISA cluster labels
CLUSTER_LABELS = {
    0: 'Not Significant',
    1: 'High-High (HOT SPOT)',
    2: 'Low-Low (COLD SPOT)',
    3: 'Low-High (Outlier)',
    4: 'High-Low (Outlier)',
    5: 'Isolated'
}


def find_orara_column(gdf: gpd.GeoDataFrame) -> Optional[str]:
    """Find the ORARA count column in the dataframe."""
    for col in ['nr_orara', 'ORARA', 'orara', 'nr_ORARA']:
        if col in gdf.columns:
            return col
    return None


# TODO: Replace with ragweed_toolkit.spatial.compute_bivariate_lisa when it
# supports univariate mode + pygeoda KNN weights (currently esda + DistanceBand only)
def run_lisa(gdf: gpd.GeoDataFrame, orara_col: str, k: int = 8) -> gpd.GeoDataFrame:
    """
    Run LISA analysis on a GeoDataFrame.

    Args:
        gdf: GeoDataFrame with point geometries (must be in metric CRS)
        orara_col: Name of the ORARA count column
        k: Number of nearest neighbors for spatial weights

    Returns:
        GeoDataFrame with LISA results added
    """
    # Create pygeoda object
    geoda_obj = pygeoda.open(gdf)

    # Adjust k if fewer points
    actual_k = min(k, len(gdf) - 1)
    if actual_k < k:
        print(f"  Note: Using k={actual_k} (fewer points than requested k={k})")

    # Create KNN weights
    w = pygeoda.knn_weights(geoda_obj, k=actual_k)

    # Run Local Moran's I
    orara_values = list(gdf[orara_col])
    lisa = pygeoda.local_moran(w, orara_values)

    # Add results to dataframe
    gdf = gdf.copy()
    gdf['lisa_cluster'] = lisa.lisa_clusters()
    gdf['lisa_pvalue'] = lisa.lisa_pvalues()
    gdf['lisa_moran_i'] = lisa.lisa_values()

    return gdf


def print_lisa_summary(gdf: gpd.GeoDataFrame, orara_col: str, title: str = ""):
    """Print summary statistics for LISA results."""
    if title:
        print(f"\n{'='*60}")
        print(title)
        print('='*60)

    total_points = len(gdf)
    total_orara = gdf[orara_col].sum()

    print(f"Total points: {total_points}")
    print(f"Total ORARA: {total_orara:,}")
    print(f"Mean ORARA: {gdf[orara_col].mean():.1f}")

    print(f"\n{'-'*60}")
    print("LISA Cluster Distribution:")
    print('-'*60)

    for cluster_code in sorted(gdf['lisa_cluster'].unique()):
        subset = gdf[gdf['lisa_cluster'] == cluster_code]
        count = len(subset)
        pct = 100 * count / total_points
        mean_orara = subset[orara_col].mean()
        sum_orara = subset[orara_col].sum()
        label = CLUSTER_LABELS.get(cluster_code, f'Type {cluster_code}')
        lisa_key = _PYGEODA_TO_LISA_KEY.get(cluster_code, "NS")
        color = LISA_COLORS.get(lisa_key, "#cccccc")

        print(f"\n{label} [{color}]:")
        print(f"  Points: {count} ({pct:.1f}%)")
        print(f"  Mean ORARA: {mean_orara:.1f}")
        print(f"  Total ORARA: {sum_orara:,}")


def analyze_single(input_path: str, output_name: str, k: int = 8) -> gpd.GeoDataFrame:
    """Analyze a single shapefile."""
    print(f"\nLoading: {input_path}")
    gdf = gpd.read_file(input_path)
    print(f"Loaded {len(gdf)} points, CRS: {gdf.crs}")

    orara_col = find_orara_column(gdf)
    if not orara_col:
        raise ValueError(f"No ORARA column found. Available: {gdf.columns.tolist()}")

    # Reproject to UTM Zone 19S (Chile)
    gdf_utm = gdf.to_crs("EPSG:32719")
    print(f"Reprojected to: {gdf_utm.crs}")

    # Run LISA
    print(f"\nRunning LISA analysis (k={k})...")
    gdf_result = run_lisa(gdf_utm, orara_col, k)

    # Print summary
    print_lisa_summary(gdf_result, orara_col, "LISA ANALYSIS RESULTS")

    # Export
    output_dir = Path("outputs/geo_exports")
    output_dir.mkdir(parents=True, exist_ok=True)

    gpkg_path = output_dir / f"{output_name}_lisa.gpkg"
    shp_path = output_dir / f"{output_name}_lisa.shp"

    gdf_result.to_file(gpkg_path, driver="GPKG")
    gdf_result.to_file(shp_path)

    print(f"\n{'='*60}")
    print("OUTPUT FILES")
    print('='*60)
    print(f"GeoPackage: {gpkg_path}")
    print(f"Shapefile: {shp_path}")

    return gdf_result


def analyze_merged(input_pattern: str, output_name: str, k: int = 8) -> gpd.GeoDataFrame:
    """Merge multiple shapefiles and analyze as one dataset."""
    input_files = sorted(Path(".").glob(input_pattern))

    if not input_files:
        raise ValueError(f"No files found matching: {input_pattern}")

    print(f"\nMerging {len(input_files)} files...")

    gdfs = []
    for f in input_files:
        gdf = gpd.read_file(f)
        gdf['source_file'] = f.stem
        gdfs.append(gdf)
        print(f"  + {f.name}: {len(gdf)} points")

    gdf_merged = pd.concat(gdfs, ignore_index=True)
    print(f"\nTotal merged: {len(gdf_merged)} points")

    orara_col = find_orara_column(gdf_merged)
    if not orara_col:
        raise ValueError(f"No ORARA column found. Available: {gdf_merged.columns.tolist()}")

    # Reproject
    gdf_utm = gdf_merged.to_crs("EPSG:32719")

    # Run LISA
    print(f"\nRunning LISA analysis (k={k})...")
    gdf_result = run_lisa(gdf_utm, orara_col, k)

    # Print summary
    print_lisa_summary(gdf_result, orara_col, "MERGED LISA ANALYSIS RESULTS")

    # Export
    output_dir = Path("outputs/geo_exports")
    gpkg_path = output_dir / f"{output_name}_merged_lisa.gpkg"
    shp_path = output_dir / f"{output_name}_merged_lisa.shp"

    gdf_result.to_file(gpkg_path, driver="GPKG")
    gdf_result.to_file(shp_path)

    print(f"\n{'='*60}")
    print("OUTPUT FILES")
    print('='*60)
    print(f"GeoPackage: {gpkg_path}")
    print(f"Shapefile: {shp_path}")

    return gdf_result


def analyze_batch(input_pattern: str, output_name: str, k: int = 8) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Analyze multiple paddocks separately (batch isolated mode).
    Each paddock is analyzed independently, then results are combined.
    """
    input_files = sorted(Path(".").glob(input_pattern))

    if not input_files:
        raise ValueError(f"No files found matching: {input_pattern}")

    print(f"\n{'='*60}")
    print("BATCH ISOLATED ANALYSIS")
    print(f"Analyzing {len(input_files)} paddocks separately")
    print('='*60)

    results = []
    summary_rows = []

    for shp_path in input_files:
        paddock_name = shp_path.stem.replace('_linked', '').replace('_', ' ').title()
        print(f"\n→ {paddock_name}")

        try:
            gdf = gpd.read_file(shp_path)
            orara_col = find_orara_column(gdf)

            if not orara_col:
                print("  ⚠ Skipping: No ORARA column")
                continue

            if len(gdf) < 10:
                print(f"  ⚠ Skipping: Too few points ({len(gdf)})")
                continue

            # Reproject and analyze
            gdf_utm = gdf.to_crs("EPSG:32719")
            gdf_result = run_lisa(gdf_utm, orara_col, k)
            gdf_result['paddock'] = paddock_name

            results.append(gdf_result)

            # Collect summary
            hotspots = len(gdf_result[gdf_result['lisa_cluster'] == 1])
            coldspots = len(gdf_result[gdf_result['lisa_cluster'] == 2])
            high_low = len(gdf_result[gdf_result['lisa_cluster'] == 4])
            significant = len(gdf_result[gdf_result['lisa_cluster'] > 0])

            summary_rows.append({
                'paddock': paddock_name,
                'n_points': len(gdf_result),
                'total_orara': int(gdf_result[orara_col].sum()),
                'mean_orara': round(gdf_result[orara_col].mean(), 1),
                'max_orara': int(gdf_result[orara_col].max()),
                'hot_spots': hotspots,
                'cold_spots': coldspots,
                'high_low_outliers': high_low,
                'pct_significant': round(100 * significant / len(gdf_result), 1)
            })

            print(f"  ✓ {len(gdf)} pts | {gdf_result[orara_col].sum():,.0f} ORARA | "
                  f"{hotspots} hot spots | {coldspots} cold spots")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    if not results:
        raise ValueError("No valid paddocks processed!")

    # Combine results
    combined = pd.concat(results, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    # Print summary table
    print(f"\n{'='*60}")
    print("BATCH ANALYSIS SUMMARY")
    print('='*60)
    print(summary.to_string(index=False))

    # Paddock ranking
    print(f"\n{'='*60}")
    print("PADDOCK RANKING (by hot spot severity)")
    print('='*60)

    ranked = summary.sort_values('hot_spots', ascending=False)
    for _, row in ranked.iterrows():
        if row['hot_spots'] > 50:
            severity = "🔴 HIGH"
        elif row['hot_spots'] > 10:
            severity = "🟡 MEDIUM"
        else:
            severity = "🟢 LOW"
        print(f"{severity} {row['paddock']}: {row['hot_spots']} hot spots, "
              f"{row['total_orara']:,} ORARA, {row['pct_significant']:.0f}% significant")

    # Export
    output_dir = Path("outputs/geo_exports")
    gpkg_path = output_dir / f"{output_name}_batch_lisa.gpkg"
    shp_path = output_dir / f"{output_name}_batch_lisa.shp"
    csv_path = output_dir / f"{output_name}_lisa_summary.csv"

    combined.to_file(gpkg_path, driver="GPKG")
    combined.to_file(shp_path)
    summary.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print("OUTPUT FILES")
    print('='*60)
    print(f"Combined GeoPackage: {gpkg_path}")
    print(f"Combined Shapefile: {shp_path}")
    print(f"Summary CSV: {csv_path}")

    return combined, summary


def generate_qgis_style_script(output_path: str, layer_name: str = "LISA Clusters"):
    """Generate a QGIS Python script for styling the results."""
    script = f'''"""
QGIS LISA Cluster Styling Script
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

Usage in QGIS Python Console (Ctrl+Alt+P):
    exec(open('{output_path.replace(".gpkg", "_style.py")}').read())
"""
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsCategorizedSymbolRenderer,
    QgsRendererCategory, QgsMarkerSymbol
)

LAYER_PATH = "{output_path}"
LAYER_NAME = "{layer_name}"

CLUSTERS = {{
    0: {{"label": "Not Significant", "color": "{LISA_COLORS['NS']}", "size": 2.5}},
    1: {{"label": "Hot Spot (High-High)", "color": "{LISA_COLORS['HH']}", "size": 4.0}},
    2: {{"label": "Cold Spot (Low-Low)", "color": "{LISA_COLORS['LL']}", "size": 3.5}},
    3: {{"label": "Low-High Outlier", "color": "{LISA_COLORS['LH']}", "size": 3.0}},
    4: {{"label": "High-Low Outlier", "color": "{LISA_COLORS['HL']}", "size": 3.5}},
}}

# Check if layer exists
existing = QgsProject.instance().mapLayersByName(LAYER_NAME)
if existing:
    layer = existing[0]
    print(f"Using existing layer: {{LAYER_NAME}}")
else:
    layer = QgsVectorLayer(LAYER_PATH, LAYER_NAME, "ogr")
    if not layer.isValid():
        raise ValueError(f"Could not load layer from {{LAYER_PATH}}")
    QgsProject.instance().addMapLayer(layer)
    print(f"Added layer: {{LAYER_NAME}} ({{layer.featureCount()}} features)")

# Find cluster field
cluster_field = None
for f in layer.fields():
    if f.name() in ['lisa_cluster', 'lisa_clust']:
        cluster_field = f.name()
        break

if not cluster_field:
    raise ValueError("No LISA cluster field found!")

# Create categories
categories = []
for code, props in CLUSTERS.items():
    symbol = QgsMarkerSymbol.createSimple({{
        'name': 'circle',
        'color': props['color'],
        'size': str(props['size']),
        'outline_color': '#000000',
        'outline_width': '0.2'
    }})
    categories.append(QgsRendererCategory(code, symbol, props['label']))

# Apply renderer
renderer = QgsCategorizedSymbolRenderer(cluster_field, categories)
layer.setRenderer(renderer)
layer.triggerRepaint()

print("LISA cluster styling applied!")
print("Legend: Red=Hot Spots, Blue=Cold Spots, Orange=Emerging Outbreaks")
'''

    style_path = output_path.replace(".gpkg", "_style.py").replace(".shp", "_style.py")
    with open(style_path, 'w') as f:
        f.write(script)

    print(f"QGIS style script: {style_path}")
    return style_path


def main():
    parser = argparse.ArgumentParser(
        description="Spatial Cluster Analysis (LISA) for Orobanche Detection Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single paddock analysis
  python scripts/stats/spatial_cluster_analysis.py \\
      --input outputs/geo_exports/apalta_celular_20260108_linked.shp \\
      --output apalta --mode single

  # Merged analysis (neighbors cross boundaries)
  python scripts/stats/spatial_cluster_analysis.py \\
      --input "outputs/geo_exports/inia_sector*_linked.shp" \\
      --output inia_rayentue --mode merged

  # Batch isolated (each paddock separately)
  python scripts/stats/spatial_cluster_analysis.py \\
      --input "outputs/geo_exports/*_celular_*_linked.shp" \\
      --output sugal_campos --mode batch
        """
    )

    parser.add_argument("--input", "-i", required=True,
                        help="Input shapefile path or glob pattern (use quotes for wildcards)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output name prefix (without extension)")
    parser.add_argument("--mode", "-m", choices=['single', 'merged', 'batch'],
                        default='single',
                        help="Analysis mode: single, merged, or batch (default: single)")
    parser.add_argument("--k", type=int, default=8,
                        help="Number of nearest neighbors for spatial weights (default: 8)")
    parser.add_argument("--no-style", action="store_true",
                        help="Skip generating QGIS style script")

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print("# SPATIAL CLUSTER ANALYSIS (pygeoda LISA)")
    print(f"# Mode: {args.mode.upper()}")
    print(f"# KNN neighbors: {args.k}")
    print(f"{'#'*60}")

    try:
        if args.mode == 'single':
            result = analyze_single(args.input, args.output, args.k)
            output_gpkg = f"outputs/geo_exports/{args.output}_lisa.gpkg"
        elif args.mode == 'merged':
            result = analyze_merged(args.input, args.output, args.k)
            output_gpkg = f"outputs/geo_exports/{args.output}_merged_lisa.gpkg"
        elif args.mode == 'batch':
            result, summary = analyze_batch(args.input, args.output, args.k)
            output_gpkg = f"outputs/geo_exports/{args.output}_batch_lisa.gpkg"

        # Generate QGIS style script
        if not args.no_style:
            print(f"\n{'-'*60}")
            generate_qgis_style_script(output_gpkg, f"LISA {args.output}")

        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print('='*60)
        print("\nTo visualize in QGIS:")
        print(f"  1. Open: {output_gpkg}")
        print("  2. Run style script in Python Console (Ctrl+Alt+P)")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
