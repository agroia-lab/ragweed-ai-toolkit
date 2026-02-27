"""
Example 05: Bivariate LISA Cluster Analysis
=============================================

Demonstrates the spatial autocorrelation pipeline from Chapter 5
(Section 5.4.1):

1. Generate synthetic spatial data with two correlated variables
   (e.g., satellite PC1 and ragweed density)
2. Compute bivariate global Moran's I
3. Decompose into LISA clusters (HH, HL, LH, LL, NS)

Requires: esda, libpysal, geopandas
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from ragweed_toolkit.spatial.lisa import (
    LISA_COLORS,
    compute_bivariate_lisa,
    lisa_summary,
)
from ragweed_toolkit.spatial.morans import compute_bivariate_morans

np.random.seed(42)

# ------------------------------------------------------------------ #
# 1.  Generate synthetic spatial data (projected CRS, metres)
# ------------------------------------------------------------------ #
n = 150
x = np.random.uniform(0, 500, n)
y = np.random.uniform(0, 500, n)

# Variable 1: satellite-derived vegetation index (e.g., PRESTO PC1)
pc1 = np.sin(x / 80) * np.cos(y / 100) + np.random.randn(n) * 0.3

# Variable 2: ragweed density (spatially correlated with PC1)
density = 0.7 * pc1 + 0.3 * np.random.randn(n)

gdf = gpd.GeoDataFrame(
    {"pc1": pc1, "density": density},
    geometry=[Point(xi, yi) for xi, yi in zip(x, y)],
    crs="EPSG:32719",  # UTM Zone 19S (Chile)
)

print("=== Synthetic Data ===")
print(f"  Points     : {n}")
print("  Area       : 500m x 500m")
print(f"  PC1 range  : [{pc1.min():.2f}, {pc1.max():.2f}]")
print(f"  Density    : [{density.min():.2f}, {density.max():.2f}]")

# ------------------------------------------------------------------ #
# 2.  Global bivariate Moran's I
# ------------------------------------------------------------------ #
print("\n=== Bivariate Moran's I (PC1 x Density) ===")
bv_result = compute_bivariate_morans(
    gdf, var_x="pc1", var_y="density",
    threshold=60.0,      # distance band in metres
    permutations=999,
)
print(f"  I          = {bv_result.I:.3f}")
print(f"  z-score    = {bv_result.z_score:.3f}")
print(f"  p-value    = {bv_result.p_value:.4f}")
print(f"  Significant: {bv_result.significant} ({bv_result.significance_stars})")
print(f"  n          = {bv_result.n}")

# ------------------------------------------------------------------ #
# 3.  LISA decomposition
# ------------------------------------------------------------------ #
print("\n=== LISA Clusters ===")
lisa_gdf = compute_bivariate_lisa(
    gdf, var_x="pc1", var_y="density",
    threshold=60.0,
    permutations=999,
    alpha=0.05,
)

summary = lisa_summary(lisa_gdf)
for cluster, stats in summary.items():
    color = LISA_COLORS[cluster]
    print(f"  {cluster:3s} : {stats['count']:4d} points ({stats['pct']:5.1f}%)  {color}")

# ------------------------------------------------------------------ #
# 4.  Interpretation guide
# ------------------------------------------------------------------ #
print("\n=== Cluster Interpretation ===")
print("  HH (red)    : Hot spot -- high PC1 surrounded by high density")
print("  LL (blue)   : Cold spot -- low PC1 surrounded by low density")
print("  HL (orange) : Spatial outlier -- high PC1, low density neighbors")
print("  LH (purple) : Spatial outlier -- low PC1, high density neighbors")
print("  NS (gray)   : Not significant at alpha=0.05")
