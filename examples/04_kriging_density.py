"""
Example 04: Variogram Fitting and Kriging Interpolation
========================================================

Demonstrates the geostatistical pipeline from Chapter 5 (Section 5.4.2):

1. Generate synthetic spatial point data (ragweed density at known locations)
2. Compute the experimental semivariogram
3. Fit an exponential variogram model
4. Print fitted parameters (nugget, sill, range)

In production, the fitted variogram feeds into ordinary_kriging() to
produce a continuous density surface over a polygon boundary.
"""

import numpy as np

from ragweed_toolkit.geostatistics.variograms import (
    compute_experimental_variogram,
    fit_exponential_model,
)

np.random.seed(42)

# ------------------------------------------------------------------ #
# 1.  Generate synthetic spatial data
# ------------------------------------------------------------------ #
# Simulate 100 observation points in a 200m x 200m field
n_points = 100
coords = np.column_stack([
    np.random.uniform(0, 200, n_points),   # easting (m)
    np.random.uniform(0, 200, n_points),   # northing (m)
])

# Simulate spatially correlated ragweed counts with a hot spot
distances_to_hotspot = np.sqrt(
    (coords[:, 0] - 120) ** 2 + (coords[:, 1] - 80) ** 2
)
# Higher counts near the hot spot + random noise
counts = np.maximum(0, 50 * np.exp(-distances_to_hotspot / 60) +
                    np.random.randn(n_points) * 10)

print(f"=== Synthetic Data ===")
print(f"  Points          : {n_points}")
print(f"  Field size      : 200m x 200m")
print(f"  Count range     : {counts.min():.0f} - {counts.max():.0f}")
print(f"  Count mean      : {counts.mean():.1f}")

# ------------------------------------------------------------------ #
# 2.  Compute experimental semivariogram
# ------------------------------------------------------------------ #
lag_centers, gamma, pair_counts = compute_experimental_variogram(
    coords, counts,
    n_lags=12,
    max_lag=120.0,  # analyse up to 120m
)

print(f"\n=== Experimental Variogram ===")
print(f"  {'Lag (m)':>10} {'Semivariance':>14} {'Pairs':>8}")
print(f"  {'-'*10} {'-'*14} {'-'*8}")
for lag, g, c in zip(lag_centers, gamma, pair_counts):
    g_str = f"{g:.1f}" if not np.isnan(g) else "NaN"
    print(f"  {lag:>10.1f} {g_str:>14} {c:>8d}")

# ------------------------------------------------------------------ #
# 3.  Fit exponential variogram model
# ------------------------------------------------------------------ #
model = fit_exponential_model(lag_centers, gamma)

print(f"\n=== Fitted Exponential Model ===")
print(f"  Nugget (C0)     : {model.nugget:.1f}")
print(f"  Partial sill (C): {model.partial_sill:.1f}")
print(f"  Sill (C0+C)     : {model.sill:.1f}")
print(f"  Range (a)       : {model.range_param:.1f} m")

# ------------------------------------------------------------------ #
# 4.  Predict semivariance at a specific distance
# ------------------------------------------------------------------ #
test_distance = 50.0
predicted_gamma = model.predict(test_distance)
print(f"\n  gamma({test_distance:.0f}m) = {predicted_gamma:.1f}")

# ------------------------------------------------------------------ #
# 5.  Kriging usage (requires GeoDataFrame + boundary polygon)
# ------------------------------------------------------------------ #
print(f"\n=== Kriging Usage (not run here) ===")
print("""
  import geopandas as gpd
  from ragweed_toolkit.geostatistics import ordinary_kriging

  grid = ordinary_kriging(
      gdf=point_observations,     # GeoDataFrame with 'density' column
      value_col='density',
      boundary=field_boundary,    # polygon GeoDataFrame
      resolution=5.0,             # 5m grid cells
      variogram_model=model,      # from fit_exponential_model()
  )
  # grid has 'predicted' and 'variance' columns
""")
