# Tutorial: Spatial Mapping

This tutorial demonstrates the geostatistical mapping pipeline: fitting semivariograms, interpolating weed density with ordinary kriging, identifying hot spots with bivariate LISA, and modelling spatially varying relationships with GWR.

**Corresponding notebook:** `notebooks/03_spatial_mapping.ipynb`

---

## Step 1: Semivariogram Analysis

The experimental semivariogram quantifies how spatial dependence changes with distance. Fit an exponential model:

$$\gamma(h) = C_0 + C \cdot [1 - \exp(-h / a)]$$

where C0 = nugget, C = partial sill, a = range.

```python
from ragweed_toolkit.geostatistics import (
    compute_experimental_variogram,
    fit_exponential_model,
)

# coords: Nx2 array of point locations
# values: N-length array of weed counts
lag_centers, gamma, pair_counts = compute_experimental_variogram(
    coords, values, n_lags=12, max_lag=200.0
)

model = fit_exponential_model(lag_centers, gamma)
print(f"Nugget: {model.nugget:.1f}")
print(f"Partial sill: {model.partial_sill:.1f}")
print(f"Range: {model.range_param:.1f} m")
```

Visualize the fitted variogram:

```python
from ragweed_toolkit.geostatistics import plot_variogram

plot_variogram(model, title="Ragweed Density Semivariogram")
```

## Step 2: Ordinary Kriging

Interpolate point observations onto a regular grid clipped to the study area boundary. The output includes both predicted values and prediction variance.

```python
from ragweed_toolkit.geostatistics import ordinary_kriging

grid = ordinary_kriging(
    gdf_observations,
    value_col="nr_ambel",
    boundary=field_boundary_gdf,
    resolution=5.0,        # 5m grid cells
    variogram_model=model,
)
# grid columns: predicted, variance, geometry
```

Or use the CLI:

```bash
ragweed-kriging \
    --points detections.gpkg \
    --value-col nr_ambel \
    --boundary field_boundary.gpkg \
    --output kriged.gpkg \
    --resolution 5.0
```

Export for QGIS:

```python
grid.to_file("kriged_density.gpkg", driver="GPKG")
```

## Step 3: Bivariate LISA Cluster Analysis

LISA (Local Indicators of Spatial Association) identifies statistically significant spatial clusters where two variables co-occur. In the chapter, PC1 (satellite) x AMBEL density revealed 68% significant pixels.

```python
from ragweed_toolkit.spatial import (
    compute_bivariate_morans,
    compute_bivariate_lisa,
    lisa_summary,
)

# Global test
bv = compute_bivariate_morans(
    gdf, var_x="PC1", var_y="AMBEL_density",
    threshold=15.0, permutations=999,
)
print(f"Bivariate Moran's I = {bv.I:.3f} (p = {bv.p_value:.4f})")

# Local decomposition
lisa_gdf = compute_bivariate_lisa(
    gdf, var_x="PC1", var_y="AMBEL_density",
    threshold=15.0, permutations=999, alpha=0.05,
)
# lisa_gdf has columns: cluster, local_I, p_value, + original columns
```

Or use the CLI:

```bash
ragweed-lisa \
    --input spatial_data.gpkg \
    --var-x PC1 \
    --var-y AMBEL_density \
    --output lisa_results.gpkg
```

### Cluster Interpretation

| Cluster | Color | Meaning | Management Action |
|---------|-------|---------|-------------------|
| HH (Hot spot) | Red | High satellite index, high weed density | Priority treatment |
| LL (Cold spot) | Blue | Low satellite, low density | Monitoring only |
| HL (Outlier) | Orange | High satellite, low density neighbors | Investigate |
| LH (Outlier) | Purple | Low satellite, high density neighbors | Emerging outbreak |

Print a summary:

```python
summary = lisa_summary(lisa_gdf)
for cluster, stats in summary.items():
    print(f"{cluster}: {stats['count']} ({stats['pct']}%)")
```

## Step 4: GWR vs OLS Comparison

Geographically Weighted Regression (GWR) allows regression coefficients to vary spatially, capturing local relationships that OLS misses.

```python
from ragweed_toolkit.spatial import compare_ols_gwr

comparison = compare_ols_gwr(
    gdf,
    y_col="AMBEL_density",
    x_cols=["PC1", "PC2", "PC3"],
)

print(f"OLS R-squared: {comparison.ols.r2:.3f}")
print(f"GWR R-squared: {comparison.gwr.r2:.3f}")
print(f"GWR bandwidth: {comparison.gwr.bandwidth:.0f} neighbors")
print(f"R2 improvement: +{comparison.r2_improvement:.3f}")
```

In the chapter, GWR improved R-squared from 0.683 (OLS) to 0.882 (GWR) for AMBEL and from 0.714 to 0.910 for LENCU.

## What's Next

- **[Satellite Integration](satellite.md)** -- Connect satellite spectral indices to the spatial pipeline
- **[Detection Workflow](detection.md)** -- Generate the point data that feeds into this analysis
