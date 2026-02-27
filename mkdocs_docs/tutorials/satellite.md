# Tutorial: Satellite Integration

This tutorial demonstrates the satellite remote sensing pipeline: computing spectral indices from Sentinel-2, extracting PRESTO embeddings, building risk zone maps, and correlating satellite features with ground-truth weed density.

**Corresponding notebook:** `notebooks/04_satellite_integration.ipynb`

---

## Background

In the chapter, single-date NDVI correlated r = 0.837 with AMBEL density and r = 0.890 with LENCU density. PRESTO foundation model embeddings (128-D from 12 months of Sentinel-2) achieved PC1 x AMBEL r = 0.739. These strong correlations support satellite-based early warning for weed management.

## Step 1: Compute Spectral Indices

The toolkit computes 9 spectral indices from a 10-band Sentinel-2 stack (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12).

```python
from ragweed_toolkit.satellite import compute_spectral_indices

# spectral: np.array with shape (10, H, W)
indices = compute_spectral_indices(spectral)
# Returns dict: {NDVI, EVI, GNDVI, RENDVI, S2WI, NBR2, BSI, Clay, SWIRd}
```

Each index characterizes a different surface property:

| Index | Description |
|-------|-------------|
| NDVI | Vegetation density |
| EVI | Atmosphere-corrected vegetation |
| GNDVI | Chlorophyll content |
| RENDVI | Phenological state |
| S2WI | Soil moisture |
| NBR2 | Moisture and crop residue |
| BSI | Bare soil fraction |
| Clay | Clay minerals (SWIR ratio) |
| SWIRd | Surface texture proxy |

Or use the CLI to process a GeoTIFF:

```bash
ragweed-indices --input sentinel2_stack.tif --output-dir indices/
```

## Step 2: Earth Engine Export (Pattern)

The toolkit wraps Earth Engine for exporting Sentinel-2 NDVI and RGB composites. This requires `ee.Authenticate()`.

```python
from ragweed_toolkit.satellite import export_composites
import ee

ee.Initialize()

paddock = ee.Geometry.Rectangle([-70.85, -34.20, -70.75, -34.10])

export_composites(
    paddock,
    start_date="2025-10-01",
    end_date="2026-03-31",
    paddock_name="Santa_Ines",
    drive_folder="satellite_exports",
)
```

## Step 3: PRESTO Embeddings (Pattern)

PRESTO is a satellite foundation model that compresses 12 months of Sentinel-2 data into 128-D embeddings per pixel. This requires the `presto-worldcereal` package.

```python
from ragweed_toolkit.satellite import (
    extract_presto_embeddings,
    apply_pca,
    apply_clustering,
)

# Extract embeddings at 10m resolution
gdf_embeddings = extract_presto_embeddings(
    paddock_gdf=paddock_boundary,
    season="25-26",
    grid_size=10,
)

# Reduce to 3 principal components
gdf_pca = apply_pca(gdf_embeddings, n_components=3)

# Cluster into management zones
gdf_clustered = apply_clustering(gdf_pca, n_clusters=5)
```

## Step 4: Risk Zone Classification

The risk zone classifier combines NDVI statistics (max, std, slope, decline) into a weighted score and assigns each pixel to Low / Medium / High risk.

```python
from ragweed_toolkit.satellite.ndvi import create_risk_zones

# gdf_ndvi must have columns: ndvi_max, ndvi_std, ndvi_slope, ndvi_decline
gdf_risk = create_risk_zones(gdf_ndvi)
# Adds columns: risk_score, risk_zone
```

Risk zone interpretation:

| Zone | Score Range | Description |
|------|------------|-------------|
| Low | Bottom third | Healthy vegetation, low weed pressure |
| Medium | Middle third | Moderate stress, routine monitoring |
| High | Top third | Vegetation depression, priority scouting |

## Step 5: Correlation Analysis

Link satellite features to ground-truth weed observations:

```python
import numpy as np

r_ambel = np.corrcoef(gdf["ndvi"], gdf["AMBEL_density"])[0, 1]
r_lencu = np.corrcoef(gdf["ndvi"], gdf["LENCU_density"])[0, 1]

print(f"NDVI vs AMBEL: r = {r_ambel:.3f}")
print(f"NDVI vs LENCU: r = {r_lencu:.3f}")
```

Chapter reference values:

| Comparison | Correlation |
|-----------|-------------|
| NDVI vs AMBEL (single-date Sept 2024) | r = 0.837 |
| NDVI vs LENCU (single-date Sept 2024) | r = 0.890 |
| PRESTO PC1 vs AMBEL (Jul-Dec 2024) | r = 0.739 |
| PRESTO PC1 vs LENCU (Jul-Dec 2024) | r = 0.717 |

## Multi-Scale Integration

The key insight from the chapter: no single scale solves the problem.

```
SATELLITE (10 m)  -->  DRONE (1-5 cm)  -->  DETECTION (mm)  -->  ADVISORY
  zone delineation      orthomosaic         species ID           management
  NDVI, PRESTO          ODM                 YOLO + SAHI          risk zones
```

Feed LISA clusters and risk zones from this tutorial into the spatial analysis pipeline for actionable management recommendations.

## What's Next

- **[Spatial Mapping](spatial.md)** -- Use satellite features in LISA and GWR analysis
- **[Detection Workflow](detection.md)** -- Field-level detection that validates satellite predictions
