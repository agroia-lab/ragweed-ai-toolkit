# Evidence 06: Satellite Embeddings x Weed Density Analysis

**Study site:** Santa Rosa paddock, lentil (*Lens culinaris*), Central Chile
**Area:** ~3.4 ha | **CRS:** EPSG:32719 (UTM Zone 19S) | **Center:** -36.532 S, -71.913 W
**Analysis date:** 2026-02-18

---

## Overview

This evidence package documents the satellite-to-weed-density correlation pipeline for the LENCU book chapter. Two complementary approaches were evaluated:

1. **Multi-temporal Presto embeddings** (Jul--Dec 2024, 128-band, 10 m) -- captures seasonal dynamics
2. **Single-date Sentinel-2 spectral indices** (Sept 24, 2024, 11 bands, 5 m) -- captures peak weed differentiation

Both approaches were validated through spatial statistics (Bivariate Moran's I, LISA, cross-variograms) and spatially-aware modeling (GWR). The key finding: **the spectral bridge works** -- satellite reflectance explains up to 91% of local weed density variance when spatial non-stationarity is accounted for.

---

## 1. Presto WorldCereal Temporal Embeddings

### 1.1 Extraction Parameters

| Parameter | Value |
|-----------|-------|
| Model | Presto (WorldCereal) |
| Temporal window | July 1 -- December 31, 2024 (6 months) |
| Satellite source | Sentinel-2 via Copernicus Data Space |
| Account | lleon@udec.cl |
| Output dimensions | 128 bands per pixel |
| Spatial resolution | 10 m |
| Pixels analyzed | 424 (inside paddock boundary) |
| Script | `scripts/satellite/extract_presto_lencu_paddock.py` |

### 1.2 PCA Variance Explained

| Component | Variance (%) | Cumulative (%) |
|-----------|-------------|----------------|
| PC1 | 42.7 | 42.7 |
| PC2 | 17.1 | 59.8 |
| PC3 | 13.7 | 73.5 |
| PC4 | 5.6 | 79.1 |
| PC5 | 5.2 | 84.3 |
| PC6--PC10 | 10.6 | **94.9** |

The first 3 components capture 73.5% of the spectral-temporal variance; 10 PCs capture 94.9%.

### 1.3 Pearson Correlations (Non-Spatial)

Pearson *r* between Presto principal components and weed counts per pixel (kriging-interpolated):

| Component | AMBEL | LENCU | POLAV | POLPE |
|-----------|-------|-------|-------|-------|
| PC1 | **0.739** | **0.717** | 0.194 | **0.645** |
| PC2 | -0.367 | -0.279 | -0.023 | -0.312 |
| PC5 | -0.210 | **-0.293** | 0.214 | -0.356 |

PC1 alone explains ~55% of AMBEL variance (r^2 = 0.546) and ~51% of LENCU variance (r^2 = 0.514).

### 1.4 Figures

| Figure | Description | Location |
|--------|-------------|----------|
| `fig_pca_by_species.png` | PCA scatter colored by weed density | `outputs/lencu_presto/` |
| `fig_correlation_heatmap.png` | Full PC x species correlation matrix | `outputs/lencu_presto/` |
| `fig_spatial_maps.png` | Side-by-side PC1 vs weed density maps | `outputs/lencu_presto/` |
| `fig_scatter_matrix.png` | Top PC-weed scatter plots | `outputs/lencu_presto/` |

---

## 2. Single-Date Sentinel-2 Spectral Analysis

### 2.1 Parameters

| Parameter | Value |
|-----------|-------|
| Image date | September 24, 2024 |
| Resolution | 5 m (resampled from 10/20 m) |
| Bands used | 11 (B2--B12, excluding B1/B9/B10) |
| Valid pixels | 1,503 (inside paddock with kriging data) |
| Spectral indices computed | 9 (NDVI, EVI, GNDVI, SWIRd, NBR2, SAVI, NDMI, BSI, NDWI) |
| Script | `scripts/satellite/analyze_sentinel_weed_density.py` |

### 2.2 Top Correlations (Pearson r)

| Index | AMBEL | LENCU | POLPE |
|-------|-------|-------|-------|
| **NDVI** | **0.837** | **0.890** | 0.827 |
| **EVI** | 0.771 | **0.829** | 0.779 |
| **GNDVI** | 0.787 | **0.853** | 0.807 |
| **SWIRd** | 0.768 | **0.830** | 0.820 |
| **NBR2** | 0.707 | **0.777** | 0.789 |

NDVI achieves r = 0.890 for LENCU and r = 0.837 for AMBEL. These are remarkably strong correlations for satellite-to-weed-count relationships.

### 2.3 Interpretation

NDVI r = 0.89 for LENCU means that in September (early spring), pixels with denser *Convolvulus* show significantly higher NDVI. This reflects **mixed weed + crop reflectance** -- above-ground weed biomass directly contributes to the spectral signal. Unlike root parasitic weeds (e.g., *Orobanche*) where the satellite signal is indirect (crop stress), here the satellite measures weed + crop green biomass directly. Denser weed patches = more total green biomass = higher NDVI.

### 2.4 Figures

| Figure | Description | Location |
|--------|-------------|----------|
| `fig_spectral_indices.png` | Spatial maps of 9 computed indices | `outputs/lencu_sentinel_analysis/` |
| `fig_kriging_maps.png` | Kriging density maps for all 4 species | `outputs/lencu_sentinel_analysis/` |
| `fig_correlation_heatmap.png` | Full 19-feature x 4-species matrix | `outputs/lencu_sentinel_analysis/` |
| `fig_scatter_matrix.png` | Top correlation scatter plots | `outputs/lencu_sentinel_analysis/` |
| `fig_pca_by_species.png` | PCA of spectral features | `outputs/lencu_sentinel_analysis/` |
| `fig_umap_by_species.png` | UMAP 2D projection | `outputs/lencu_sentinel_analysis/` |
| `fig_spatial_maps.png` | Spectral vs weed density side-by-side | `outputs/lencu_sentinel_analysis/` |

---

## 3. Spatial Statistics

### 3.1 Bivariate Moran's I (Global)

Tests whether the correlation between satellite embeddings and weed density is **spatially structured** (not just point-wise).

**Weight matrix:** DistanceBand (threshold = 15 m), row-standardized.
**Script:** `scripts/satellite/spatial_correlation_lencu.py`

| Pair | Moran's I | p-value | Significance |
|------|-----------|---------|--------------|
| PC1 x AMBEL | **0.706** | 0.001 | *** |
| PC1 x LENCU | **0.694** | 0.001 | *** |
| PC2 x AMBEL | -0.321 | 0.001 | *** |
| PC2 x LENCU | -0.254 | 0.001 | *** |
| PC5 x LENCU | -0.254 | 0.001 | *** |

Moran's I = 0.706 for PC1 x AMBEL means that locations with high Presto PC1 values tend to be *spatially near* locations with high Ambrosia density. This goes beyond Pearson r -- it confirms the correlation is spatially structured.

### 3.2 Bivariate Local LISA (Local Moran's I)

Identifies **where** the spatial clusters occur at the pixel level.

| Analysis | Total pixels | Significant | HH | LL | HL | LH |
|----------|-------------|-------------|----|----|----|----|
| PC1 x AMBEL | 380 | **258 (67.9%)** | 109 | 126 | 22 | 1 |
| PC1 x LENCU | 380 | **219 (57.6%)** | 92 | 99 | 28 | 0 |

**Cluster interpretation:**

| Cluster | Meaning | Count (AMBEL) | Implication |
|---------|---------|---------------|-------------|
| HH (High-High) | High PC1 + high weed density | 109 | Hot spots -- weed-infested zones visible from satellite |
| LL (Low-Low) | Low PC1 + low weed density | 126 | Cold spots -- clean zones confirmed by satellite |
| HL (High-Low) | High PC1 but low weeds | 22 | Outliers -- spectral signal from non-weed sources |
| LH (Low-High) | Low PC1 but high weeds | 1 | Rare -- weeds not captured spectrally |

68% of the paddock shows significant spatial clusters. Very few outliers (LH = 1), indicating the spectral-weed relationship is spatially consistent.

**Output:** `bivariate_lisa_clusters.gpkg` (QGIS-ready)
**Figure:** `bivariate_lisa_map.png`

### 3.3 Cross-Variograms

Quantifies the **spatial scale** at which spectral-weed co-variation operates.

| Variable pair | Range (sill distance) |
|---------------|----------------------|
| PC1 auto-variogram | ~84 m |
| PC1 x AMBEL cross-variogram | **~90 m** |
| PC1 x LENCU cross-variogram | **~90 m** |

The weed-spectral relationship operates in **~90 meter patches**. Beyond 90 m, knowing a pixel's PC1 value provides no additional information about weed density at a distant location. This defines the spatial grain of the relationship.

**Figure:** `cross_variograms.png`

### 3.4 Geographically Weighted Regression (GWR)

Tests whether the satellite-weed relationship varies across space (**spatial non-stationarity**).

**Script:** `scripts/satellite/gwr_lencu.py`

| Model | OLS R^2 | GWR R^2 | GWR Adj R^2 | AICc | Bandwidth |
|-------|---------|---------|-------------|------|-----------|
| AMBEL ~ PC1+PC2+PC3 | 0.683 | **0.882** | 0.861 | 3930 | 49 nn |
| LENCU ~ PC1+PC2+PC3 | 0.594 | **0.910** | 0.894 | 3783 | 49 nn |

GWR improves R^2 by **+0.20 (AMBEL)** and **+0.32 (LENCU)** over global OLS. This confirms strong spatial non-stationarity -- the relationship between Presto embeddings and weed counts varies significantly across the paddock. The local model explains **88--91%** of weed density variance.

**Outputs:**

| File | Description | Location |
|------|-------------|----------|
| `gwr_results.gpkg` | Local R^2 + local coefficients | `outputs/lencu_presto/spatial_analysis/` |
| `gwr_summary.csv` | OLS vs GWR model statistics | `outputs/lencu_presto/spatial_analysis/` |
| `fig_gwr_local_r2.png` | Maps of where model fits well vs poorly | `outputs/lencu_presto/spatial_analysis/` |
| `fig_gwr_local_coefs.png` | Maps of local PC1 coefficient strength | `outputs/lencu_presto/spatial_analysis/` |
| `fig_gwr_summary_table.png` | OLS vs GWR comparison table | `outputs/lencu_presto/spatial_analysis/` |

---

## 4. Single-Date vs Multi-Temporal Comparison

| Approach | Best AMBEL corr. | Best LENCU corr. | Strengths |
|----------|-------------------|-------------------|-----------|
| Single-date NDVI (Sept 24) | r = 0.837 | r = 0.890 | Simple, high correlation, interpretable |
| Presto PC1 (Jul--Dec, 128-dim) | r = 0.739 | r = 0.717 | Temporal dynamics, captures phenology |
| Presto GWR (PC1--PC3, local) | R^2 = 0.882 | R^2 = 0.910 | Highest explanatory power (local model) |

Single-date NDVI captures a snapshot of peak weed differentiation with remarkably high correlations. Presto embeddings capture temporal dynamics but need spatially-aware modeling (GWR) to unlock their full predictive power. Both approaches confirm the underlying signal.

---

## 5. Weed Detection Ground Truth

### 5.1 Detection Points

| Property | Value |
|----------|-------|
| Source file | `puntos lenteja v2.shp` |
| Features | 1,685 points (individual drone photograms) |
| Detection method | YOLOv11 object detection |
| Flight dates | September 10--17, 2024 |
| Columns | Image_Name, UTM_Eastin, UTM_Northi, Altitude, AMBEL, LENCU, POLAV, POLPE, malez_tota |

### 5.2 Weed Species Statistics

| Species | Scientific name | Mean/photogram | Max/photogram |
|---------|----------------|----------------|---------------|
| AMBEL | *Ambrosia artemisiifolia* | 120 | 774 |
| LENCU | *Convolvulus arvensis* | 160 | 827 |
| POLAV | *Polygonum aviculare* | 2.5 | 31 |
| POLPE | *Polygonum persicaria* | 15.2 | 508 |

### 5.3 Kriging Interpolation (SmartMap)

Continuous density surfaces at 5 m resolution (51 x 32 grid):

| Species | Raster file | Value range |
|---------|-------------|-------------|
| AMBEL | `1_Krig_AMBEL_Grid_Map.tiff` | 1.9 -- 393.4 |
| LENCU | `1_Krig_LENCU_Grid_Map.tiff` | 27.4 -- 402.4 |
| POLAV | `1_Krig_POLAV_Grid_Map.tiff` | 0.4 -- 5.3 |
| POLPE | `1_Krig_POLPE_Grid_Map.tiff` | 0.1 -- 60.3 |

### 5.4 CENIA Multi-Date Inferences

Five YOLOv11 inference runs across September 2024:

| Date | Resolution | Rows | Notes |
|------|-----------|------|-------|
| 6-Sept-2024 | 640 + 1024 | ~1,700 | Flight 1 |
| 9-Sept-2024 | 640 + 1024 | ~1,700 | Flight 2 |
| 10-Sept-2024 | 640 + 1024 | ~1,700 | Flight 3 |
| 16-Sept-2024 | 640 + 1024 | ~1,700 | Flight 4 |
| 17-Sept-2024 | 640 + 1024 | ~1,700 | Flight 5 (used for main analysis) |

Columns: filename, latitude, longitude, altitude, AMBEL, LENCU, POLAV, POLPE, UTM_Easting, UTM_Northing

---

## 6. Ancillary Data

### 6.1 EM38 Electrical Conductivity

| Property | Value |
|----------|-------|
| Instrument | EM38 |
| Points | 1,899 |
| Depths | 75 cm and 150 cm |
| CRS | EPSG:32719 (shapefile), WGS84 (txt) |

Integration with SmartMap produced SVM interpolations for AMBEL and LENCU using CE_75CM as a covariate.

### 6.2 Drone Orthomosaics

| Date | Type | Resolution | Notes |
|------|------|------------|-------|
| 6-Sept-2024 | NDVI + RGB | ~1.3 cm | Pre-weed-peak |
| 17-Sept-2024 | Multispectral (5 bands) + RGB | ~1.3 cm | Peak weed period |
| 17-Sept-2024 | Vegetation indices | ~1.3 cm | GNDVI, LCI, MCARI, NDRE, NDVI |
| 13-Dec-2024 | RGB + multispectral | ~1.3 cm | Near-harvest |

### 6.3 Ground Photos (December 20, 2024)

| Property | Value |
|----------|-------|
| Images | 186 JPG files |
| Camera | vivo V2025 (smartphone) |
| Resolution | 4608 x 3456 px |
| GPS | Yes (within paddock) |
| GeoPackage | `lencu 20 dic SR.gpkg` (181 geotagged points) |

### 6.4 Paddock Boundary

| Format | File |
|--------|------|
| Shapefile | `poligono lentejas v2.shp` (EPSG:32719) |
| KML | `lencu paddock.kml` (WGS84) |
| Geometry | ~260 x 164 m, 1 polygon |

---

## 7. Complete Data Inventory

### 7.1 External Drive: DRONES

Base path: `/media/malezainia1/DRONES/ambel-data/Proyecto_lentejas_v2/`

```
Proyecto_lentejas_v2/
  puntos lenteja v2.shp              # 1,685 weed detection points
  poligono lentejas v2.shp           # Paddock boundary
  smart map outputs/
    0_Dados.csv                      # Raw data (1,685 rows)
    0_Variograma_*.png               # Variograms (AMBEL, LENCU, POLAV, POLPE)
    1_Krig_*_Grid_Map.tiff           # Kriging rasters (4 species)
    1_Krig_*_Grid_Map_SD.csv         # Kriging standard deviation
    1_SVM_POLAV_Grid_Map.tiff        # SVM interpolation
    1_SVM_POLPE_Grid_Map.tiff
    2_ZM_AMBEL_Class.tiff            # Fuzzy zone classification
    2_ZM_AMBEL_MP.csv                # Membership probabilities
    2_ZM_AMBEL_Vars.csv              # Variable means per zone
    2_ZM_POLPE_Class.tiff
    2_ZM_POLPE_MP.csv
    2_ZM_POLPE_Vars.csv
  Smart-Map/
    0_Dados.csv                      # 1,899 pts with CE_75CM
    1_Krig_CE_75CM_Grid_Map.tiff     # CE kriging
    1_SVM_AMBEL_Grid_Map.tiff        # SVM with CE covariate
    1_SVM_LENCU_Grid_Map.tiff
    2_ZM_CE_75CM_Class.tiff          # CE zone classification
    2_ZM_LENCU_Class.*               # LENCU zones (CE-integrated)
  sentinel sept oct 24/
    lencuSR-sept24PixDfields/
      2024-09-24, Boundary1.data.tif # 11-band clipped (59x37 px, 5m)
      2024-09-24, Boundary1.rgb.tif  # RGB composite
      NDVI 24 sept.data.tif          # NDVI raster
    stacksCapas/                     # Band stacks
    Non supervised congedo/          # Unsupervised classification
  ortomosaicos/
    Lencu SR Px 060924/              # 6-Sept: RGB + NDVI (~1.3 cm)
    Proyecto sin nombre 15/
      Lencu_dronMULTI_17sept24.data.tif  # 17-Sept: 5-band multispectral
      Lencu_dronRGB_17sept24.data.tif    # 17-Sept: RGB
    indices_lencu_17sept24/          # GNDVI, LCI, MCARI, NDRE, NDVI
  inferenciasCenia/                  # 5 dates x 2 resolutions
  Lencu_SR_cosecha2025/              # 57 geotagged JPGs (Dec 26, 2024)
  Fotos lentejas 20 dic 24/
    *.jpg (186 images)               # Ground photos (vivo V2025)
    lencu 20 dic SR.gpkg             # 181 geotagged points
    lencu 13 dic 24/
      Ortomosaico.rgb.tif            # 13-Dec RGB orthomosaic
      Ortomosaico.data.tif           # 13-Dec multispectral
  geoda lencu/                       # GeoDa spatial analysis project
  shp_zanahoria/                     # 201,251 carrot detections (different crop)
```

### 7.2 External Drive: E

Base path: `/media/malezainia1/E/Proyecto_lentejas_v2/`

```
Proyecto_lentejas_v2/
  poligono lentejas v2.shp           # Paddock boundary (copy)
  Smart-Map/                         # SmartMap+CE outputs (copy)
  EM38Ronald/
    CeLentejasUTM/
      CE_Lentejas_75cm_UTM.shp       # 1,899 pts, EPSG:32719
      CE_Lentejas_75cm.txt           # 1,900 rows (WGS84)
      CE_Lentejas_150cm.txt          # 1,900 rows (WGS84)
    CeLentejas/                      # Alternative projections
    CeLentejasUTM2/                  # Alternative projections
    SR27.kmz, SRQ10.kmz, etc.       # Survey path KMZ files
  Fotos lentejas 20 dic 24/          # 186 ground photos (copy)
```

### 7.3 Project Outputs

Base path: `/home/malezainia1/dev/INIA_DeepLearning_Ubuntu_mod_lleon/`

```
outputs/lencu_presto/
  presto_jul_dec_2024.tif            # 128-band raw embeddings (GeoTIFF)
  presto_jul_dec_2024.gpkg           # 424 pixels with all 128 bands
  presto_analysis_qgis.gpkg          # PC1-PC5 + weed density (QGIS-ready)
  analysis_results.csv               # PC1-PC10 + species + UTM coords
  aligned_data.csv                   # Embeddings aligned with kriging
  correlation_table.csv              # PC x species correlation matrix
  kriging_ambel.tif                  # CRS-assigned kriging copy
  kriging_lencu.tif
  kriging_polav.tif
  kriging_polpe.tif
  fig_pca_by_species.png
  fig_correlation_heatmap.png
  fig_spatial_maps.png
  fig_scatter_matrix.png
  ADD_LENCU_LAYERS.py                # QGIS loading script
  spatial_analysis/
    bivariate_lisa_clusters.gpkg     # LISA HH/LL/HL/LH clusters
    bivariate_lisa_map.png
    cross_variograms.png
    correlation_summary.csv          # All Moran's I + variogram stats
    gwr_results.gpkg                 # Local R^2 + local coefficients
    gwr_summary.csv                  # OLS vs GWR comparison
    fig_gwr_local_r2.png
    fig_gwr_local_coefs.png
    fig_gwr_summary_table.png

outputs/lencu_sentinel_analysis/
  analysis_data.csv                  # 1,503 pixels x 19 features + 4 species
  correlation_table.csv              # 19-feature x 4-species correlations
  fig_spectral_indices.png
  fig_kriging_maps.png
  fig_correlation_heatmap.png
  fig_scatter_matrix.png
  fig_pca_by_species.png
  fig_umap_by_species.png
  fig_spatial_maps.png
  run.log                            # Full analysis log
```

---

## 8. Analysis Scripts

| Script | Purpose | Key inputs | Key outputs |
|--------|---------|------------|-------------|
| `scripts/satellite/extract_presto_lencu_paddock.py` | Extract Presto 128-band embeddings from Copernicus Data Space, run PCA, compute correlations with kriging weed density | Paddock boundary, Sentinel-2 archive (Jul--Dec 2024) | `presto_jul_dec_2024.tif`, `presto_analysis_qgis.gpkg`, figures |
| `scripts/satellite/analyze_sentinel_weed_density.py` | Single-date Sentinel-2 spectral index analysis: compute 9 indices, correlate with kriging, PCA, UMAP | Clipped 11-band Sentinel-2 (Sept 24), kriging rasters | `analysis_data.csv`, `correlation_table.csv`, figures |
| `scripts/satellite/spatial_correlation_lencu.py` | Bivariate Moran's I, LISA cluster detection, cross-variograms | Presto PCs + weed density from `analysis_results.csv` | `bivariate_lisa_clusters.gpkg`, `correlation_summary.csv`, figures |
| `scripts/satellite/gwr_lencu.py` | Geographically Weighted Regression (OLS vs GWR comparison) | Presto PCs + weed density | `gwr_results.gpkg`, `gwr_summary.csv`, figures |

### Dependencies

- Python: `rasterio`, `geopandas`, `scikit-learn`, `pysal` (esda, mgwr), `scipy`, `matplotlib`, `umap-learn`
- External: Copernicus Data Space account (Presto extraction), Google Earth Engine (optional)

---

## 9. QGIS Integration

### 9.1 QGIS Project

```
research_docs/lencu_book_chapter/proyecto_qgis_maleza1INIA_ambel_embedings.qgz
```

### 9.2 Load All Layers

In QGIS Python Console (`Ctrl+Alt+P`):

```python
exec(open('/home/malezainia1/dev/INIA_DeepLearning_Ubuntu_mod_lleon/outputs/lencu_presto/ADD_LENCU_LAYERS.py').read())
```

### 9.3 Available GeoPackage Layers

| Layer | File | Contents |
|-------|------|----------|
| Presto analysis | `presto_analysis_qgis.gpkg` | PC1--PC5 + AMBEL/LENCU per pixel |
| LISA clusters | `bivariate_lisa_clusters.gpkg` | HH/LL/HL/LH cluster labels |
| GWR results | `gwr_results.gpkg` | Local R^2 + local PC coefficients |

---

## 10. Key Conclusion: The Spectral Bridge Works

### Summary Statistics

| Metric | AMBEL (*Ambrosia*) | LENCU (*Convolvulus*) |
|--------|-------|-------|
| Pearson r (Presto PC1) | 0.739 | 0.717 |
| Pearson r (Single-date NDVI) | 0.837 | 0.890 |
| Bivariate Moran's I (PC1) | 0.706*** | 0.694*** |
| LISA significant clusters | 258/380 (67.9%) | 219/380 (57.6%) |
| Cross-variogram range | ~90 m | ~90 m |
| GWR R^2 (PC1+PC2+PC3) | **0.882** | **0.910** |
| GWR improvement over OLS | +0.20 | +0.32 |

### Mechanism

These are **above-ground broadleaf weeds** whose biomass directly contributes to satellite reflectance. Unlike root parasitic weeds (e.g., *Orobanche*) where the satellite signal is indirect through crop stress, here the satellite measures **weed + crop mixed reflectance**. Denser weed patches produce more green biomass, higher NDVI, and distinctive spectral-temporal signatures captured by Presto embeddings.

### Spatial Structure

The weed-spectral relationship is not uniform across the paddock:
- 68% of pixels show significant bivariate spatial clusters (LISA)
- The co-variation operates in ~90 m patches (cross-variogram range)
- GWR explains 88--91% of local variance vs 59--68% for global OLS
- This spatial non-stationarity likely reflects soil heterogeneity, microclimate, and management history

### Implication for Precision Weed Management

Satellite-derived spectral signatures at 10 m resolution can reliably predict sub-field weed density patterns. This enables:
1. **Early-season weed mapping** from freely available Sentinel-2 imagery
2. **Zone-specific herbicide prescriptions** based on predicted weed pressure
3. **Temporal monitoring** through multi-date Presto embeddings across the growing season
