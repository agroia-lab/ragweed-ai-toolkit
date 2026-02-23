# Evidence 04: Orthomosaic Ragweed Detection Pipeline

## Overview

This document presents the orthomosaic detection pipeline for ragweed (*Ambrosia artemisiifolia*) in a Chilean lentil field. The pipeline tiles a high-resolution Pix4D drone orthomosaic, characterizes the visual domain through embeddings, quantifies domain shift against 9 existing ragweed databases, and selects representative tiles for annotation via active learning. Phases 0 through 3 have been completed; Phases 4 through 8 are pending.

---

## 1. Pipeline Architecture (9 Phases)

```
ORTHOMOSAIC ──> [0] TILE ──> [1] EMBED ──> [2] DOMAIN SHIFT ──> [3] SELECT & LABEL
    2.4 GB       2,151        ResNet50       MMD analysis         80 tiles
    73k x 41k    valid tiles  2048-dim       vs 9 databases       K-means + stratified
    EPSG:32719   1024x1024    UMAP/tSNE/PCA  overall=0.4422       for Roboflow/CVAT

                              ──> [4] TRANSFER LEARN ──> [5] SAHI INFERENCE ──> [6] ACTIVE LEARNING
                                  Fine-tune from           All 2,151 tiles      Uncertainty mining
                                  Run 3 (mAP50=0.886)     on dual RTX 4090     label 30 more, retrain

                                         ──> [7] FINAL MAP (QGIS) ──> [8] DOCUMENTATION
                                              GeoPackage overlay        Methods + Results
                                              LISA cluster analysis     Book chapter annex
```

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | Completed | Tile orthomosaic into 1024x1024 px JPEG tiles |
| 1 | Completed | Extract ResNet50 embeddings + dimensionality reduction |
| 2 | Completed | Domain shift analysis (MMD) against 9 ragweed databases |
| 3 | Completed | K-means stratified tile selection (80 tiles) |
| 4 | Pending | Transfer learning from best AMBEL model |
| 5 | Pending | SAHI inference on all 2,151 tiles |
| 6 | Pending | Active learning loop (uncertainty mining) |
| 7 | Pending | Final georeferenced detection map in QGIS |
| 8 | Pending | Full methodology and results documentation |

---

## 2. Orthomosaic Source

| Property | Value |
|----------|-------|
| File | `/media/malezainia2/E/Proyecto_lentejas_v2/Fotos lentejas 20 dic 24/lencu 13 dic 24/Ortomosaico.rgb.tif` |
| Dimensions | 73,137 x 40,738 pixels |
| File size | 2.4 GB |
| Ground sampling distance | ~3.9 mm/pixel |
| CRS | WGS 84 / UTM zone 19S (EPSG:32719) |
| Ground coverage | ~283 m x 158 m (~4.5 ha) |
| Software | Pix4Dfields 2.8.0 |
| Bands | 4 (RGB + Alpha) |
| Valid pixel coverage | ~80% (remainder is nodata border) |
| Acquisition date | 13 December 2024 |
| Field | Lentil field ("lencu"), Chile |

The orthomosaic resolves individual ragweed plants at 3.9 mm/pixel. The 20% nodata border is the typical triangular fringe produced by Pix4D's stitching algorithm around the edges of the mapped area.

---

## 3. Phase 0 -- Tiling

### Method

The orthomosaic was split into a regular grid of 1024 x 1024 pixel tiles (~4 m x 4 m ground coverage each). The script opens the GeoTIFF in windowed mode (reads small chunks, never the full 2.4 GB into memory), steps through grid positions, checks the alpha band for each window, and discards tiles where more than 10% of pixels are nodata (border regions). Valid tiles are saved as RGB JPEG files with grid-position names (e.g., `tile_r0012_c0045.jpg`). A CSV manifest records the UTM coordinates for each tile.

### Results

| Metric | Value |
|--------|-------|
| Grid dimensions | 72 columns x 40 rows |
| Total grid positions | 2,880 |
| Valid tiles (>90% pixel coverage) | **2,151** (74.7%) |
| Skipped tiles (nodata border) | 729 (25.3%) |
| Tile size | 1024 x 1024 px |
| Ground tile coverage | ~4 m x 4 m |
| Minimum valid fraction | 0.9 |
| Processing time | 75.3 seconds |

### Grid Zone Distribution

| Zone | Tiles |
|------|-------|
| center | 312 |
| mid-right | 281 |
| mid-left | 279 |
| bot-center | 249 |
| top-left | 238 |
| bot-right | 237 |
| top-center | 231 |
| bot-left | 181 |
| top-right | 143 |

### Output Files

| Output | Path |
|--------|------|
| Tiles directory | `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_tiles_1024/` |
| Tile manifest CSV | `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_tiles_1024/tile_manifest.csv` |
| Tiling summary JSON | `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_tiles_1024/tiling_summary.json` |

### Reproduction Command

```bash
conda activate virtual_environment_yolo

python scripts/cli/tile_orthomosaic.py \
    --raster "/media/malezainia2/E/Proyecto_lentejas_v2/Fotos lentejas 20 dic 24/lencu 13 dic 24/Ortomosaico.rgb.tif" \
    --output "/media/malezainia2/E/Proyecto_lentejas_v2/ortho_tiles_1024" \
    --tile-size 1024 \
    --min-valid 0.9
```

---

## 4. Phase 1 -- Embedding Extraction

### Method

ResNet50 (pretrained on ImageNet) was used to extract a 2048-dimensional feature vector ("embedding") from each of the 2,151 tiles. These embeddings represent a visual fingerprint of each tile in a high-dimensional feature space. Three dimensionality reduction methods were applied for visualization:

- **UMAP** (Uniform Manifold Approximation and Projection) -- preserves local and global structure
- **t-SNE** (t-distributed Stochastic Neighbor Embedding) -- preserves local neighborhood relationships
- **PCA** (Principal Component Analysis) -- linear projection capturing maximum variance

### Results

| Metric | Value |
|--------|-------|
| Embedding dimension | 2,048 (ResNet50 penultimate layer) |
| Number of tiles embedded | 2,151 |
| PCA cumulative variance at 50 dims | **94.75%** |
| Embedding extraction time | ~5 seconds (batch=64, cuda:1, RTX 4090) |
| UMAP + t-SNE + PCA reduction time | ~15 seconds |

The 94.75% variance explained at 50 dimensions indicates that the orthomosaic tile feature space is well-structured and amenable to clustering. UMAP visualizations show distinct groupings corresponding to tile content (bare soil, dense crop, weedy patches, field edges).

### Output Files

| Output | Path |
|--------|------|
| Raw embeddings (2048-dim) | `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_embeddings/embeddings.npz` |
| Reduced coordinates CSV | `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_embeddings/embeddings_reduced.csv` |
| UMAP by grid zone (interactive) | `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_embeddings/umap_by_grid_zone.html` |
| UMAP by row | `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_embeddings/umap_by_row.html` |
| UMAP by valid fraction | `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_embeddings/umap_by_valid_frac.html` |
| PCA by grid zone (interactive) | `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_embeddings/pca_by_grid_zone.html` |
| t-SNE by grid zone (interactive) | `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_embeddings/tsne_by_grid_zone.html` |

### Reproduction Command

```bash
conda activate virtual_environment_yolo

python scripts/cli/embed_ortho_tiles.py \
    --tiles "/media/malezainia2/E/Proyecto_lentejas_v2/ortho_tiles_1024" \
    --manifest "/media/malezainia2/E/Proyecto_lentejas_v2/ortho_tiles_1024/tile_manifest.csv" \
    --output "/media/malezainia2/E/Proyecto_lentejas_v2/ortho_embeddings"
```

---

## 5. Phase 2 -- Domain Shift Analysis

### Method

Maximum Mean Discrepancy (MMD) was computed between the orthomosaic tile embeddings and the embeddings from 9 existing ragweed image databases (5,338 images total). MMD is a kernel-based statistical test that measures the distance between two distributions in a Reproducing Kernel Hilbert Space. A Gaussian kernel was used with gamma estimated from the median pairwise distance (global gamma = 0.332787). Statistical significance was assessed via 1,000 permutations.

### Deployment Decision Framework

| MMD Range | Risk Level | Action |
|-----------|------------|--------|
| < 0.15 | Minimal | Deploy model directly |
| 0.15 -- 0.30 | Moderate | Deploy with data augmentation |
| **0.30 -- 0.45** | **Significant** | **Active learning required** |
| > 0.45 | Severe | Collect substantial new labeled data |

### Overall Result

| Metric | Value |
|--------|-------|
| **Overall MMD** | **0.4422** |
| **Risk level** | **SIGNIFICANT** |
| **Recommendation** | **Active learning required** |
| Closest database | CL_StaRosa (MMD = 0.4339) |
| Farthest database | WeedCube (MMD = 0.7232) |
| Global gamma | 0.332787 |
| Permutation tests | 1,000 |
| Computation time | 1,085.7 s (~18 min) |

The overall MMD of 0.4422 falls at the boundary between the "active learning required" zone (0.30--0.45) and the "collect substantial new data" zone (>0.45), confirming that transfer learning with active learning iterations is the correct approach rather than direct deployment.

### Per-Database MMD Values

| Database | N images | MMD | p-value | Significant |
|----------|----------|-----|---------|-------------|
| CL_StaRosa | 75 | 0.4339 | 0.001 | Yes |
| ND_Aerial | 277 | 0.4708 | 0.001 | Yes |
| WeedCrop | 495 | 0.4967 | 0.001 | Yes |
| ND_Indiv | 878 | 0.5147 | 0.001 | Yes |
| CL_Alberto | 659 | 0.5308 | 0.001 | Yes |
| MI_3Season | 719 | 0.5465 | 0.001 | Yes |
| CL_Seba | 2,065 | 0.6032 | 0.001 | Yes |
| Purdue | 150 | 0.7173 | 0.001 | Yes |
| WeedCube | 20 | 0.7232 | 0.001 | Yes |

All 9 databases show statistically significant domain shift (p = 0.001 for all). The orthomosaic tiles are closest to CL_StaRosa (Chilean ground-level photos from Santa Rosa) and farthest from WeedCube (a curated international dataset).

### Tile Distance Statistics

| Statistic | Distance to training centroid |
|-----------|-------------------------------|
| Minimum | 0.7791 |
| Median | 0.9378 |
| Mean | 0.9339 |
| Maximum | 1.0260 |
| Std. deviation | 0.0307 |

The tight standard deviation (0.0307) indicates that tiles are uniformly distant from the training distribution. This is a consistent, field-wide domain shift rather than a partial one affecting only some regions.

### Cross-Domain Validation

A separate cross-domain evaluation (`malezainia2/ambel_crossdomain/`) tested the best AMBEL model (Run 3, mAP50 = 0.886) against 4 international databases with labeled test sets. Result: **mAP50 = 0.108, zero detections on all 4 databases**. This independently confirms that the model does not generalize to other domains without fine-tuning.

### Output Files

| Output | Path |
|--------|------|
| Domain shift report (JSON) | `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_domain_shift/domain_shift_report.json` |
| MMD bar chart (PNG) | `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_domain_shift/mmd_comparison.png` |
| Joint UMAP (interactive HTML) | `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_domain_shift/joint_umap.html` |
| Joint UMAP (static PNG) | `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_domain_shift/joint_umap.png` |
| Joint UMAP coordinates | `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_domain_shift/joint_umap_coords.csv` |
| Per-tile distances CSV | `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_domain_shift/tile_distances.csv` |

### Reproduction Command

```bash
conda activate virtual_environment_yolo

python scripts/cli/compute_ortho_domain_shift.py \
    --tile-embeddings "/media/malezainia2/E/Proyecto_lentejas_v2/ortho_embeddings/embeddings.npz" \
    --reference-embeddings "/media/malezainia2/E/rageweed_international_databases/ragweed_embeddings/embeddings.npz" \
    --output "/media/malezainia2/E/Proyecto_lentejas_v2/ortho_domain_shift"
```

---

## 6. Phase 3 -- Tile Selection for Labeling

### Method

Tiles were selected using a two-stage strategy to maximize diversity:

1. **K-means clustering** (k = 10) on UMAP 2D coordinates to identify visually distinct tile groups
2. **Stratified sampling** within each cluster based on distance-to-training-centroid, using three strata:
   - Near-training (25 tiles): tiles closest to the existing training distribution
   - Mid-distance (40 tiles): tiles at intermediate distances
   - Far-from-training (15 tiles): tiles most different from training data

### Selection Results

| Metric | Value |
|--------|-------|
| Total tiles selected | **80** |
| Selection method | K-means (k=10) + quantile-based stratification |
| Near-training allocation | 25 tiles |
| Mid-distance allocation | 40 tiles |
| Far-from-training allocation | 15 tiles |
| Grid zone coverage | All 9 zones represented |
| Distance range | 0.7791 -- 0.9833 |

### Cluster Allocation

| Cluster | Total tiles | Selected | Distance range |
|---------|-------------|----------|----------------|
| 0 | 243 | 9 | 0.861 -- 0.975 |
| 1 | 241 | 9 | 0.900 -- 0.973 |
| 2 | 142 | 5 | 0.779 -- 0.926 |
| 3 | 177 | 7 | 0.793 -- 0.974 |
| 4 | 279 | 10 | 0.901 -- 0.983 |
| 5 | 156 | 6 | 0.864 -- 0.958 |
| 6 | 241 | 9 | 0.822 -- 0.939 |
| 7 | 184 | 7 | 0.904 -- 0.960 |
| 8 | 185 | 7 | 0.867 -- 0.954 |
| 9 | 303 | 11 | 0.859 -- 0.948 |

### Labeling Strategy (Dual-Track)

**Track A -- Roboflow (initial labeling):**
- Browser-based annotation platform, no local installation required
- Model-assisted pre-annotation using best.pt (Run 3, mAP50 = 0.886)
- Single class: AMBEL
- Estimated labeling time: 6--8 hours (~5 min per tile)
- Export format: YOLO

**Track B -- CVAT (active learning iterations):**
- Self-hosted Docker instance for faster iterative labeling
- REST API for scripted tile upload/export
- Set up on `feature/cvat-integration` branch
- To be used from Phase 6 onward

### Dataset Configuration

- **YAML config:** `configs/yaml_config/data_ambel-ortho-v1.yaml`
- **Single class:** AMBEL
- **Planned split:** 56 train / 16 valid / 8 test (70/20/10)

```yaml
path: /media/malezainia2/E/ProcessingData/ambel_ortho_v1
train: train/images
val: valid/images
test: test/images
nc: 1
names: ['AMBEL']
```

### Output Files

| Output | Path |
|--------|------|
| Selected tiles CSV | `/media/malezainia2/E/Proyecto_lentejas_v2/selected_tiles.csv` |
| Tiles copied for labeling | `/media/malezainia2/E/Proyecto_lentejas_v2/tiles_for_labeling/` (80 images) |
| Dataset directory | `/media/malezainia2/E/ProcessingData/ambel_ortho_v1/{train,valid,test}/{images,labels}/` |
| Dataset YAML | `configs/yaml_config/data_ambel-ortho-v1.yaml` |
| Roboflow labeling guide | `malezainia2/ambel_ortho/labeling_instructions.md` |
| CVAT setup guide | `malezainia2/ambel_ortho/cvat_setup_instructions.md` |

---

## 7. Pending Phases (4--8)

### Phase 4 -- Transfer Learning

Fine-tune the best existing AMBEL model (Run 3, mAP50 = 0.886 on ground-level photos) on the newly labeled orthomosaic tiles. The model already recognizes ragweed from ground perspective; transfer learning adapts it to the drone orthomosaic domain. Planned parameters: 30 epochs, dual RTX 4090, image size 1024 px, batch 32.

```bash
python scripts/cli/train_yolo_cli.py \
    --model training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/weights/best.pt \
    --data configs/yaml_config/data_ambel-ortho-v1.yaml \
    --epochs 30 --batch 32 --imgsz 1024 --device "0,1" \
    --project training_results/malezainia2/ambel_ortho \
    --name ortho_transfer_v1
```

**Target:** mAP50 > 0.70 (first iteration).

### Phase 5 -- SAHI Inference

Run SAHI (Slicing Aided Hyper Inference) with the fine-tuned model on all 2,151 tiles. Convert pixel-level detections to georeferenced UTM coordinates using the tile manifest, producing a GeoPackage for QGIS.

**Estimated runtime:** ~20 minutes on a single RTX 4090, or ~10 minutes using dual-GPU parallel inference.

### Phase 6 -- Active Learning Loop

Mine the 30 most uncertain tiles from SAHI output using 4 criteria (low confidence 40%, distribution shift 25%, borderline density 20%, confidence entropy 15%). Label those tiles, add to the training set, retrain. Repeat until convergence: mAP50 > 0.80, total count change < 5%, rising mean confidence, falling uncertainty scores. Typically 2--3 rounds, each adding ~30 tiles.

### Phase 7 -- Final Map

Generate the final georeferenced GeoPackage (`ambel_ortho_detections_final.gpkg`) from the converged model. Load in QGIS with the orthomosaic as the raster base layer. Apply graduated symbology by confidence. Optionally run LISA (Local Moran's I) spatial cluster analysis to identify ragweed hotspots.

### Phase 8 -- Documentation

Write full methods and results documentation for the book chapter annex. Structure: `malezainia2/ambel_ortho/methods/01_orthomosaic_detection_methods.md` and `malezainia2/ambel_ortho/resultados/01_orthomosaic_detection_results.md`.

---

## 8. Existing Assets and Pre-Trained Models

### Trained AMBEL Models

| Run | Dataset | mAP50 | Model Path |
|-----|---------|-------|------------|
| **Run 3 (BEST)** | Lentejas 640px | **0.886** | `training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/weights/best.pt` |
| Run 2 | Lentejas 1024px | 0.871 | `run2_lentejas1024_b8_1gpu/weights/best.pt` |
| Run 1 | Seba 640px | 0.810 | `run1_seba640_b32_1gpu/weights/best.pt` |
| Run 4 | Combined transfer | 0.839 | `combined_v1_transfer_20260218_171619/weights/best.pt` |

### Ragweed Embedding Reference Data

Pre-computed ResNet50 embeddings for 5,338 images across 9 international ragweed databases, stored at:
`/media/malezainia2/E/rageweed_international_databases/ragweed_embeddings/embeddings.npz`

---

## 9. Scripts and Reproduction

### Pipeline Scripts (on master branch)

| Script | Purpose | Phase |
|--------|---------|-------|
| `scripts/cli/tile_orthomosaic.py` | Split GeoTIFF into tiles with rasterio windowed reading | 0 |
| `scripts/cli/embed_ortho_tiles.py` | ResNet50 embedding extraction + UMAP/t-SNE/PCA | 1 |
| `scripts/cli/compute_ortho_domain_shift.py` | MMD analysis + deployment gate + joint UMAP | 2 |
| `scripts/cli/train_yolo_cli.py` | YOLO training (transfer learning) | 4 |
| `scripts/cli/sahi_inference_cli.py` | SAHI sliced inference | 5 |
| `scripts/cli/mine_uncertain_samples.py` | Uncertainty mining for active learning | 6 |
| `scripts/cli/detections_to_geopackage.py` | Convert pixel detections to GeoPackage | 7 |

### CVAT Integration Scripts (on `feature/cvat-integration` branch)

| Script | Purpose |
|--------|---------|
| `scripts/cli/cvat_upload_tiles.py` | Upload tiles + pre-annotations to CVAT REST API |
| `scripts/cli/cvat_export_yolo.py` | Download YOLO-format labels from CVAT |
| `docker/cvat/docker-compose.override.yml` | CVAT Docker configuration |

### Configuration

| File | Purpose |
|------|---------|
| `configs/yaml_config/data_ambel-ortho-v1.yaml` | Dataset YAML for training (1 class: AMBEL) |

---

## 10. Execution Environment

| Property | Value |
|----------|-------|
| Machine | malezainia2 |
| GPUs | 2x NVIDIA RTX 4090 (cuda:0, cuda:1) |
| Conda environment | `virtual_environment_yolo` |
| Python | `/home/malezainia2/anaconda3/envs/virtual_environment_yolo/bin/python` |
| Key dependencies | rasterio 1.4.3, torch 2.4.1, umap-learn 0.5.11, scikit-learn 1.6.0, plotly 6.5.2, ultralytics >= 8.4.0 |

---

## 11. Complete Data Paths

### Phase 0 -- Tiling

```
/media/malezainia2/E/Proyecto_lentejas_v2/ortho_tiles_1024/
    tile_r0000_c0000.jpg ... tile_r0039_c0071.jpg    (2,151 valid tiles)
    tile_manifest.csv                                  (georeferencing table)
    tiling_summary.json                                (processing statistics)
```

### Phase 1 -- Embeddings

```
/media/malezainia2/E/Proyecto_lentejas_v2/ortho_embeddings/
    embeddings.npz              (2,151 x 2,048 raw features)
    embeddings_reduced.csv      (UMAP + t-SNE + PCA coordinates)
    umap_by_grid_zone.html      (interactive visualization)
    umap_by_row.html
    umap_by_valid_frac.html
    pca_by_grid_zone.html
    tsne_by_grid_zone.html
```

### Phase 2 -- Domain Shift

```
/media/malezainia2/E/Proyecto_lentejas_v2/ortho_domain_shift/
    domain_shift_report.json    (full MMD results)
    mmd_comparison.png          (bar chart of per-database MMD)
    joint_umap.html             (interactive: tiles + 9 databases)
    joint_umap.png              (static version)
    joint_umap_coords.csv       (coordinates for all points)
    tile_distances.csv          (per-tile distance to training centroid)
```

### Phase 3 -- Selected Tiles and Dataset

```
/media/malezainia2/E/Proyecto_lentejas_v2/
    selected_tiles.csv          (80 selected tile metadata)
    tiles_for_labeling/         (80 JPEG tiles copied for upload)

/media/malezainia2/E/ProcessingData/ambel_ortho_v1/
    train/images/               (56 tiles, 70%)
    train/labels/               (YOLO .txt annotations)
    valid/images/               (16 tiles, 20%)
    valid/labels/
    test/images/                (8 tiles, 10%)
    test/labels/
```

### Planning and Session Documents

```
malezainia2/ambel_ortho/
    AMBEL_ORTHO_PLAN.md                     (full 9-phase plan)
    labeling_instructions.md                (Roboflow guide, Spanish)
    cvat_setup_instructions.md              (CVAT Docker guide, Spanish)
    sessions/
        session_1_checkpoint.md             (Phase 0-1 completion)
        session_2_checkpoint.md             (Phase 2-3 completion)
```

---

## 12. Session Execution Log

| Session | Date | Phases | Team Structure | Duration |
|---------|------|--------|----------------|----------|
| 1 | 2026-02-22 | 0, 1 | Leader only (sequential) | ~2 min (tiling 75s + embeddings 20s) |
| 2 | 2026-02-22 | 2, 3 | Leader + 3 parallel agents | ~20 min (domain shift 18 min + parallel prep 3 min) |
| 3 | Pending | 4 | Leader + trainer (background GPU) + inference-prep | -- |
| 4 | Pending | 5, 6 | Leader + dual-GPU SAHI + merge + uncertainty miner | -- |
| 5 | Pending | 7, 8 | Leader + 3 parallel doc writers | -- |

### Session 2 Agent Team

| Agent | Task | Duration | Status |
|-------|------|----------|--------|
| domain-shift-analyzer | Phase 2: MMD computation (1000 permutations x 9 databases) | ~18 min | Completed |
| tile-selector | K-means clustering + stratified sampling of 80 tiles | ~3 min | Completed |
| config-creator | Dataset YAML + directory structure + Roboflow guide | ~2 min | Completed |
| cvat-setup | Docker config + bridge scripts (on worktree branch) | ~5 min | Completed |

---

## 13. Key Findings

1. **Consistent domain shift:** The overall MMD of 0.4422 confirms that the orthomosaic domain differs significantly from all 9 existing ragweed databases. Direct model deployment would produce unreliable results.

2. **Uniform shift across field:** The tight standard deviation of tile distances (0.0307) shows this is a field-wide phenomenon, not localized to certain regions. All tiles are roughly equally different from the training data.

3. **Chilean data is closest:** CL_StaRosa (Chilean ground photos from the same region) has the lowest MMD (0.4339), while international curated datasets (WeedCube, Purdue) are farthest. This confirms geographic and acquisition-method proximity matters.

4. **Active learning validated:** The cross-domain evaluation (mAP50 = 0.108, zero detections on international test sets) independently confirms that fine-tuning is essential. The embedding-based domain shift analysis and the detection-based cross-domain evaluation reach the same conclusion through different methods.

5. **Efficient labeling:** Only 80 tiles (3.7% of 2,151) need manual annotation in the first round, with the active learning loop expected to add another 60--90 tiles over 2--3 iterations, for a total of ~140--170 tiles (6--8% of the field).
