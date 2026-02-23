# Evidence 02: Ragweed Embedding Space Analysis

**Book Chapter Section:** Domain Shift and Generalization Analysis
**Analysis Date:** 2026-02-21
**Pipeline Version:** 3.0

---

## 1. Overview

This evidence package documents a cross-dataset visual pattern exploration for ragweed (*Ambrosia artemisiifolia*) using deep learning embeddings. The analysis projects 5,338 images from 9 databases spanning 2 continents into a shared feature space using ResNet50, then quantifies inter-database distances with Maximum Mean Discrepancy (MMD) and Proxy A-Distance (PAD) to establish a distribution-aware deployment framework.

**Key finding:** Database-specific clustering dominates the embedding space. Images cluster by acquisition equipment and conditions, not by ragweed morphology. This has direct implications for cross-domain model deployment.

---

## 2. Method

### 2.1 Feature Extraction

| Parameter | Value |
|-----------|-------|
| **Model** | ResNet50 (torchvision) |
| **Weights** | ImageNet V2 (`IMAGENET1K_V2`) |
| **Architecture modification** | Final classification layer removed; output from global average pooling |
| **Embedding dimensionality** | 2,048 per image |
| **Preprocessing** | Resize to 224x224, normalize with ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225] |
| **Inference** | Batch size 16, 4 DataLoader workers, `torch.no_grad()` mode |
| **Hardware** | NVIDIA RTX 4090 (24 GB VRAM), `cuda:1` |
| **Augmentation** | None (intentionally preserving acquisition-specific characteristics) |

### 2.2 Dimensionality Reduction

Four projection methods were applied to the 5,338 x 2,048 embedding matrix:

| Method | Components | Pre-processing | Key Parameters |
|--------|------------|----------------|----------------|
| **UMAP 2D** | 2 | Pre-PCA: 2048 -> 50 dims | n_neighbors=15, min_dist=0.1, metric=cosine |
| **UMAP 3D** | 3 | Pre-PCA: 2048 -> 50 dims | n_neighbors=15, min_dist=0.1, metric=cosine |
| **t-SNE 2D** | 2 | Pre-PCA: 2048 -> 50 dims | perplexity=30, max_iter=1000 |
| **PCA 2D** | 2 | None (direct on 2048 dims) | -- |

All methods used `random_state=42` for reproducibility. The pre-PCA step (2048 -> 50 dimensions) accelerates UMAP and t-SNE while preserving the dominant variance structure.

### 2.3 Domain Distance Metrics

Two complementary pairwise distance metrics were computed between all 9 databases:

**Maximum Mean Discrepancy (MMD):** A kernel-based measure of the distance between two probability distributions in a reproducing kernel Hilbert space (Gretton et al., 2012). MMD = 0 indicates identical distributions; larger values indicate greater distributional divergence. Gaussian RBF kernel with bandwidth selected by the median heuristic (global gamma = 0.343) on L2-normalized embeddings.

**Proxy A-Distance (PAD):** Measures how easily a linear classifier can distinguish between two databases (Ben-David et al., 2007, 2010). PAD = 2 * (1 - 2 * error), where error is the cross-validated misclassification rate of a LinearSVC. PAD near 0 means indistinguishable; PAD near 2 means completely separable. Computed on PCA-reduced embeddings (100 dimensions, 58.7% variance explained).

### 2.4 Visualization Design

Each visualization is a self-contained HTML file (~70 MB) built with:

- **Plotly** scatter plots (2D or 3D) with `plotly_dark` template
- **Custom CSS** implementing a FiftyOne-inspired dark theme (background: `#262626`, accent: `#FF6900`, font: Palanquin)
- **450px side panel** ("Sample Inspector") with image preview, click-to-enlarge fullscreen modal, filename bar, metadata card (source, database, origin badge, labels badge), and stats bar
- **Thumbnail system:** 200x200 JPEG thumbnails at quality 70, base64-encoded and cached in `thumbnails_cache.json` (68 MB), embedded directly into HTML for instant hover/click preview

---

## 3. Databases (9 Sources, 5,338 Images)

| # | Source ID | Database Name | Images | Origin | Equipment | Species | Labelled | Notes |
|---|-----------|---------------|--------|--------|-----------|---------|----------|-------|
| 1 | 01_NorthDakota_Aerial | ImageWeeds NorthDakota -- Aerial | 277 | North Dakota, USA | DJI Phantom 4 Pro (drone) | *A. artemisiifolia* | Yes | Whitelist-filtered from multi-class; XML/JSON/TXT labels |
| 2 | 01_NorthDakota_Individual | ImageWeeds NorthDakota -- Individual | 878 | North Dakota, USA | Canon EOS 90D (ground) | *A. artemisiifolia* | Yes | Dedicated ragweed folder |
| 3 | 03_Michigan_3Season | 3SeasonWeedDet10 Michigan | 719 | Michigan, USA | iPhone SE, iPhone 11 Pro, Nikon D3300, Canon EOS 4000D | *A. artemisiifolia* | Yes | 3 seasons (2021: 512, 2022: 93, 2023: 114); whitelist-filtered |
| 4 | 05_Purdue_4Weed | 4Weed Purdue | 150 | Indiana, USA | Webcam on UTV, 1080x720 | *A. trifida* (giant ragweed) | Yes | Dedicated ragweed folder |
| 5 | 06_USDA_WeedCube | WeedCube USDA | 20 | Maryland, USA | Specim FX10 hyperspectral (pseudo-RGB renders) | *A. artemisiifolia* | Yes | 224 spectral bands (400--1000 nm); PNG pseudo-RGB |
| 6 | 07_WeedCrop_PrecAg | WeedCrop Precision Ag | 495 | Multi-site, USA | Field camera, multi-crop | *A. artemisiifolia* | Yes | YOLO class 8 = ragweed; filtered across 8 crop subfolders |
| 7 | CL_Seba_Labelled | Chilean Ambrosia (Seba) | 2,065 | Santiago, Chile | Field camera + Roboflow augmentation | *A. artemisiifolia* | Yes | Roboflow YOLO format; train/valid/test splits |
| 8 | CL_Alberto_Labelled | Chilean Ambrosia (Alberto) | 659 | Santiago, Chile | Field camera + Roboflow augmentation | *A. artemisiifolia* | Yes | Roboflow YOLO format; train/valid/test splits |
| 9 | CL_StaRosa_Unlabelled | Chilean Ambrosia (Sta Rosa) | 75 | Santa Rosa, Chile | Field camera (original, ~7 MB each) | *A. artemisiifolia* | No | Raw field photos, Feb 2023 |
| | **TOTAL** | | **5,338** | | | | | **2,539 international + 2,799 Chilean** |

### 3.1 Geographic Distribution

- **International databases** (6 sources, 2,539 images): North Dakota (aerial + ground), Michigan (3 seasons across 2021--2023), Indiana/Purdue (UTV webcam), Maryland/USDA (hyperspectral), multi-site precision agriculture
- **Chilean databases** (3 sources, 2,799 images): Santiago metropolitan region and central Chile -- Seba (Roboflow-augmented), Alberto (Roboflow-augmented), Santa Rosa (raw originals, unlabelled)

### 3.2 Multi-Class Filtering

Three databases contained multiple weed species. Only images containing ragweed were included, selected via whitelist files generated by parsing annotation files:

| Whitelist File | Source Database | Filenames Selected |
|----------------|----------------|-------------------|
| `whitelist_01a_aerial.txt` | DB01a NorthDakota Aerial | 277 |
| `whitelist_03_michigan.txt` | DB03 Michigan 3Season | 719 |
| `whitelist_07_weedcrop.txt` | DB07 WeedCrop PrecAg | 495 |

---

## 4. Results

### 4.1 Embedding Space Clustering (UMAP 2D by Database)

The primary visualization reveals **strong database-specific clustering**:

- **CL_Seba_Labelled** (2,065 images): Largest, densest cluster. Roboflow augmentation produces highly uniform visual features, causing tight grouping.
- **01_NorthDakota_Individual** (878 images): Splits into several sub-clusters, likely corresponding to different ragweed growth stages or photo acquisition sessions.
- **03_Michigan_3Season** (719 images): Distinct cluster reflecting consistent field photography with consumer cameras across 3 seasons.
- **CL_Alberto_Labelled** (659 images): Clusters separately from CL_Seba despite both being Chilean Roboflow-augmented -- indicates different source imagery and/or augmentation pipelines.
- **07_WeedCrop_PrecAg** (495 images): Distributed cluster spanning images from 8 different crop backgrounds.
- **01_NorthDakota_Aerial** (277 images): Isolated cluster -- aerial (drone) perspective is visually distinct from all ground-level photography.
- **05_Purdue_4Weed** (150 images): Tight, isolated cluster reflecting uniform UTV webcam acquisition.
- **CL_StaRosa_Unlabelled** (75 images): Small cluster near the Chilean labelled sources.
- **06_USDA_WeedCube** (20 images): Tiny, isolated cluster -- pseudo-RGB renders from hyperspectral cubes have unique visual characteristics.

### 4.2 Geographic Separation (UMAP 2D by Origin)

Clear separation between Chilean and international images:

- **Chilean images** (2,799): Cluster in one region of embedding space
- **International images** (2,539): Spread across a different region
- **Minimal overlap** between groups, confirming that field conditions, camera hardware, background vegetation, and lighting produce visually distinct imagery by geography

### 4.3 Pairwise MMD Distance Matrix

Full 9x9 MMD distance matrix (Gaussian RBF kernel, median heuristic bandwidth):

| | CL_Seba | ND_Indiv | MI_3Season | CL_Alberto | WeedCrop | ND_Aerial | Purdue | CL_StaRosa | WeedCube |
|---|---------|----------|------------|------------|----------|-----------|--------|------------|----------|
| **CL_Seba** | 0.000 | 0.270 | 0.358 | 0.355 | 0.372 | 0.427 | 0.454 | 0.347 | 0.482 |
| **ND_Indiv** | 0.270 | 0.000 | 0.202 | 0.282 | 0.161 | 0.223 | 0.331 | 0.267 | 0.399 |
| **MI_3Season** | 0.358 | 0.202 | 0.000 | 0.269 | 0.152 | 0.210 | 0.266 | 0.264 | 0.527 |
| **CL_Alberto** | 0.355 | 0.282 | 0.269 | 0.000 | 0.307 | 0.325 | 0.435 | 0.184 | 0.469 |
| **WeedCrop** | 0.372 | 0.161 | 0.152 | 0.307 | 0.000 | 0.060 | 0.361 | 0.235 | 0.525 |
| **ND_Aerial** | 0.427 | 0.223 | 0.210 | 0.325 | 0.060 | 0.000 | 0.457 | 0.206 | 0.561 |
| **Purdue** | 0.454 | 0.331 | 0.266 | 0.435 | 0.361 | 0.457 | 0.000 | 0.482 | 0.639 |
| **CL_StaRosa** | 0.347 | 0.267 | 0.264 | 0.184 | 0.235 | 0.206 | 0.482 | 0.000 | 0.544 |
| **WeedCube** | 0.482 | 0.399 | 0.527 | 0.469 | 0.525 | 0.561 | 0.639 | 0.544 | 0.000 |

### 4.4 Most Similar and Most Distant Pairs

**Five most similar pairs (lowest MMD):**

| Rank | Pair | MMD | Interpretation |
|------|------|-----|----------------|
| 1 | WeedCrop -- ND_Aerial | 0.060 | Multi-background field imagery overlaps |
| 2 | MI_3Season -- WeedCrop | 0.152 | Similar field photography protocols |
| 3 | ND_Indiv -- WeedCrop | 0.161 | Ground-level field images overlap |
| 4 | CL_Alberto -- CL_StaRosa | 0.184 | Same Chilean geographic origin |
| 5 | MI_3Season -- ND_Indiv | 0.202 | US ground-level cameras share characteristics |

**Five most distant pairs (highest MMD):**

| Rank | Pair | MMD | Interpretation |
|------|------|-----|----------------|
| 32 | CL_StaRosa -- WeedCube | 0.544 | Chilean field vs. hyperspectral |
| 33 | ND_Aerial -- WeedCube | 0.561 | Drone RGB vs. hyperspectral |
| 34 | CL_Seba -- WeedCube | 0.482 | Roboflow-augmented vs. hyperspectral |
| 35 | MI_3Season -- WeedCube | 0.527 | Consumer cameras vs. hyperspectral |
| 36 | **Purdue -- WeedCube** | **0.639** | **UTV webcam vs. hyperspectral (maximum)** |

### 4.5 Per-Database Generalization Risk

Average MMD to all other databases -- a composite measure of how "isolated" each database is in embedding space:

| Database | N Images | Avg MMD | Avg PAD | Risk Level |
|----------|----------|---------|---------|------------|
| WeedCube | 20 | 0.518 | 2.00 | Highest -- hyperspectral is a fundamentally different domain |
| Purdue | 150 | 0.428 | 2.00 | High -- uniform UTV webcam creates narrow distribution |
| CL_Seba | 2,065 | 0.383 | 2.00 | High -- Roboflow augmentation creates internally homogeneous but externally isolated cluster |
| CL_Alberto | 659 | 0.328 | 1.99 | Moderate |
| CL_StaRosa | 75 | 0.316 | 1.99 | Moderate |
| ND_Aerial | 277 | 0.309 | 1.96 | Moderate |
| MI_3Season | 719 | 0.281 | 1.99 | Low-moderate |
| WeedCrop | 495 | 0.272 | 1.96 | Low -- most internally diverse, bridges multiple databases |
| ND_Indiv | 878 | 0.267 | 1.99 | Lowest -- high internal diversity |

**Key insight:** The most generalizable databases (WeedCrop, ND_Indiv) are those with the most internal diversity -- multiple crop backgrounds, growth stages, and photo sessions. Diversity within a database translates to proximity to other databases in embedding space.

### 4.6 Hierarchical Clustering

Ward's linkage dendrogram on the MMD distance matrix reveals nested structure:

- **Inner core:** WeedCrop and ND_Aerial merge first (MMD = 0.060), joined by MI_3Season and ND_Indiv -- these 4 databases share similar field photography conditions
- **Chilean cluster:** CL_Alberto and CL_StaRosa form a pair (MMD = 0.184), reflecting shared geographic origin
- **Isolated databases:** CL_Seba clusters separately due to Roboflow augmentation homogenizing its visual signature. Purdue (UTV webcam) and WeedCube (hyperspectral) are the most distant outliers

### 4.7 Proxy A-Distance

All PAD values are near the maximum (2.0), confirming that a simple linear classifier can almost perfectly identify which database any image comes from. Every database has a distinct visual signature in embedding space.

---

## 5. Deployment Gate Framework

A distribution-aware deployment strategy based on monitoring MMD between training data and incoming deployment data:

| MMD Range | Interpretation | Recommended Action |
|-----------|---------------|--------------------|
| < 0.15 | Distributions overlap substantially | Deploy directly. Augmentation sufficient. |
| 0.15 -- 0.30 | Moderate domain gap | Deploy with aggressive augmentation + test-time augmentation. Monitor performance. |
| 0.30 -- 0.45 | Significant domain gap | **Active learning required.** Sample and label representative images from target domain. |
| > 0.45 | Extreme domain gap | Do not deploy without retraining. Treat as new domain requiring substantial labeled data. |

### 5.1 Empirical Validation

The cross-domain detection experiment (Evidence 04) validated these thresholds:

| Scenario | MMD Range | Prediction | Observed Result |
|----------|-----------|------------|-----------------|
| CL model -> International data | 0.27 -- 0.43 | Active learning / new data required | mAP50 = 0.108 (failure, 78pp drop from 0.886) |
| Combined CL + INTL training | same | Training with target data resolves gap | mAP50 = 0.874 on international (success) |

The MMD distances of 0.27--0.43 between Chilean and international databases correctly predicted that the Chilean-only model would fail on international data, and that including target-domain training data would be necessary.

---

## 6. Key Findings

1. **Acquisition conditions dominate the embedding space.** The strongest clustering signal is camera type, resolution, field-of-view, lighting, and background vegetation -- not ragweed morphology. Images from the same database cluster together regardless of the specific ragweed plant photographed.

2. **Chilean vs International separation is pronounced.** Geographic and equipment differences create a clear domain gap (MMD 0.27--0.43). A model trained exclusively on one origin will fail on the other.

3. **Roboflow augmentation creates homogeneity.** The two Roboflow-augmented Chilean datasets (Seba, Alberto) form very dense clusters, suggesting that augmentation reduces visual diversity rather than increasing it in embedding space.

4. **Aerial imagery occupies a distinct region.** The NorthDakota Aerial dataset is well-separated from all ground-level sources, confirming that drone imagery requires separate handling or domain adaptation.

5. **The most generalizable databases are the most diverse ones.** Internal diversity (multiple cameras, backgrounds, growth stages) translates to external compatibility. This argues for collecting training data with maximum acquisition diversity rather than maximum image count under uniform conditions.

6. **MMD provides a continuous, label-free measure of domain gap** that serves as a practical deployment gate for agricultural advisory systems.

---

## 7. Implications for Deployment

- **Domain-balanced sampling is essential** to prevent CL_Seba (39% of all images) from dominating training
- **WeedCrop and ND_Indiv serve as natural bridge databases** (lowest Avg MMD) due to internal diversity
- **WeedCube and Purdue should be treated as separate evaluation benchmarks** rather than training sources (Avg MMD > 0.42)
- **Active learning** is the pragmatic solution for agricultural deployment: monitor distributions with MMD and target labeling effort where domain gaps exceed 0.30
- **MMD computation requires no labels** -- only the images and a feature extractor -- making it practical for pre-deployment assessment

---

## 8. Generated Visualizations

### 8.1 Interactive HTML Visualizations (Embedding Projections)

| File | Method | Colored By | Key Insight |
|------|--------|------------|-------------|
| `viz_umap_2d_by_database.html` | UMAP 2D | source_collection (9 groups) | Strong database-specific clustering |
| `viz_umap_2d_by_dbgroup.html` | UMAP 2D | database (5 groups) | DB01a+01b merge; CL sources cluster together |
| `viz_umap_2d_by_origin.html` | UMAP 2D | origin (Chilean/International) | Clear geographic separation |
| `viz_umap_2d_by_labelled.html` | UMAP 2D | labelled (True/False) | Unlabelled Sta Rosa cluster visible |
| `viz_umap_3d_by_database.html` | UMAP 3D | source_collection | 3D confirms separation with depth structure |
| `viz_tsne_2d_by_database.html` | t-SNE 2D | source_collection | Tighter clusters, clearer sub-structure |
| `viz_pca_2d_by_database.html` | PCA 2D | source_collection | Less separation; linear limitations |

Each HTML file is ~70 MB (self-contained with embedded base64 thumbnails) and features a FiftyOne-inspired dark theme with a 450px side panel for image inspection.

### 8.2 Static PNG Snapshots

Corresponding PNG snapshots for each HTML visualization:

| File | Description |
|------|-------------|
| `viz_umap_2d_by_database.png` | UMAP 2D colored by database |
| `viz_umap_2d_by_dbgroup.png` | UMAP 2D colored by database group |
| `viz_umap_2d_by_origin.png` | UMAP 2D colored by origin |
| `viz_umap_2d_by_labelled.png` | UMAP 2D colored by labelled status |
| `viz_umap_3d_by_database.png` | UMAP 3D colored by database |
| `viz_tsne_2d_by_database.png` | t-SNE 2D colored by database |
| `viz_pca_2d_by_database.png` | PCA 2D colored by database |

### 8.3 Domain Distance Figures (Publication Quality)

Generated by `generate_figures.py` for the book chapter:

| File | Description |
|------|-------------|
| `fig_heatmap_combined.png` | Side-by-side 9x9 MMD and PAD heatmaps |
| `fig_heatmap_mmd.png` | Full-resolution MMD pairwise distance matrix |
| `fig_heatmap_pad.png` | Full-resolution PAD pairwise matrix |
| `fig_mmd_dendrogram.png` | Ward's linkage dendrogram on MMD distances |
| `fig_generalization_risk_barplot.png` | Per-database average MMD barplot |
| `fig_mmd_network.png` | Database similarity network (edges for MMD < 0.25) |
| `fig_umap_with_centroids.png` | UMAP 2D with database centroids marked |
| `fig_origin_separation.png` | UMAP 2D colored by Chilean vs International origin |
| `fig_active_learning_concept.png` | Conceptual illustration of active learning for bridging domain gaps |

---

## 9. Data Paths

### 9.1 Embedding Data (External Drive)

All embedding outputs are stored on the external drive:

```
/media/malezainia2/E/rageweed_international_databases/ragweed_embeddings/
  embeddings.npz              # 19 MB -- Raw embeddings (5,338 x 2,048) + metadata dicts
  embeddings_reduced.csv      # 1.6 MB -- UMAP/t-SNE/PCA coordinates with metadata
  config.json                 # 3 KB -- Pipeline configuration and parameters
  thumbnails_cache.json       # 68 MB -- 200x200 base64 JPEG thumbnails
  mmd_matrix.csv              # 9x9 MMD pairwise distance matrix
  pad_matrix.csv              # 9x9 PAD pairwise distance matrix
  generalization_risk_scores.csv  # Per-database avg MMD and avg PAD
  viz_*.html                  # 7 interactive HTML visualizations (~70 MB each)
  viz_*.png                   # 7 static PNG snapshots
  heatmap_combined.png        # MMD + PAD side-by-side heatmap
  heatmap_mmd.png             # MMD-only heatmap
  heatmap_pad.png             # PAD-only heatmap
```

### 9.2 Publication Figures

```
/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/malezainia2/resultados/figures/
  fig_heatmap_combined.png
  fig_heatmap_mmd.png
  fig_heatmap_pad.png
  fig_mmd_dendrogram.png
  fig_generalization_risk_barplot.png
  fig_mmd_network.png
  fig_umap_with_centroids.png
  fig_origin_separation.png
  fig_active_learning_concept.png
```

### 9.3 Reports and Documentation

```
/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/malezainia2/ragweed_embeddings/
  RAGWEED_EMBEDDINGS_REPORT.md   # Full methodology and results report (v3.0)
  RAGWEED_PHOTO_INVENTORY.md     # Complete per-image inventory of all 5,338 images

/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/malezainia2/resultados/
  05_domain_shift_generalization_analysis.md   # Full analysis write-up with figures
  05_domain_shift_generalization_analysis.tex  # LaTeX version for PDF generation
```

### 9.4 Source Image Databases

```
/media/malezainia2/E/rageweed_international_databases/
  01_ImageWeeds_NorthDakota/Aerial_Weeds/Images/          # DB01a (277 images)
  01_ImageWeeds_NorthDakota/Individual_Weed/ragweed/images/  # DB01b (878 images)
  03_3SeasonWeedDet10_Michigan/                            # DB03 (719 images)
  05_4Weed_Purdue/ragweed/images/                          # DB05 (150 images)
  06_WeedCube_USDA/ragweed/                                # DB06 (20 images)
  7.include this/.../Weed-crop RGB dataset/                # DB07 (495 images)
/media/malezainia2/E/ProcessingData/
  ambrosia.Seba/                                           # CL_Seba (2,065 images)
  ambrosia.dataset_alberto/ambrosia.dataset/               # CL_Alberto (659 images)
/media/malezainia2/D/extracted/
  Ambrosia.Sta.Rosa/fotos.proyecto01-02-23/                # CL_StaRosa (75 images)
```

### 9.5 Whitelist Files (Multi-Class Filtering)

```
/media/malezainia2/E/rageweed_international_databases/
  whitelist_01a_aerial.txt     # 277 filenames
  whitelist_03_michigan.txt    # 719 filenames
  whitelist_07_weedcrop.txt    # 495 filenames
```

---

## 10. Code to Reproduce

### 10.1 Embedding Extraction Pipeline

**Script:** `/media/malezainia2/E/rageweed_international_databases/generate_ragweed_embeddings.py`

```bash
# Activate conda environment
conda activate base  # requires torch, torchvision, umap-learn, plotly, pandas, numpy, pillow

# Run full pipeline (reuses cached embeddings if available)
cd /media/malezainia2/E/rageweed_international_databases
python generate_ragweed_embeddings.py

# Force complete regeneration
python generate_ragweed_embeddings.py --force

# Regenerate only HTML visualizations (fast, ~10 seconds)
python generate_ragweed_embeddings.py --viz-only

# Quick test with limited images
python generate_ragweed_embeddings.py --max-images 500

# Custom parameters
python generate_ragweed_embeddings.py --batch-size 32 --device cuda:0 --umap-neighbors 30
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 16 | Batch size for embedding extraction |
| `--device` | cuda:1 | PyTorch device |
| `--max-images` | all | Limit total images (for testing) |
| `--num-workers` | 4 | DataLoader workers |
| `--output-dir` | `ragweed_embeddings/` | Output directory |
| `--force` | false | Force regeneration of embeddings |
| `--viz-only` | false | Regenerate only HTML visualizations |
| `--skip-dimreduce` | false | Skip dimensionality reduction and visualization |
| `--umap-neighbors` | 15 | UMAP n_neighbors parameter |
| `--umap-min-dist` | 0.1 | UMAP min_dist parameter |
| `--tsne-perplexity` | 30 | t-SNE perplexity parameter |

### 10.2 Domain Distance Computation

**Script:** `/media/malezainia2/E/rageweed_international_databases/ragweed_embeddings/compute_domain_distances.py`

```bash
# Compute pairwise MMD and Proxy A-Distance
cd /media/malezainia2/E/rageweed_international_databases/ragweed_embeddings
python compute_domain_distances.py
```

Reads `embeddings.npz`, produces `mmd_matrix.csv`, `pad_matrix.csv`, `generalization_risk_scores.csv`, and heatmap PNGs.

### 10.3 Publication Figure Generation

**Script:** `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/malezainia2/resultados/figures/generate_figures.py`

```bash
# Generate all 9 publication-quality figures
cd /home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/malezainia2/resultados/figures
python generate_figures.py
```

Reads data from external drive (`mmd_matrix.csv`, `generalization_risk_scores.csv`, `embeddings_reduced.csv`), produces 9 PNG figures in the same directory.

---

## 11. Pipeline Configuration (config.json)

```json
{
  "model": "resnet50",
  "embedding_dim": 2048,
  "batch_size": 16,
  "num_workers": 4,
  "max_images": null,
  "device": "cuda:1",
  "n_images": 5338,
  "sources": [
    {"name": "01_NorthDakota_Aerial", "whitelist": "whitelist_01a_aerial.txt", "labelled": true},
    {"name": "01_NorthDakota_Individual", "labelled": true},
    {"name": "03_Michigan_3Season", "whitelist": "whitelist_03_michigan.txt", "labelled": true},
    {"name": "05_Purdue_4Weed", "labelled": true},
    {"name": "06_USDA_WeedCube", "filter_ext": ".png", "labelled": true},
    {"name": "07_WeedCrop_PrecAg", "whitelist": "whitelist_07_weedcrop.txt", "labelled": true},
    {"name": "CL_Seba_Labelled", "labelled": true},
    {"name": "CL_Alberto_Labelled", "labelled": true},
    {"name": "CL_StaRosa_Unlabelled", "labelled": false}
  ],
  "timestamp": "2026-02-21T21:44:28"
}
```

---

## 12. Output File Sizes

| File | Size | Description |
|------|------|-------------|
| `embeddings.npz` | 19 MB | Raw embeddings (5,338 x 2,048) + metadata |
| `embeddings_reduced.csv` | 1.6 MB | UMAP/t-SNE/PCA coordinates with metadata |
| `thumbnails_cache.json` | 68 MB | 200x200 base64 JPEG thumbnails for hover |
| `config.json` | 3 KB | Pipeline configuration and parameters |
| `mmd_matrix.csv` | <1 KB | 9x9 MMD pairwise distance matrix |
| `pad_matrix.csv` | <1 KB | 9x9 PAD pairwise distance matrix |
| `generalization_risk_scores.csv` | <1 KB | Per-database avg MMD and PAD |
| `viz_*.html` (x7) | ~70 MB each | Interactive visualizations with embedded thumbnails |
| `viz_*.png` (x7) | ~0.5--1.1 MB each | Static PNG snapshots |
| `fig_*.png` (x9) | ~0.2--0.5 MB each | Publication-quality figures |

---

## 13. Software Dependencies

| Component | Version / Details |
|-----------|-------------------|
| **GPU** | NVIDIA RTX 4090 (24 GB VRAM) via `cuda:1` |
| **PyTorch** | torch + torchvision (ResNet50 IMAGENET1K_V2) |
| **Dimensionality reduction** | umap-learn, scikit-learn (t-SNE, PCA) |
| **Domain distances** | scikit-learn (LinearSVC for PAD), numpy (RBF kernel for MMD) |
| **Visualization** | Plotly (scatter plots + custom HTML/CSS/JS), matplotlib + seaborn (publication figures) |
| **Clustering** | scipy (Ward's linkage dendrogram) |
| **Image processing** | Pillow (thumbnails) |
| **Data handling** | pandas, numpy |
| **Conda environment** | `base` |

---

## 14. References

- Ben-David, S., et al. (2007, 2010). A theory of learning from different domains. *Machine Learning*.
- Gretton, A., et al. (2012). A kernel two-sample test. *Journal of Machine Learning Research*, 13, 723--773.
- Torralba, A. & Efros, A. A. (2011). Unbiased look at dataset bias. *CVPR*.
- Barbedo, J. G. A. (2018). Impact of dataset size and variety on the effectiveness of deep learning. *Computers and Electronics in Agriculture*.
- Beery, S., et al. (2018). Recognition in terra incognita. *ECCV*.
- Settles, B. (2009). Active learning literature survey. *University of Wisconsin-Madison*.

---

*Evidence compiled 2026-02-23. Source data generated 2026-02-21.*
