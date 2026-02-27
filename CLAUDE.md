# CLAUDE.md — Ragweed AI Toolkit

## Project

This repository is the companion toolkit for a Springer Nature book chapter:
- **Title**: AI-Driven Spatial Decision-Support for Sustainable Weed Management under Climate Variability
- **Volume**: Smart-Agriculture and Technology-Innovation: Facing the Dynamic of Climate Change
- **Editor**: Prof. Noureddine Benkeblia, University of the West Indies
- **Publisher**: Springer Nature
- **Format**: LaTeX (svmult.cls), APA 7th Edition citations

## Package Name

**ragweed-ai-toolkit** (PyPI) / `ragweed_toolkit` (Python import)

## Installation

```bash
# Core (numpy, scipy, geopandas, matplotlib, sklearn, pandas)
pip install -e .

# With all optional modules
pip install -e ".[all]"

# Specific extras
pip install -e ".[detection]"      # ultralytics, sahi, opencv
pip install -e ".[embeddings]"     # torch, torchvision, umap-learn
pip install -e ".[spatial]"        # esda, libpysal, mgwr, spreg
pip install -e ".[satellite]"      # earthengine-api, rasterio
pip install -e ".[viz]"            # plotly, matplotlib-scalebar
pip install -e ".[dev]"            # pytest, ruff, pre-commit
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --tb=short        # shorter traceback
pytest tests/test_embeddings.py -v  # single module
```

## Package Structure

```
src/ragweed_toolkit/
├── detection/         # YOLO training + SAHI inference + GPS + evaluation
├── embeddings/        # ResNet-50 features, MMD, PCA/UMAP/t-SNE, viz
├── geostatistics/     # Variograms + ordinary kriging
├── spatial/           # Moran's I, LISA, GWR, cross-variograms
├── satellite/         # Sentinel-2 indices, PRESTO, NDVI, composites
├── crossdomain/       # Multi-source dataset builder + evaluation
├── orthomosaic/       # Raster tiling for drone orthomosaics
└── viz/               # Publication-quality figures (plotly + matplotlib)
```

## Import Patterns

```python
# Detection
from ragweed_toolkit.detection import TrainingConfig, train
from ragweed_toolkit.detection import SahiConfig, run_sahi, run_sahi_batch

# Embeddings
from ragweed_toolkit.embeddings import FeatureExtractor, extract_embeddings
from ragweed_toolkit.embeddings import compute_mmd, permutation_test, deployment_gate
from ragweed_toolkit.embeddings import reduce_embeddings

# Geostatistics
from ragweed_toolkit.geostatistics import (
    compute_experimental_variogram, fit_exponential_model, ordinary_kriging,
)

# Spatial
from ragweed_toolkit.spatial import compute_bivariate_morans, compute_bivariate_lisa
from ragweed_toolkit.spatial import fit_ols, fit_gwr, compare_ols_gwr
from ragweed_toolkit.spatial import compute_cross_variogram

# Satellite
from ragweed_toolkit.satellite import compute_spectral_indices

# Cross-domain
from ragweed_toolkit.crossdomain import build_international_dataset, build_combined_dataset
```

## Examples

Self-contained demo scripts in `examples/`:

| Script | What it demonstrates |
|--------|---------------------|
| `01_train_detector.py` | TrainingConfig setup (print-only, no GPU) |
| `02_sahi_field_inference.py` | SahiConfig + batch inference pattern |
| `03_embeddings_mmd.py` | MMD domain shift + deployment gate |
| `04_kriging_density.py` | Variogram fitting from synthetic points |
| `05_lisa_clusters.py` | Bivariate LISA with synthetic spatial data |
| `06_spectral_indices.py` | 9 Sentinel-2 indices from mock bands |

Run any example:
```bash
python examples/03_embeddings_mmd.py
```

## Key Message

"We explored and tested multiple tools for recognizing and mapping weeds at complementary
scales. None alone solves the problem. Together, they form a toolkit for decision-making
under climate variability -- and we put this repository into the community's hands."

## Three-Act Structure

- **Act I -- Field Detection**: YOLO + SAHI detection -> geostatistical mapping -> management zones
- **Act II -- Domain Generalization**: Cross-domain evaluation, MMD gates, multi-domain recovery
- **Act III -- Satellite Integration**: PRESTO embeddings, NDVI correlation, GWR, LISA clusters

## Key Results (Verified)

| Metric | Value | Source |
|--------|-------|--------|
| Best mAP50 (AMBEL detection) | 0.886 | Training Run 3 |
| Cross-domain collapse (CL->INTL) | mAP50 = 0.108 | Phase 1 |
| Multi-domain recovery | mAP50 = 0.874 | Phase 2 |
| NDVI-AMBEL correlation | r = 0.837 | Single-date Sept 24 |
| NDVI-LENCU correlation | r = 0.890 | Single-date Sept 24 |
| PRESTO PC1-AMBEL | r = 0.739 | Jul-Dec 2024 |
| PRESTO PC1-LENCU | r = 0.717 | Jul-Dec 2024 |
| Bivariate Moran's I (PC1 x AMBEL) | 0.706 (p=0.001) | 999 permutations |
| GWR R-squared AMBEL | 0.882 | PC1+PC2+PC3, BW=49nn |
| GWR R-squared LENCU | 0.910 | PC1+PC2+PC3, BW=49nn |
| LISA significant pixels | 68% | PC1 x AMBEL |

## Editorial Rules (MANDATORY for all text output)

- **Language**: American English (Merriam-Webster)
- **Citations**: APA 7th Edition. Ampersand inside parentheses: (Author & Author, Year). "and" in running text.
- **Species names**: Always italics: *Ambrosia artemisiifolia*, *Convolvulus arvensis*
- **Headings**: Decimal numbering (1, 1.1, 1.1.1). Cross-references by number: "see Sect. 1.2"
- **Abbreviations**: Define at first use (YOLO, SAHI, UAV, DSS, PRESTO, NDVI, LISA, mAP, GWR, OLS, MMD)
- **Figures**: Fig. N.1, Fig. N.2 (N = chapter number). Captions at end of text file.
- **Tables**: Table N.1, Table N.2. Built in LaTeX, not as images.

## CLI Entry Points (pip install)

After `pip install -e .`, these commands are available:

| Command | Module | Description |
|---------|--------|-------------|
| `ragweed-train` | detection.trainer | Train YOLO model with validated hyperparameters |
| `ragweed-sahi` | detection.inference | SAHI sliced inference on images |
| `ragweed-embed` | embeddings.extractor | Extract ResNet-50 CNN embeddings |
| `ragweed-mmd` | embeddings.mmd | Compute MMD + deployment gate |
| `ragweed-tile` | orthomosaic.tiling | Tile GeoTIFF orthomosaic for YOLO |
| `ragweed-kriging` | geostatistics.kriging | Ordinary kriging interpolation |
| `ragweed-lisa` | spatial.lisa | Bivariate LISA cluster analysis |
| `ragweed-indices` | satellite.indices | Compute 9 spectral indices from GeoTIFF |

```bash
ragweed-train --help
ragweed-mmd --source embeddings_a.npy --target embeddings_b.npy
ragweed-indices --input sentinel2.tif --output-dir indices/
```

## Notebook Tutorials

| Notebook | Workflow |
|----------|----------|
| `notebooks/01_detection_workflow.ipynb` | Train → SAHI → GPS → shapefile → map |
| `notebooks/02_crossdomain_analysis.ipynb` | Embeddings → MMD → deployment gates → UMAP |
| `notebooks/03_spatial_mapping.ipynb` | Variogram → kriging → LISA → GWR |
| `notebooks/04_satellite_integration.ipynb` | Indices → PRESTO → correlation → risk zones |

All notebooks use synthetic data and are self-contained.

## CLI Tools (tools/)

27 CLI scripts in `tools/` organized by module. Each is a thin wrapper that imports
core logic from `ragweed_toolkit` and adds argparse + formatted output.

```
tools/
├── 01_detection/          # YOLO training + SAHI inference (2 scripts)
├── 02_embedding/          # ResNet-50 embeddings, MMD domain shift (5 scripts)
├── 03_crossdomain/        # Dataset building + cross-domain evaluation (4 scripts)
├── 04_orthomosaic/        # Tiling, geo-export, active learning (3 scripts)
├── 05_geostatistics/      # LISA cluster analysis (2 scripts, pygeoda backend)
└── 06_satellite/          # NDVI, PRESTO, GWR, spectral indices (11 scripts)
```

Run any tool:
```bash
python tools/01_detection/train_yolo_cli.py --help
python tools/03_crossdomain/crossdomain_eval.py --model best.pt --manifest manifest.csv
python tools/06_satellite/extract_presto_embeddings.py --paddock all --season 25-26
```

**Note:** Geostatistics scripts use pygeoda (KNN weights) while the library uses
esda (DistanceBand weights). Core analysis functions have TODO comments for future
unification when the library API supports both backends.

## File Locations

| What | Where |
|------|-------|
| Package source | `src/ragweed_toolkit/` |
| CLI tools | `tools/` |
| Tests | `tests/` |
| Examples | `examples/` |
| Notebooks | `notebooks/` |
| Chapter LaTeX | `research_docs/lencu_book_chapter/chapter/main.tex` |
| Figures | `research_docs/lencu_book_chapter/chapter/figures/` |
| Evidence docs | `evidence/` |
| Configs | `configs/` |
| Data pointers | `data/README.md` |
| Narrative roadmap | `docs/CHAPTER_STORYLINE.md` |
| Conda env | `environment.yml` |

## Session History

| Session | Date | Focus |
|---------|------|-------|
| 1 | 2026-02-26 | Scaffold package, implement all 8 modules (detection, embeddings, geostatistics, spatial, satellite, crossdomain, orthomosaic, viz) |
| 2 | 2026-02-26 | Viz module implementation, pytest test suite, example scripts, CLAUDE.md update |
| 3 | 2026-02-27 | Refactor 27 tools/ CLI scripts to import from ragweed_toolkit (thin wrappers) |
| 4 | 2026-02-27 | CLI entry points (8 commands via pyproject.toml), 4 notebook tutorials, environment.yml |
