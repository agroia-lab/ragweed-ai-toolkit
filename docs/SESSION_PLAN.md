# Ragweed AIToolkit — Multi-Session Plan

## Overview

Transform the DLM6 Springer chapter companion code into a publishable
open-source Python toolkit. Each session runs independently in a
separate Claude Code window.

**Repository:** `/home/malezainia2/dev/dlm6-weed-toolkit/`
**Package:** `ragweed-ai-toolkit` (import as `ragweed_toolkit`)

---

## Session Map

| Session | Focus | Depends on | Status |
|---------|-------|------------|--------|
| 1 | Package scaffold + 7 core modules | — | DONE |
| 2 | Viz module + unit tests + examples + CLAUDE.md | S1 | DONE |
| 3 | Refactor tools/ to import from package | S2 | DONE |
| 4 | CLI entry points + notebook tutorials | S3 | DONE |
| 5 | GitHub setup + CI/CD + docs site | S4 | DONE |
| 6 | Integration testing with real data | S5 | DONE |

---

## Session 1 — Package Scaffold + Core Modules (DONE)

**Completed 2026-02-26**

Created `src/ragweed_toolkit/` with 7 sub-packages (24 modules, 6K lines):
- detection (trainer, inference, gps, evaluation)
- embeddings (extractor, mmd, reduction, viz)
- crossdomain (dataset_builder, evaluation, comparison)
- orthomosaic (tiling, geo_export, active_learning)
- geostatistics (variograms, kriging)
- spatial (morans, lisa, gwr, cross_variograms)
- satellite (composites, ndvi, indices, presto)

Also: pyproject.toml, LICENSE (MIT), .gitignore, README.md, docs/ARCHITECTURE.md

**Commit:** `f86a776`

---

## Session 2 — Viz + Tests + Examples (DONE)

**Completed 2026-02-26**

- viz/ module: maps, scatter, heatmaps, panels, style (publication-quality 300 DPI)
- Unit tests for all modules (pytest, synthetic data)
- Example scripts (6 demos showing library usage)
- CLAUDE.md updated for package structure

---

## Session 3 — Refactor tools/ to Import from Package

**Goal:** Make the 27 existing CLI scripts in tools/ import their core logic
from `ragweed_toolkit` instead of being monolithic. This validates the
library API works end-to-end.

**Scope:**
1. Refactor tools/01_detection/ (2 scripts) → import from ragweed_toolkit.detection
2. Refactor tools/02_embedding/ (5 scripts) → import from ragweed_toolkit.embeddings
3. Refactor tools/03_crossdomain/ (4 scripts) → import from ragweed_toolkit.crossdomain
4. Refactor tools/04_orthomosaic/ (3 scripts) → import from ragweed_toolkit.orthomosaic
5. Refactor tools/05_geostatistics/ (2 scripts) → import from ragweed_toolkit.geostatistics + spatial
6. Refactor tools/06_satellite/ (11 scripts) → import from ragweed_toolkit.satellite + spatial
7. Verify all refactored scripts still run (--help at minimum)

**Agent team:** 3 agents (detection+embedding, crossdomain+ortho, spatial+satellite)

**Expected output:** 27 refactored scripts, all importing from ragweed_toolkit

---

## Session 4 — CLI Entry Points + Notebook Tutorials (DONE)

**Completed 2026-02-27**

- 8 CLI entry points via `[project.scripts]` in pyproject.toml
- `main()` functions with argparse in all 8 target modules
- 4 Jupyter notebook tutorials with synthetic data:
  - 01_detection_workflow.ipynb (14KB)
  - 02_crossdomain_analysis.ipynb (11KB)
  - 03_spatial_mapping.ipynb (16KB)
  - 04_satellite_integration.ipynb (17KB)
- environment.yml for conda reproducible setup
- Fixed lazy imports for ultralytics (optional dependency)
- All 8 commands verified: `ragweed-{train,sahi,embed,mmd,tile,kriging,lisa,indices} --help`

---

## Session 5 — GitHub + CI/CD + Documentation Site

**Goal:** Push to GitHub, set up automated testing, build documentation site.

**Scope:**
1. Create GitHub repo at agroia-lab/ragweed-ai-toolkit
2. Set up GitHub Actions CI:
   - .github/workflows/test.yml (pytest on push/PR)
   - .github/workflows/lint.yml (ruff check)
   - Matrix: Python 3.10, 3.11, 3.12
3. Build mkdocs documentation site:
   - mkdocs.yml configuration
   - docs/index.md (from README)
   - docs/api/ — auto-generated API reference (mkdocstrings)
   - docs/tutorials/ — rendered from notebooks
   - docs/chapter/ — link to Springer chapter
4. Add badges to README (CI status, coverage, docs)
5. Create CONTRIBUTING.md
6. Create CHANGELOG.md
7. Tag v0.1.0 release

**Agent team:** 2 agents (CI/CD, docs site)

---

## Session 6 — Integration Testing with Real Data (DONE)

**Completed 2026-02-27**

Validated full toolkit pipeline end-to-end with real project data:

- **Detection pipeline:** SAHI inference on CL_Seba images (best.pt, 640px slices),
  GPS extraction → GeoPackage export, detection structure validation
- **Embedding pipeline:** ResNet-50 extraction (CL vs INTL), MMD domain shift
  computation, deployment gate tier mapping, 9-database pre-computed verification
- **Spatial pipeline:** Kriging raster loading (4 variogram models, EPSG:32719),
  PRESTO PC1 shapefile validation, bivariate LISA (68% significant, HH=29.5%, LL=32.6%)
- **Satellite pipeline:** 9 spectral indices (NDVI, EVI, GNDVI, RENDVI, S2WI, NBR2,
  BSI, Clay, SWIRd) from synthetic Sentinel-2 bands
- 2 integration test files: `test_integration_detection.py` (11 tests),
  `test_integration_spatial.py` (12 tests)
- All lint errors fixed, `slow` pytest marker registered
- Version bump to v0.2.0, tagged and pushed

**Final count:** 148 tests passing, 2 skipped (dependency checks)

**Agent team:** 2 agents (detection+embedding, spatial+satellite)

---

## Data Dependencies by Session

| Session | Requires data? | What |
|---------|---------------|------|
| 1 | No | Pure code |
| 2 | No | Synthetic/mock data only |
| 3 | No | Only --help verification |
| 4 | No | Notebooks use synthetic data |
| 5 | No | CI uses synthetic tests |
| 6 | **YES** | External drive data needed |
