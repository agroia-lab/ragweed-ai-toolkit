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
| 3 | Refactor tools/ to import from package | S2 | PENDING |
| 4 | CLI entry points + notebook tutorials | S3 | PENDING |
| 5 | GitHub setup + CI/CD + docs site | S4 | PENDING |
| 6 | Integration testing with real data | S5 | PENDING |

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

## Session 4 — CLI Entry Points + Notebook Tutorials

**Goal:** Add proper CLI entry points via pyproject.toml console_scripts so
users can run `ragweed-train`, `ragweed-sahi`, etc. Create Jupyter notebook
tutorials for each major workflow.

**Scope:**
1. Add [project.scripts] to pyproject.toml:
   - `ragweed-train` → ragweed_toolkit.detection.trainer:main
   - `ragweed-sahi` → ragweed_toolkit.detection.inference:main
   - `ragweed-embed` → ragweed_toolkit.embeddings.extractor:main
   - `ragweed-mmd` → ragweed_toolkit.embeddings.mmd:main
   - `ragweed-tile` → ragweed_toolkit.orthomosaic.tiling:main
   - `ragweed-kriging` → ragweed_toolkit.geostatistics.kriging:main
   - `ragweed-lisa` → ragweed_toolkit.spatial.lisa:main
2. Add main() functions with argparse to each target module
3. Create notebooks/:
   - 01_detection_workflow.ipynb (train → infer → GPS → shapefile)
   - 02_crossdomain_analysis.ipynb (embeddings → MMD → deployment gates)
   - 03_spatial_mapping.ipynb (kriging → LISA → GWR)
   - 04_satellite_integration.ipynb (PRESTO → indices → risk zones)
4. Create conda environment.yml for reproducible setup

**Agent team:** 2 agents (CLI entry points, notebooks)

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

## Session 6 — Integration Testing with Real Data

**Goal:** Validate the full toolkit pipeline works end-to-end with real
project data from the external drive.

**Scope:**
1. End-to-end detection test:
   - Load CL_Seba test images
   - Run SAHI inference with best.pt
   - Extract GPS → export shapefile
   - Compare with known ground truth counts
2. End-to-end embedding test:
   - Extract embeddings from 9 databases
   - Compute pairwise MMD matrix
   - Verify matches chapter Table values
3. End-to-end spatial test:
   - Load Santa Rosa kriging grid
   - Run bivariate LISA (PC1 × AMBEL)
   - Verify 68% significant pixels
4. Create integration test script: tests/test_integration.py
5. Document data paths and expected results
6. Final cleanup, version bump to v0.2.0

**Agent team:** 2 agents (detection+embedding pipeline, spatial+satellite pipeline)

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
