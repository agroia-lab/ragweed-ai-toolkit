# Ragweed AIToolkit — Session 5: GitHub + CI/CD + Documentation Site

## Context

This is Session 5 of the ragweed-ai-toolkit multi-session build.

**Previous sessions (all DONE):**
- Session 1: Package scaffold + 8 core modules (detection, embeddings, geostatistics, spatial, satellite, crossdomain, orthomosaic, viz) — 38 source files, 6K lines
- Session 2: Viz module, pytest test suite (8 test files), 6 example scripts, CLAUDE.md
- Session 3: Refactored 27 tools/ CLI scripts to import from ragweed_toolkit
- Session 4: 8 CLI entry points via pyproject.toml, 4 Jupyter notebook tutorials, environment.yml

**Toolkit location:** `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/research_docs/lencu_book_chapter/toolkit/`

**Important — Dual git repos:**
The toolkit has its own `.git` at the toolkit root (3 commits, NO remote).
It is ALSO tracked as a subdirectory inside the parent INIA project.
For Session 5, work ONLY in the toolkit's own git repo.

**Uncommitted changes:** Sessions 3-4 work is uncommitted in the toolkit repo.
Before starting, commit these changes:
```bash
cd /home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/research_docs/lencu_book_chapter/toolkit
git add -A
git commit -m "Sessions 3-4: Refactor tools/ + CLI entry points + notebooks + environment.yml"
```

Read `docs/SESSION_PLAN.md` and `CLAUDE.md` for the full roadmap and package structure.

## Goal

Make the toolkit publicly available: push to GitHub, set up CI/CD, build a documentation site.

## Prerequisites

Before starting, check and install if needed:
```bash
# Check gh CLI
gh --version || echo "NEED TO INSTALL: sudo apt install gh && gh auth login"

# Check mkdocs
pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-jupyter
```

## Task Breakdown

Use an agent team with 2 agents:

### Agent 1: GitHub Repository + CI/CD + Release

#### Step 1: Create GitHub repository

```bash
cd /home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/research_docs/lencu_book_chapter/toolkit

# Create repo (ask user to confirm org name first)
gh repo create agroia-lab/ragweed-ai-toolkit --public \
    --description "Multi-scale AI toolkit for weed detection, mapping, and spatial analysis" \
    --source . --push
```

If `gh` is not available, guide the user to create the repo manually on GitHub, then:
```bash
git remote add origin git@github.com:agroia-lab/ragweed-ai-toolkit.git
git push -u origin master
```

#### Step 2: GitHub Actions — Tests

Create `.github/workflows/test.yml`:
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --tb=short
```

**Note:** Use `.[dev]` not `.[all]` — tests use synthetic data and don't need
ultralytics, earthengine-api, torch, etc. The core dependencies (numpy, scipy,
scikit-learn, geopandas, matplotlib, pandas) are enough for the test suite.

#### Step 3: GitHub Actions — Lint

Create `.github/workflows/lint.yml`:
```yaml
name: Lint
on: [push, pull_request]
jobs:
  ruff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install ruff
      - run: ruff check src/ tools/ examples/
```

#### Step 4: Create CONTRIBUTING.md

Cover:
- Development setup (`pip install -e ".[all,dev]"`)
- Running tests (`pytest tests/ -v`)
- Code style (ruff, line-length 100)
- PR workflow
- Module structure overview

#### Step 5: Create CHANGELOG.md

```markdown
# Changelog

## [0.1.0] - 2026-02-27

### Added
- 8 core modules: detection, embeddings, geostatistics, spatial, satellite, crossdomain, orthomosaic, viz
- 8 CLI entry points: ragweed-{train,sahi,embed,mmd,tile,kriging,lisa,indices}
- 27 CLI tool scripts in tools/
- 6 example scripts
- 4 Jupyter notebook tutorials
- pytest test suite (8 test files)
- conda environment.yml
```

#### Step 6: Add badges to README.md

Add at the top of README.md:
```markdown
[![Tests](https://github.com/agroia-lab/ragweed-ai-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/agroia-lab/ragweed-ai-toolkit/actions/workflows/test.yml)
[![Lint](https://github.com/agroia-lab/ragweed-ai-toolkit/actions/workflows/lint.yml/badge.svg)](https://github.com/agroia-lab/ragweed-ai-toolkit/actions/workflows/lint.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
```

#### Step 7: Tag v0.1.0 release

```bash
git tag -a v0.1.0 -m "Initial release: 8 modules, 8 CLI commands, 4 notebooks"
git push origin v0.1.0
```

### Agent 2: Documentation Site (mkdocs)

#### Step 1: Create mkdocs.yml

```yaml
site_name: Ragweed AI Toolkit
site_description: Multi-scale AI toolkit for weed detection, mapping, and spatial analysis
site_url: https://agroia-lab.github.io/ragweed-ai-toolkit
repo_url: https://github.com/agroia-lab/ragweed-ai-toolkit
repo_name: agroia-lab/ragweed-ai-toolkit

theme:
  name: material
  palette:
    - scheme: default
      primary: green
      accent: amber
  features:
    - navigation.tabs
    - navigation.sections
    - content.code.copy

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [src]
          options:
            show_source: true
            show_root_heading: true

nav:
  - Home: index.md
  - Installation: installation.md
  - CLI Reference: cli.md
  - Tutorials:
    - Detection Workflow: tutorials/detection.md
    - Cross-Domain Analysis: tutorials/crossdomain.md
    - Spatial Mapping: tutorials/spatial.md
    - Satellite Integration: tutorials/satellite.md
  - API Reference:
    - detection: api/detection.md
    - embeddings: api/embeddings.md
    - crossdomain: api/crossdomain.md
    - orthomosaic: api/orthomosaic.md
    - geostatistics: api/geostatistics.md
    - spatial: api/spatial.md
    - satellite: api/satellite.md
    - viz: api/viz.md
  - Chapter Results: chapter.md
  - Contributing: contributing.md
  - Changelog: changelog.md
```

#### Step 2: Create documentation pages

All docs go in a `mkdocs_docs/` directory (to avoid conflict with existing `docs/`):

1. **mkdocs_docs/index.md** — Based on README.md content, adapted for web
2. **mkdocs_docs/installation.md** — Detailed install guide:
   - pip install (core, extras, all)
   - conda environment.yml
   - Development setup
3. **mkdocs_docs/cli.md** — CLI reference for all 8 commands:
   - ragweed-train, ragweed-sahi, ragweed-embed, ragweed-mmd
   - ragweed-tile, ragweed-kriging, ragweed-lisa, ragweed-indices
   - Include `--help` output for each
4. **mkdocs_docs/api/*.md** — One per module, using mkdocstrings:
   ```markdown
   # Detection Module
   ::: ragweed_toolkit.detection
   ```
5. **mkdocs_docs/tutorials/*.md** — Narrative guides extracted from notebooks:
   - detection.md, crossdomain.md, spatial.md, satellite.md
6. **mkdocs_docs/chapter.md** — Key results from the Springer chapter:
   - Best mAP50: 0.886
   - Cross-domain collapse/recovery
   - NDVI-AMBEL correlation: r = 0.837
   - GWR R² AMBEL: 0.882
   - Citation block for the chapter
7. **mkdocs_docs/contributing.md** — Symlink or copy from CONTRIBUTING.md
8. **mkdocs_docs/changelog.md** — Symlink or copy from CHANGELOG.md

#### Step 3: Update mkdocs.yml docs_dir

```yaml
docs_dir: mkdocs_docs
```

#### Step 4: Build and test locally

```bash
mkdocs build
mkdocs serve  # Preview at http://localhost:8000
```

#### Step 5: GitHub Actions — Docs deployment

Create `.github/workflows/docs.yml`:
```yaml
name: Deploy Docs
on:
  push:
    branches: [master]
permissions:
  contents: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install mkdocs mkdocs-material mkdocstrings[python]
      - run: mkdocs gh-deploy --force
```

## Important Notes

- The toolkit repo is at `.../toolkit/` inside the INIA project — work from there
- Ask user to confirm `agroia-lab` as the GitHub org before creating the repo
- If `gh` CLI is not installed, provide manual GitHub instructions
- Use `mkdocs_docs/` instead of `docs/` to avoid conflicts with existing `docs/` dir
- Tests should run with `.[dev]` only (no GPU/EE dependencies needed)
- The docs site uses GitHub Pages (free for public repos)

## Commit Message
```
Session 5: GitHub setup + CI/CD workflows + mkdocs documentation site
```
