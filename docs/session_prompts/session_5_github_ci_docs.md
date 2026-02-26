# Ragweed AIToolkit — Session 5: GitHub + CI/CD + Documentation Site

## Context

This is Session 5 of the ragweed-ai-toolkit multi-session build.

**Previous sessions:**
- Session 1: Package scaffold + 7 core modules
- Session 2: Viz module, unit tests, examples, CLAUDE.md
- Session 3: Refactored tools/ scripts to import from package
- Session 4: CLI entry points + notebook tutorials

**Repository:** `/home/malezainia2/dev/dlm6-weed-toolkit/`

Read `docs/SESSION_PLAN.md` for the full roadmap.

## Goal

Make the toolkit publicly available: push to GitHub, set up CI/CD, build
a documentation site.

## Task Breakdown

Use an agent team with 2 agents:

### Agent 1: GitHub + CI/CD

1. **Create GitHub repository:**
   ```bash
   gh repo create agroia-lab/ragweed-ai-toolkit --public --description \
       "Multi-scale AI toolkit for weed detection, mapping, and spatial analysis"
   git remote add origin git@github.com:agroia-lab/ragweed-ai-toolkit.git
   git push -u origin master
   ```

2. **GitHub Actions — Tests:**
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
         - run: pip install -e ".[all,dev]"
         - run: pytest tests/ -v --tb=short
   ```

3. **GitHub Actions — Lint:**
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
         - run: ruff check src/
   ```

4. **Add badges to README.md:**
   ```markdown
   [![Tests](https://github.com/agroia-lab/ragweed-ai-toolkit/actions/workflows/test.yml/badge.svg)](...)
   [![Lint](https://github.com/agroia-lab/ragweed-ai-toolkit/actions/workflows/lint.yml/badge.svg)](...)
   ```

5. **Create CONTRIBUTING.md** — development workflow, PR guidelines

6. **Create CHANGELOG.md** — session-based history

7. **Tag v0.1.0 release:**
   ```bash
   git tag -a v0.1.0 -m "Initial release: 7 modules, 24 sub-modules"
   git push origin v0.1.0
   ```

### Agent 2: Documentation Site (mkdocs)

1. **Install mkdocs:**
   ```bash
   pip install mkdocs mkdocs-material mkdocstrings[python]
   ```

2. **Create mkdocs.yml:**
   ```yaml
   site_name: Ragweed AI Toolkit
   site_url: https://agroia-lab.github.io/ragweed-ai-toolkit
   theme:
     name: material
     palette:
       primary: green
       accent: amber
   plugins:
     - search
     - mkdocstrings:
         handlers:
           python:
             paths: [src]
   nav:
     - Home: index.md
     - Installation: installation.md
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

3. **Create docs pages:**
   - docs/index.md (from README)
   - docs/installation.md (detailed install guide)
   - docs/api/*.md (`::: ragweed_toolkit.detection` for each module)
   - docs/tutorials/*.md (from notebook content)
   - docs/chapter.md (key results + citation)

4. **Build and deploy:**
   ```bash
   mkdocs build
   mkdocs gh-deploy  # Pushes to gh-pages branch
   ```

5. **Add GitHub Actions for docs:**
   `.github/workflows/docs.yml` — auto-deploy on push to master

## Important Notes

- Ask user to confirm GitHub org name (agroia-lab) before creating repo
- Check if user has `gh` CLI authenticated
- The docs site uses GitHub Pages (free for public repos)

## Commit Message
```
Session 5: GitHub setup + CI/CD workflows + mkdocs documentation site
```
