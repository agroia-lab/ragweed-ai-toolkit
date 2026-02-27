# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.2.0] - 2026-02-27

### Added
- Integration test suite for detection + embedding pipeline (`tests/test_integration_detection.py`)
  - SAHI batch inference with synthetic images and mock YOLO model
  - GPS extraction to GeoDataFrame and GeoPackage export
  - ResNet-50 embedding extraction with deterministic verification
  - MMD cross-domain computation and deployment gate tier validation
  - Pre-computed embedding loading (.npz) and MMD matrix computation
- Integration test suite for spatial + satellite pipeline (`tests/test_integration_spatial.py`)
  - Kriging raster loading and validation (4 variogram models, CRS checks)
  - PRESTO PC1 shapefile loading with column and CRS verification
  - Bivariate LISA on real PRESTO x AMBEL data (68% significant, HH=29.5%, LL=32.6%)
  - Spectral indices computation (9 indices) with vegetation and bare-soil signals
- `slow` pytest marker registered in pyproject.toml

### Changed
- Test suite expanded from 127 to 150 tests (148 pass, 2 skip)
- All integration tests use `@pytest.mark.skipif` for graceful degradation when external data is unavailable

## [0.1.0] - 2026-02-27

### Added
- 8 core modules: detection, embeddings, geostatistics, spatial, satellite, crossdomain, orthomosaic, viz
- 8 CLI entry points: ragweed-{train,sahi,embed,mmd,tile,kriging,lisa,indices}
- 27 CLI tool scripts in tools/ (organized by module)
- 6 example scripts demonstrating library usage
- 4 Jupyter notebook tutorials (synthetic data, self-contained)
- pytest test suite (8 test files covering all modules)
- Publication-quality visualization module (300 DPI)
- conda environment.yml for reproducible setup
- pyproject.toml with optional dependency groups
- MIT License
