# Installation

## Requirements

- Python 3.10 or later
- pip (recommended) or conda

## Install from PyPI

```bash
# Core package (numpy, pandas, geopandas, matplotlib, scipy, scikit-learn)
pip install ragweed-ai-toolkit
```

## Install from Source

```bash
git clone https://github.com/agroia-lab/ragweed-ai-toolkit.git
cd ragweed-ai-toolkit
pip install -e .
```

## Optional Extras

Install only the modules you need:

```bash
# Detection: YOLO training + SAHI inference
pip install -e ".[detection]"      # + ultralytics, sahi, opencv

# Embeddings: ResNet-50 features, MMD, UMAP
pip install -e ".[embeddings]"     # + torch, torchvision, umap-learn

# Spatial: Moran's I, LISA, GWR
pip install -e ".[spatial]"        # + pygeoda, esda, libpysal, mgwr

# Satellite: Earth Engine, rasterio
pip install -e ".[satellite]"      # + earthengine-api, rasterio

# Orthomosaic: Drone mosaic tiling
pip install -e ".[orthomosaic]"    # + rasterio

# Visualization: Plotly, matplotlib-scalebar
pip install -e ".[viz]"            # + plotly, matplotlib-scalebar
```

## Install Everything

```bash
# All optional modules
pip install -e ".[all]"

# All modules + development tools (pytest, ruff, pre-commit)
pip install -e ".[all,dev]"
```

## Conda Environment

An `environment.yml` is provided for reproducible environments:

```bash
conda env create -f environment.yml
conda activate ragweed-toolkit
pip install -e ".[all]"
```

## Development Setup

```bash
git clone https://github.com/agroia-lab/ragweed-ai-toolkit.git
cd ragweed-ai-toolkit
pip install -e ".[all,dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/
```

## Verifying the Installation

After installing, verify the package loads correctly:

```python
import ragweed_toolkit
print(ragweed_toolkit.__version__)
# 0.1.0
```

Check that CLI commands are available:

```bash
ragweed-train --help
ragweed-sahi --help
ragweed-embed --help
ragweed-mmd --help
ragweed-tile --help
ragweed-kriging --help
ragweed-lisa --help
ragweed-indices --help
```
