# Contributing to Ragweed AI Toolkit

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/ragweed-ai-toolkit.git
cd ragweed-ai-toolkit

# 2. Install in development mode with all dependencies
pip install -e ".[all,dev]"

# 3. Verify installation
pytest tests/ -v
```

## Running Tests

```bash
# Full test suite
pytest tests/ -v

# Single module
pytest tests/test_embeddings.py -v

# With short tracebacks
pytest tests/ -v --tb=short
```

All tests use synthetic data and require no external files, GPUs, or network access.

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

- **Line length**: 100 characters
- **Target**: Python 3.10+
- **Style**: PEP 8 with Ruff defaults

Check your code before submitting:

```bash
# Lint
ruff check src/ tools/ examples/

# Auto-fix
ruff check --fix src/ tools/ examples/
```

## Pull Request Workflow

1. **Fork** the repository on GitHub.
2. **Create a branch** from `master`:
   ```bash
   git checkout -b feature/my-improvement
   ```
3. **Make your changes** and add tests if applicable.
4. **Run the test suite** to verify nothing is broken:
   ```bash
   pytest tests/ -v
   ```
5. **Run the linter**:
   ```bash
   ruff check src/ tools/ examples/
   ```
6. **Commit** with a clear message describing the change.
7. **Push** and open a Pull Request against `master`.

## Module Structure

The package is organized into 8 modules under `src/ragweed_toolkit/`:

| Module | Purpose |
|--------|---------|
| `detection` | YOLO training, SAHI inference, GPS linking, evaluation |
| `embeddings` | ResNet-50 feature extraction, MMD domain shift, PCA/UMAP/t-SNE |
| `crossdomain` | Multi-source dataset building, cross-domain evaluation |
| `orthomosaic` | GeoTIFF tiling for drone orthomosaics |
| `geostatistics` | Experimental variograms, model fitting, ordinary kriging |
| `spatial` | Moran's I, LISA clusters, GWR, cross-variograms |
| `satellite` | Sentinel-2 spectral indices, PRESTO embeddings, NDVI |
| `viz` | Publication-quality figures (matplotlib + plotly, 300 DPI) |

## Adding a New Module

1. Create the module directory under `src/ragweed_toolkit/<module_name>/`.
2. Add an `__init__.py` that exports the public API.
3. Add a test file at `tests/test_<module_name>.py` using synthetic data.
4. Add an example script at `examples/NN_<description>.py`.
5. If the module needs extra dependencies, add an optional group in `pyproject.toml`.
6. Update `CLAUDE.md` with import patterns and CLI details.

## Adding a CLI Tool

CLI tools live in `tools/` organized by module number. Each tool is a thin wrapper that:

1. Imports core logic from `ragweed_toolkit`.
2. Adds `argparse` for command-line arguments.
3. Prints formatted output.

See existing tools for the pattern. Place new tools in the appropriate subdirectory.

## Reporting Issues

Open an issue on GitHub with:

- A clear description of the problem or suggestion.
- Steps to reproduce (if reporting a bug).
- Python version and OS.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
