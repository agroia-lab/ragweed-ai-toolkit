# Ragweed AIToolkit — Session 4: CLI Entry Points + Notebook Tutorials

## Context

This is Session 4 of the ragweed-ai-toolkit multi-session build.

**Previous sessions:**
- Session 1: Package scaffold + 7 core modules
- Session 2: Viz module, unit tests, examples, CLAUDE.md
- Session 3: Refactored 27 tools/ scripts to import from package

**Repository:** `/home/malezainia2/dev/dlm6-weed-toolkit/`

Read `docs/SESSION_PLAN.md` for the full roadmap.

## Goal

1. Add proper CLI entry points via pyproject.toml so the package provides
   command-line tools when installed
2. Create Jupyter notebook tutorials for each major workflow
3. Create conda environment.yml for reproducible setup

## Task Breakdown

Use an agent team with 2 agents:

### Agent 1: CLI Entry Points

Add `[project.scripts]` to pyproject.toml:

```toml
[project.scripts]
ragweed-train = "ragweed_toolkit.detection.trainer:main"
ragweed-sahi = "ragweed_toolkit.detection.inference:main"
ragweed-embed = "ragweed_toolkit.embeddings.extractor:main"
ragweed-mmd = "ragweed_toolkit.embeddings.mmd:main"
ragweed-tile = "ragweed_toolkit.orthomosaic.tiling:main"
ragweed-kriging = "ragweed_toolkit.geostatistics.kriging:main"
ragweed-lisa = "ragweed_toolkit.spatial.lisa:main"
ragweed-indices = "ragweed_toolkit.satellite.indices:main"
```

For each module, add a `main()` function at the bottom that:
1. Parses arguments with argparse
2. Calls the library functions
3. Prints results / saves output

Example pattern:
```python
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute MMD deployment gate")
    parser.add_argument("--source", required=True, help="Path to source embeddings .npz")
    parser.add_argument("--target", required=True, help="Path to target embeddings .npz")
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()

    source = np.load(args.source)["embeddings"]
    target = np.load(args.target)["embeddings"]

    mmd_value = compute_mmd(source, target)
    result = deployment_gate(mmd_value)

    print(f"MMD = {mmd_value:.4f}")
    print(f"Tier: {result.tier}")
    print(f"Recommendation: {result.recommendation}")

if __name__ == "__main__":
    main()
```

After adding all main() functions, reinstall and verify:
```bash
pip install -e ".[all]"
ragweed-train --help
ragweed-sahi --help
ragweed-embed --help
ragweed-mmd --help
ragweed-tile --help
ragweed-kriging --help
ragweed-lisa --help
ragweed-indices --help
```

### Agent 2: Notebook Tutorials + Environment

Create `notebooks/` with Jupyter notebooks:

1. **notebooks/01_detection_workflow.ipynb**
   - Configure and inspect TrainingConfig
   - Run SAHI inference on example images
   - Extract GPS, create GeoDataFrame
   - Export to shapefile
   - Plot detections on map

2. **notebooks/02_crossdomain_analysis.ipynb**
   - Extract ResNet-50 embeddings from multiple sources
   - Compute pairwise MMD matrix
   - Check deployment gates (all 4 tiers)
   - Visualize with UMAP scatter + MMD heatmap
   - Discuss implications for deployment

3. **notebooks/03_spatial_mapping.ipynb**
   - Fit semivariogram to point data
   - Run ordinary kriging → density grid
   - Compute bivariate LISA (satellite × weed density)
   - Generate cluster map (HH/LL/HL/LH)
   - Run GWR, compare with OLS

4. **notebooks/04_satellite_integration.ipynb**
   - Export Sentinel-2 composites (show API, don't actually run EE)
   - Compute spectral indices (9 indices from band array)
   - Extract PRESTO embeddings (show pattern)
   - Correlate with field observations
   - Create risk zones

Use synthetic data where possible. For Earth Engine parts, show the code
but wrap in `if False:` blocks with clear comments.

Also create `environment.yml`:
```yaml
name: ragweed-toolkit
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.10
  - pytorch>=2.0
  - torchvision>=0.15
  - numpy>=1.24
  - pandas>=2.0
  - geopandas>=0.14
  - matplotlib>=3.7
  - scipy>=1.11
  - scikit-learn>=1.3
  - rasterio>=1.3
  - jupyter
  - pip
  - pip:
    - ragweed-ai-toolkit[all]
```

## Commit Message
```
Session 4: Add CLI entry points (8 commands) + notebook tutorials (4 workflows)
```
