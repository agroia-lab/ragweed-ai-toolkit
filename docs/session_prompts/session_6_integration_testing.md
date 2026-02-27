# Ragweed AIToolkit — Session 6: Integration Testing with Real Data

## Context

This is Session 6 (final) of the ragweed-ai-toolkit multi-session build.

**Previous sessions (all DONE):**
- Session 1: Package scaffold + 8 core modules (38 source files, 6K lines)
- Session 2: Viz module, pytest test suite (8 test files), 6 example scripts, CLAUDE.md
- Session 3: Refactored 27 tools/ CLI scripts to import from ragweed_toolkit
- Session 4: 8 CLI entry points via pyproject.toml, 4 Jupyter notebook tutorials, environment.yml
- Session 5: GitHub repo (agroia-lab/ragweed-ai-toolkit), 3 CI workflows (test/lint/docs), mkdocs site, CONTRIBUTING.md, CHANGELOG.md, v0.1.0 tag

**Toolkit location:** `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/research_docs/lencu_book_chapter/toolkit/`

**GitHub repo:** https://github.com/agroia-lab/ragweed-ai-toolkit
**Docs site:** https://agroia-lab.github.io/ragweed-ai-toolkit

**Important — Dual git repos:**
The toolkit has its own `.git` at the toolkit root (6 commits, remote: agroia-lab/ragweed-ai-toolkit).
It is ALSO tracked as a subdirectory inside the parent INIA project.
For Session 6, work ONLY in the toolkit's own git repo.

Read `docs/SESSION_PLAN.md` and `CLAUDE.md` for the full roadmap and package structure.

## Goal

Validate the full toolkit pipeline works end-to-end with **real project data** from the
external drive. This is the final session — clean up, verify, tag v0.2.0 stable release.

**IMPORTANT:** This session requires the external drive mounted at `/media/malezainia2/E/`.
Check before starting:
```bash
ls /media/malezainia2/E/ProcessingData/ && echo "Drive OK"
```

## Machine & Environment

| Property | Value |
|----------|-------|
| Machine | malezainia2 |
| GPU | 2x NVIDIA RTX 4090 (24 GB VRAM each) |
| Preferred GPU | cuda:1 |
| Conda env | `base` (has torch, torchvision, umap-learn, geopandas, plotly) |
| Python | `/home/malezainia2/anaconda3/bin/python` |

## Prerequisites

Before starting:
```bash
cd /home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/research_docs/lencu_book_chapter/toolkit

# Verify drive is mounted
ls /media/malezainia2/E/ProcessingData/ambrosia-lentejas-seba-v1/

# Install package in dev mode
pip install -e ".[all,dev]"

# Verify tests pass
pytest tests/ -v --tb=short
```

## Task Breakdown

Use an agent team with 2 agents:

### Agent 1: Detection + Embedding Pipeline

#### Step 1: End-to-end detection test

Test the full pipeline: load images → SAHI inference → GPS extraction → shapefile export.

```python
from ragweed_toolkit.detection import SahiConfig, run_sahi_batch, extract_gps_geodataframe
from pathlib import Path

MODEL = "/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/weights/best.pt"
IMAGE_DIR = Path("/media/malezainia2/E/ProcessingData/ambrosia-lentejas-seba-v1/test/images/")
image_paths = sorted(IMAGE_DIR.glob("*.jpg"))[:10]  # Test with 10 images

# SAHI inference
config = SahiConfig(slice_size=640, overlap=0.2, conf_threshold=0.25)
results = run_sahi_batch(str(MODEL), [str(p) for p in image_paths], config)
print(f"Detections: {sum(len(r) for r in results)}")

# GPS extraction → shapefile
gdf = extract_gps_geodataframe([str(p) for p in image_paths])
gdf.to_file("/tmp/test_detections.gpkg", driver="GPKG")
print(f"GPS points: {len(gdf)}")
```

**Expected:** Detections found, GPS coordinates extracted, valid GeoPackage written.

#### Step 2: End-to-end embedding test

Test embeddings extraction + MMD domain shift + deployment gate.

```python
from ragweed_toolkit.embeddings import FeatureExtractor, extract_embeddings, compute_mmd, deployment_gate
from pathlib import Path

extractor = FeatureExtractor(device="cuda:1")

# Source: Chile (CL_Seba)
cl_dir = Path("/media/malezainia2/E/ProcessingData/ambrosia-lentejas-seba-v1/test/images/")
cl_images = sorted(cl_dir.glob("*.jpg"))[:100]

# Target: International (ND_Individual)
intl_dir = Path("/media/malezainia2/E/ProcessingData/ambel-international-v1/")
# Find ND_Individual images within the international dataset
nd_images = sorted(intl_dir.rglob("*.jpg"))[:100]

embeddings_cl = extract_embeddings(extractor, [str(p) for p in cl_images])
embeddings_nd = extract_embeddings(extractor, [str(p) for p in nd_images])

mmd_value = compute_mmd(embeddings_cl, embeddings_nd)
result = deployment_gate(mmd_value)
print(f"CL vs ND: MMD={mmd_value:.3f}, tier={result.tier}, {result.recommendation}")
```

**Expected:** MMD in range 0.27–0.43 (orange tier — domain shift detected).

#### Step 3: Verify pre-computed embeddings

Load the pre-computed 9-database embeddings and verify MMD matrix matches chapter values.

```python
import numpy as np

# Pre-computed embeddings from the 9-database analysis
npz = np.load("/media/malezainia2/E/rageweed_international_databases/ragweed_embeddings/embeddings.npz")
print(f"Keys: {list(npz.keys())}")
print(f"Shape: {npz[list(npz.keys())[0]].shape}")

# Compute pairwise MMD matrix between databases
# Verify range matches chapter: 0.27–0.43
```

#### Step 4: Write integration test

Create `tests/test_integration_detection.py`:
- Uses `@pytest.mark.skipif` for when external drive is not mounted
- Tests detection pipeline with 5 images
- Tests embedding extraction with 20 images
- Tests MMD computation with small batches
- Tests deployment gate output structure

### Agent 2: Spatial + Satellite Pipeline

#### Step 1: End-to-end kriging test

Load kriging rasters and verify they load correctly through the toolkit.

```python
import rasterio

KRIGING_DIR = "/media/malezainia2/E/lencu_santarosa_full/kriging/"
raster_path = KRIGING_DIR + "kriging_total_det_exponential_5m.tif"

with rasterio.open(raster_path) as src:
    data = src.read(1)
    print(f"Shape: {data.shape}, CRS: {src.crs}")
    print(f"Min: {data.min():.2f}, Max: {data.max():.2f}, Mean: {data.mean():.2f}")
```

**Expected:** Valid raster in EPSG:32719, values representing weed density counts.

#### Step 2: End-to-end PRESTO + LISA test

Load PRESTO PC1 shapefile and test bivariate LISA with weed density.

```python
import geopandas as gpd
from ragweed_toolkit.spatial import compute_bivariate_lisa, lisa_summary

# PRESTO PC1 shapefile (424 pixels, Santa Rosa paddock)
gdf = gpd.read_file("/home/malezainia2/Downloads/presto_PC1.shp")
print(f"Columns: {list(gdf.columns)}")
print(f"Rows: {len(gdf)}")

# Check what columns contain PC1 and weed density
# Then run bivariate LISA
# gdf_lisa = compute_bivariate_lisa(gdf, var_x="PC1", var_y="AMBEL_density",
#                                    threshold=15.0, permutations=999)
# print(lisa_summary(gdf_lisa))
```

**Expected reference values (from chapter):**
- Bivariate Moran's I (PC1 × AMBEL): 0.706 (p=0.001)
- LISA significant pixels: ~68%
- HH (Hot Spot): ~29%
- LL (Cold Spot): ~33%

#### Step 3: Spectral indices test

Test the 9-index computation from actual Sentinel-2 data (or synthetic 10-band array
matching Sentinel-2 band order if no raw scene is on this machine).

```python
import numpy as np
from ragweed_toolkit.satellite import compute_spectral_indices

# Create synthetic Sentinel-2 bands matching real value ranges
# Band order: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
bands = np.random.uniform(0.01, 0.5, size=(10, 100, 100)).astype(np.float32)
bands[6] = 0.35  # B8 (NIR) higher for vegetation
bands[2] = 0.05  # B4 (Red) lower for vegetation

indices = compute_spectral_indices(bands)
print(f"Indices computed: {list(indices.keys())}")
# Should include: NDVI, EVI, GNDVI, SWIRd, NBR2, SAVI, NDMI, BSI, NDWI
```

#### Step 4: Write integration test

Create `tests/test_integration_spatial.py`:
- Uses `@pytest.mark.skipif` for when external drive / PRESTO data is not available
- Tests kriging raster loading
- Tests PRESTO shapefile loading + bivariate LISA
- Tests spectral index computation (synthetic data — always runs)

#### Step 5: Final cleanup and release

1. **Run full test suite:** `pytest tests/ -v`
2. **Run lint:** `ruff check src/ tools/ examples/`
3. **Review TODO comments:** `grep -r "TODO" src/ --include="*.py"` — remove or resolve
4. **Update CHANGELOG.md:** Add v0.2.0 section
5. **Version bump:** Update `version = "0.2.0"` in `pyproject.toml`
6. **Update docs:** `mkdocs build` to verify
7. **Commit, push, tag:**
   ```bash
   git add -A
   git commit -m "Session 6: Integration tests with real data + v0.2.0 stable release"
   git push origin master
   git tag -a v0.2.0 -m "Stable release: integration-tested with real data"
   git push origin v0.2.0
   ```

## Data Locations (Verified on malezainia2)

| Data | Path | Size |
|------|------|------|
| CL_Seba YOLO dataset | `/media/malezainia2/E/ProcessingData/ambrosia-lentejas-seba-v1/` | YOLO format (train/valid/test) |
| International dataset | `/media/malezainia2/E/ProcessingData/ambel-international-v1/` | 732K images |
| Combined dataset | `/media/malezainia2/E/ProcessingData/ambel-combined-intl-v1/` | Combined CL+INTL |
| Alberto dataset | `/media/malezainia2/E/ProcessingData/ambrosia.dataset_alberto/` | 659 images |
| 9-database embeddings | `/media/malezainia2/E/rageweed_international_databases/ragweed_embeddings/embeddings.npz` | 20 MB |
| Ortho embeddings | `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_embeddings/embeddings.npz` | 5.3 MB |
| Kriging rasters (4 models) | `/media/malezainia2/E/lencu_santarosa_full/kriging/*.tif` | Exponential, Gaussian, Linear, Spherical |
| PRESTO PC1 shapefile | `/home/malezainia2/Downloads/presto_PC1.shp` | 424 pixels, EPSG:32719 |
| Best model (Run 3) | `training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/weights/best.pt` | 49 MB |
| Combined model | `training_results/malezainia2/ambel_crossdomain/combined_intl_v1_20260222_161204/weights/best.pt` | 49 MB |

## Key Results to Verify (from Chapter Evidence)

| Metric | Expected Value | Source |
|--------|---------------|--------|
| Best mAP50 (Run 3) | 0.886 | Training progression |
| Cross-domain collapse CL→INTL | mAP50 = 0.108 | Cross-domain Phase 1 |
| Multi-domain recovery | mAP50 = 0.874 | Cross-domain Phase 2 |
| MMD range (9 databases) | 0.27–0.43 | Embedding analysis |
| PRESTO PC1 × AMBEL correlation | r = 0.739 | Satellite analysis |
| PRESTO PC1 × LENCU correlation | r = 0.717 | Satellite analysis |
| NDVI × AMBEL (single-date) | r = 0.837 | Satellite analysis |
| Bivariate Moran's I (PC1 × AMBEL) | 0.706 (p=0.001) | LISA 999 permutations |
| GWR R² AMBEL | 0.882 (BW=49nn) | GWR 3-PC model |
| GWR R² LENCU | 0.910 (BW=49nn) | GWR 3-PC model |
| LISA significant pixels | ~68% | PC1 × AMBEL |
| HH (Hot Spot) | ~29% | LISA clusters |
| LL (Cold Spot) | ~33% | LISA clusters |

## Verification Checklist

- [ ] External drive mounted at `/media/malezainia2/E/`
- [ ] `pip install -e ".[all,dev]"` works
- [ ] All unit tests pass (`pytest tests/ -v`)
- [ ] Detection integration test passes (SAHI + GPS + shapefile)
- [ ] Embedding integration test passes (extract + MMD + gate)
- [ ] Kriging raster loads correctly
- [ ] PRESTO shapefile loads correctly
- [ ] Bivariate LISA runs on real data
- [ ] Spectral indices compute correctly
- [ ] All integration tests pass
- [ ] All CLI commands work (`ragweed-{train,sahi,embed,mmd,tile,kriging,lisa,indices} --help`)
- [ ] Documentation site builds (`mkdocs build`)
- [ ] Lint passes (`ruff check src/ tools/ examples/`)
- [ ] No critical TODO comments in src/
- [ ] CHANGELOG.md updated for v0.2.0
- [ ] Version bumped to 0.2.0 in pyproject.toml
- [ ] Tagged v0.2.0 and pushed to GitHub
- [ ] CI passes (all 3 workflows green)

## Important Notes

- Use `cuda:1` as preferred GPU device
- Integration tests should use `@pytest.mark.skipif` for data that requires the external drive
- Keep tests fast: use small subsets (5–20 images) rather than full datasets
- The satellite analysis (GWR, NDVI correlations) was originally run on malezainia1 — some data may need to be adapted. Work with what's available on malezainia2.
- PRESTO PC1 shapefile is in `/home/malezainia2/Downloads/` — copy it to the toolkit's `data/` directory for reproducibility

## Commit Message
```
Session 6: Integration tests with real data + v0.2.0 stable release
```
