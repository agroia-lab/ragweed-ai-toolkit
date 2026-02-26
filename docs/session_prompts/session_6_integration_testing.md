# Ragweed AIToolkit — Session 6: Integration Testing with Real Data

## Context

This is Session 6 (final) of the ragweed-ai-toolkit multi-session build.

**Previous sessions:**
- Session 1: Package scaffold + 7 core modules
- Session 2: Viz module, unit tests, examples, CLAUDE.md
- Session 3: Refactored tools/ scripts to import from package
- Session 4: CLI entry points + notebook tutorials
- Session 5: GitHub + CI/CD + documentation site

**Repository:** `/home/malezainia2/dev/dlm6-weed-toolkit/`

Read `docs/SESSION_PLAN.md` for the full roadmap.

## Goal

Validate the full toolkit pipeline works end-to-end with real project data.
This is the final session — clean up, verify, and tag a stable release.

**IMPORTANT:** This session requires the external drive to be mounted at
`/media/malezainia2/E/` or similar. Check before starting.

## Task Breakdown

Use an agent team with 2 agents:

### Agent 1: Detection + Embedding Pipeline

1. **End-to-end detection test:**
   ```python
   from ragweed_toolkit.detection import SahiConfig, run_sahi_batch, extract_gps_geodataframe

   MODEL = "training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/weights/best.pt"
   IMAGES = "/media/malezainia2/E/ProcessingData/ambrosia-lentejas-seba-v1/test/images/"

   config = SahiConfig(slice_size=640, overlap=0.2, conf_threshold=0.25)
   results = run_sahi_batch(MODEL, image_paths[:10], config)
   gdf = extract_gps_geodataframe(image_paths[:10])
   gdf.to_file("test_output.gpkg", driver="GPKG")
   ```

2. **End-to-end embedding test:**
   ```python
   from ragweed_toolkit.embeddings import FeatureExtractor, extract_embeddings, compute_mmd, deployment_gate

   extractor = FeatureExtractor()
   embeddings_cl = extract_embeddings(extractor, cl_seba_images[:100])
   embeddings_intl = extract_embeddings(extractor, nd_individual_images[:100])

   mmd_value = compute_mmd(embeddings_cl, embeddings_intl)
   result = deployment_gate(mmd_value)
   print(f"CL vs ND_Individual: MMD={mmd_value:.3f}, tier={result.tier}")
   # Expected: orange tier (MMD ~0.30-0.43)
   ```

3. **Verify chapter Table values:**
   - Pairwise MMD matrix for all 9 databases
   - Compare with chapter's reported range (0.27-0.43)
   - UMAP visualization colored by database

4. **Write integration test:** `tests/test_integration_detection.py`

### Agent 2: Spatial + Satellite Pipeline

1. **End-to-end spatial test:**
   - Load Santa Rosa kriging rasters (from evidence data)
   - Load PRESTO embeddings
   - Run bivariate LISA (PC1 × AMBEL density)
   - Verify: ~68% significant pixels, HH ~29%, LL ~33%

2. **End-to-end GWR test:**
   - Load aligned grid (380 pixels)
   - Fit GWR with PC1+PC2+PC3 → AMBEL density
   - Verify: R² ≈ 0.882, bandwidth ≈ 49

3. **Spectral indices test:**
   - Load actual Sentinel-2 scene
   - Compute 9 indices
   - Compare NDVI correlation with chapter value (r = 0.837)

4. **Write integration test:** `tests/test_integration_spatial.py`

5. **Final cleanup:**
   - Remove any TODO comments
   - Ensure all docstrings are complete
   - Run full test suite
   - Version bump to v0.2.0

## Data Locations

| Data | Path |
|------|------|
| CL_Seba dataset | `/media/malezainia2/E/ProcessingData/ambrosia-lentejas-seba-v1/` |
| International dataset | `/media/malezainia2/E/ProcessingData/ambel-international-v1/` |
| Best model (Run 3) | `training_results/malezainia2/ambel_seba/run3_*/weights/best.pt` |
| ResNet50 embeddings | `/media/malezainia2/E/rageweed_international_databases/ragweed_embeddings/embeddings.npz` |
| PRESTO outputs | `outputs/lencu_presto/` |
| Kriging rasters | From evidence docs |
| Santa Rosa shapefile | `puntos lenteja v2.shp` |

## Verification Checklist

- [ ] `pip install -e ".[all]"` works
- [ ] All unit tests pass (`pytest tests/ -v`)
- [ ] All integration tests pass
- [ ] All CLI commands work (`ragweed-train --help`, etc.)
- [ ] All example scripts run
- [ ] Documentation site builds (`mkdocs build`)
- [ ] No TODO comments in production code
- [ ] CHANGELOG.md updated
- [ ] Version bumped to v0.2.0
- [ ] Tagged and pushed to GitHub

## Commit Message
```
Session 6: Integration tests with real data + v0.2.0 stable release
```
