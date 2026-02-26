# Ragweed AIToolkit — Session 3: Refactor tools/ to Import from Package

## Context

This is Session 3 of the ragweed-ai-toolkit multi-session build.

**Previous sessions:**
- Session 1: Created package scaffold + 7 core modules (24 files, 6K lines)
- Session 2: Added viz module, unit tests, examples, updated CLAUDE.md

**Repository:** `/home/malezainia2/dev/dlm6-weed-toolkit/`

Read `docs/SESSION_PLAN.md` for the full roadmap.

## Goal

Refactor the 27 existing CLI scripts in `tools/` so they import core logic
from `ragweed_toolkit` instead of being monolithic. This validates the library
API works for real workflows.

## Task Breakdown

Use an agent team with 3 agents working in parallel:

### Agent 1: Detection + Embedding scripts (7 scripts)

Refactor these to import from the package:

```
tools/01_detection/train_yolo_cli.py      → from ragweed_toolkit.detection import TrainingConfig, train
tools/01_detection/sahi_inference_cli.py   → from ragweed_toolkit.detection import SahiConfig, run_sahi_batch
tools/02_embedding/generate_embeddings.py  → from ragweed_toolkit.embeddings import FeatureExtractor, extract_embeddings
tools/02_embedding/generate_embeddings_multi.py → same
tools/02_embedding/compute_embeddings.py   → from ragweed_toolkit.embeddings import ...
tools/02_embedding/embed_ortho_tiles.py    → from ragweed_toolkit.embeddings import ...
tools/02_embedding/compute_ortho_domain_shift.py → from ragweed_toolkit.embeddings import compute_mmd, deployment_gate
```

**Pattern:** Keep argparse CLI in the script, but replace the internal implementation
with calls to the library. The script becomes a thin CLI wrapper.

### Agent 2: Crossdomain + Orthomosaic scripts (7 scripts)

```
tools/03_crossdomain/build_international_dataset.py → from ragweed_toolkit.crossdomain import build_international_dataset
tools/03_crossdomain/build_combined_dataset.py      → from ragweed_toolkit.crossdomain import build_combined_dataset
tools/03_crossdomain/crossdomain_eval.py            → from ragweed_toolkit.crossdomain import run_crossdomain_evaluation
tools/03_crossdomain/crossdomain_compare.py         → from ragweed_toolkit.crossdomain import compare_evaluations
tools/04_orthomosaic/tile_orthomosaic.py             → from ragweed_toolkit.orthomosaic import tile_orthomosaic
tools/04_orthomosaic/detections_to_geopackage.py     → from ragweed_toolkit.orthomosaic import detections_to_geopackage
tools/04_orthomosaic/mine_uncertain_samples.py       → from ragweed_toolkit.orthomosaic import score_images, export_uncertain_panels
```

### Agent 3: Geostatistics + Satellite scripts (13 scripts)

```
tools/05_geostatistics/spatial_cluster_analysis.py   → from ragweed_toolkit.spatial import compute_bivariate_lisa
tools/05_geostatistics/bivariate_lisa_soil.py        → from ragweed_toolkit.spatial import compute_bivariate_morans, compute_bivariate_lisa
tools/06_satellite/export_satellite_composites.py    → from ragweed_toolkit.satellite import export_composites
tools/06_satellite/extract_embeddings.py             → from ragweed_toolkit.satellite import ...
tools/06_satellite/extract_ndvi_timeseries.py        → from ragweed_toolkit.satellite import compute_ndvi_statistics
tools/06_satellite/extract_presto_embeddings.py      → from ragweed_toolkit.satellite import extract_presto_embeddings
tools/06_satellite/extract_presto_lencu_paddock.py   → same
tools/06_satellite/extract_presto_custom_windows.py  → same
tools/06_satellite/compare_presto_windows.py         → same
tools/06_satellite/map_presto_windows.py             → from ragweed_toolkit.viz import ...
tools/06_satellite/gwr_lencu.py                      → from ragweed_toolkit.spatial import fit_gwr, compare_ols_gwr
tools/06_satellite/spatial_correlation_lencu.py       → from ragweed_toolkit.spatial import compute_bivariate_morans
tools/06_satellite/analyze_sentinel_weed_density.py  → from ragweed_toolkit.satellite import ...
```

## Refactoring Rules

1. **Keep CLI interface identical** — same argparse arguments, same output format
2. **Replace internal functions** with library imports
3. **Keep script-specific logic** (file path handling, logging, progress bars) in the script
4. **Don't force it** — if a script has complex logic not yet in the library, add a TODO comment and leave it
5. **Test with --help** — every refactored script must still parse arguments correctly

## Verification

After all refactoring:
```bash
for f in tools/*/*.py; do
    python "$f" --help > /dev/null 2>&1 && echo "OK: $f" || echo "FAIL: $f"
done
```

## Commit Message
```
Session 3: Refactor tools/ CLI scripts to import from ragweed_toolkit package
```
