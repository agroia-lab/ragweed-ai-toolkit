# Architecture

## Design Principles

1. **Dual layout**: `src/ragweed_toolkit/` is the importable library; `tools/` contains CLI scripts that call the library. Both coexist.
2. **Optional dependencies**: Each module declares its extras in `pyproject.toml`. You only install what you need (`pip install -e ".[detection]"`).
3. **Lazy imports**: Sub-package `__init__.py` files use try/except so missing optional deps don't break unrelated modules.
4. **Chapter traceability**: Every function docstring references the chapter section and equation it implements.

## Module Dependency Graph

```
detection ─────────────────────────────────── (standalone)
embeddings ────────────────────────────────── (standalone)
crossdomain ───── depends on ── detection     (for evaluation)
orthomosaic ───── depends on ── detection     (for SAHI + GPS)
geostatistics ─────────────────────────────── (standalone)
spatial ───────── depends on ── geostatistics (for variogram models)
satellite ─────────────────────────────────── (standalone)
viz ───────────── depends on ── all above     (for figure generation)
```

## Data Flow

```
Raw images ──> detection (YOLO+SAHI) ──> point detections (GeoDataFrame)
                                              │
              ┌───────────────────────────────┤
              v                               v
    geostatistics (kriging)           embeddings (ResNet-50)
              │                               │
              v                               v
    density grid (5 m)                MMD deployment gate
              │                        (green/yellow/orange/red)
              v
    spatial (LISA + GWR)
              │
              v
    management zones (HH/LL/HL/LH)
```

## File Naming Conventions

- Library modules: `src/ragweed_toolkit/<package>/<module>.py`
- CLI scripts: `tools/<NN>_<package>/<script>.py`
- Tests: `tests/test_<package>_<module>.py`
- Examples: `examples/<use_case>.py`
