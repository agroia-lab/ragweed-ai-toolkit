"""
Example 06: Sentinel-2 Spectral Indices
=========================================

Computes all 9 spectral indices from a synthetic 10-band Sentinel-2
array using the ragweed_toolkit.satellite.indices module.

Band order: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12

The indices characterize vegetation, soil, and moisture at each pixel:
    NDVI, EVI, GNDVI, RENDVI, S2WI, NBR2, BSI, Clay, SWIRd
"""

import numpy as np

from ragweed_toolkit.satellite.indices import (
    BAND_NAMES,
    INDEX_DESCRIPTIONS,
    compute_spectral_indices,
)

np.random.seed(42)

# ------------------------------------------------------------------ #
# 1.  Create a synthetic 10-band Sentinel-2 "image"
# ------------------------------------------------------------------ #
H, W = 64, 64  # small tile for demonstration

# Reflectance ranges loosely based on Sentinel-2 values
band_ranges = {
    "B2":  (0.02, 0.10),   # Blue
    "B3":  (0.03, 0.12),   # Green
    "B4":  (0.02, 0.08),   # Red
    "B5":  (0.03, 0.15),   # Red Edge 1
    "B6":  (0.10, 0.30),   # Red Edge 2
    "B7":  (0.15, 0.40),   # Red Edge 3
    "B8":  (0.15, 0.50),   # NIR
    "B8A": (0.15, 0.45),   # Narrow NIR
    "B11": (0.05, 0.30),   # SWIR1
    "B12": (0.03, 0.20),   # SWIR2
}

spectral = np.zeros((10, H, W), dtype=np.float32)
for i, name in enumerate(BAND_NAMES):
    lo, hi = band_ranges[name]
    spectral[i] = np.random.uniform(lo, hi, (H, W))

print(f"=== Synthetic Sentinel-2 Image ===")
print(f"  Shape : {spectral.shape} (bands, H, W)")
print(f"  Bands : {', '.join(BAND_NAMES)}")

# ------------------------------------------------------------------ #
# 2.  Compute all 9 indices
# ------------------------------------------------------------------ #
indices = compute_spectral_indices(spectral)

print(f"\n=== Computed Indices ({len(indices)} total) ===")
print(f"  {'Index':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}  Description")
print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*40}")

for name, arr in indices.items():
    desc = INDEX_DESCRIPTIONS[name]
    print(f"  {name:<8} {arr.mean():>8.3f} {arr.std():>8.3f} "
          f"{arr.min():>8.3f} {arr.max():>8.3f}  {desc}")

# ------------------------------------------------------------------ #
# 3.  Compute a subset of indices
# ------------------------------------------------------------------ #
subset = compute_spectral_indices(spectral, indices=["NDVI", "BSI", "S2WI"])
print(f"\n=== Subset (3 indices) ===")
for name, arr in subset.items():
    print(f"  {name:<8} : shape={arr.shape}, mean={arr.mean():.3f}")

# ------------------------------------------------------------------ #
# 4.  Apply a validity mask (e.g., exclude clouds or no-data pixels)
# ------------------------------------------------------------------ #
valid_mask = np.ones((H, W), dtype=bool)
valid_mask[0:10, :] = False  # top 10 rows are "cloud"

masked = compute_spectral_indices(spectral, valid=valid_mask, indices=["NDVI"])
ndvi = masked["NDVI"]
n_nan = np.isnan(ndvi).sum()
print(f"\n=== With validity mask ===")
print(f"  NaN pixels  : {n_nan} (masked as cloud)")
print(f"  Valid pixels : {H * W - n_nan}")
print(f"  NDVI mean (valid only) : {np.nanmean(ndvi):.3f}")
