"""
Spectral Indices
=================

Compute spectral vegetation, soil, and moisture indices from Sentinel-2
bands. Supports both raster arrays (numpy) and Earth Engine images.

Indices
-------
NDVI : Normalized Difference Vegetation Index
EVI : Enhanced Vegetation Index
GNDVI : Green NDVI (chlorophyll content)
RENDVI : Red Edge NDVI (phenological state)
S2WI : Sentinel-2 Water Index (soil moisture)
NBR2 : Normalized Burn Ratio 2 (moisture/residue)
BSI : Bare Soil Index
Clay : Clay minerals index (SWIR1/SWIR2)
SWIRd : SWIR difference (texture proxy)

Functions
---------
compute_spectral_indices
    Compute all 9 indices from a 10-band Sentinel-2 array.
compute_spectral_indices_ee
    Compute indices as Earth Engine bands for server-side processing.
"""

from typing import Dict, List, Optional

import numpy as np

# Sentinel-2 band mapping (10 bands used for indices)
BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
BAND_INDEX = {name: i for i, name in enumerate(BAND_NAMES)}

# Index definitions for reference
INDEX_DESCRIPTIONS = {
    "NDVI": "Normalized Difference Vegetation Index — vegetation density",
    "EVI": "Enhanced Vegetation Index — atmosphere-corrected vegetation",
    "GNDVI": "Green NDVI — chlorophyll content",
    "RENDVI": "Red Edge NDVI — phenological state",
    "S2WI": "Sentinel-2 Water Index — soil moisture (SWIR1/NIR)",
    "NBR2": "Normalized Burn Ratio 2 — moisture and crop residue",
    "BSI": "Bare Soil Index — exposed soil fraction",
    "Clay": "Clay minerals index — SWIR1/SWIR2 ratio",
    "SWIRd": "SWIR difference — surface texture proxy",
}

# Epsilon to avoid division by zero
_EPS = 1e-9


def compute_spectral_indices(
    spectral: np.ndarray,
    valid: Optional[np.ndarray] = None,
    *,
    indices: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Compute spectral indices from a Sentinel-2 band stack.

    Args:
        spectral: Array of shape ``(10, H, W)`` with bands in order:
            B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12.
        valid: Boolean mask of shape ``(H, W)``. Invalid pixels are set
            to ``NaN``. If ``None``, all pixels are treated as valid.
        indices: List of index names to compute. If ``None``, computes
            all 9 indices.

    Returns:
        Dict mapping index name to ``(H, W)`` float array.
    """
    if spectral.shape[0] != 10:
        raise ValueError(
            f"Expected 10-band array (B2-B12), got {spectral.shape[0]} bands"
        )

    B2 = spectral[0].astype(float)   # Blue
    B3 = spectral[1].astype(float)   # Green
    B4 = spectral[2].astype(float)   # Red
    B5 = spectral[3].astype(float)   # Red Edge 1
    B8 = spectral[6].astype(float)   # NIR
    B8A = spectral[7].astype(float)  # Narrow NIR
    B11 = spectral[8].astype(float)  # SWIR1
    B12 = spectral[9].astype(float)  # SWIR2

    all_indices = indices or list(INDEX_DESCRIPTIONS.keys())
    result: Dict[str, np.ndarray] = {}

    for name in all_indices:
        if name == "NDVI":
            result["NDVI"] = (B8 - B4) / (B8 + B4 + _EPS)
        elif name == "EVI":
            result["EVI"] = 2.5 * (B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1 + _EPS)
        elif name == "GNDVI":
            result["GNDVI"] = (B8 - B3) / (B8 + B3 + _EPS)
        elif name == "RENDVI":
            result["RENDVI"] = (B8A - B5) / (B8A + B5 + _EPS)
        elif name == "S2WI":
            result["S2WI"] = (B8 - B11) / (B8 + B11 + _EPS)
        elif name == "NBR2":
            result["NBR2"] = (B11 - B12) / (B11 + B12 + _EPS)
        elif name == "BSI":
            result["BSI"] = (
                (B11 + B4) - (B8 + B2)
            ) / ((B11 + B4) + (B8 + B2) + _EPS)
        elif name == "Clay":
            result["Clay"] = B11 / (B12 + _EPS)
        elif name == "SWIRd":
            result["SWIRd"] = B11 - B12
        else:
            raise ValueError(f"Unknown index: {name!r}")

    if valid is not None:
        for arr in result.values():
            arr[~valid] = np.nan

    return result


ALL_INDEX_NAMES = list(INDEX_DESCRIPTIONS.keys())


def compute_spectral_indices_ee(image):
    """Compute spectral indices as Earth Engine bands.

    Args:
        image: ``ee.Image`` with Sentinel-2 bands (B2-B12).

    Returns:
        ``ee.Image`` with one band per index.
    """
    import ee

    B2 = image.select("B2").toFloat()
    B3 = image.select("B3").toFloat()
    B4 = image.select("B4").toFloat()
    B5 = image.select("B5").toFloat()
    B8 = image.select("B8").toFloat()
    B8A = image.select("B8A").toFloat()
    B11 = image.select("B11").toFloat()
    B12 = image.select("B12").toFloat()

    ndvi = B8.subtract(B4).divide(B8.add(B4)).rename("NDVI")
    evi = (
        B8.subtract(B4)
        .multiply(2.5)
        .divide(B8.add(B4.multiply(6)).subtract(B2.multiply(7.5)).add(1))
        .rename("EVI")
    )
    gndvi = B8.subtract(B3).divide(B8.add(B3)).rename("GNDVI")
    rendvi = B8A.subtract(B5).divide(B8A.add(B5)).rename("RENDVI")
    s2wi = B8.subtract(B11).divide(B8.add(B11)).rename("S2WI")
    nbr2 = B11.subtract(B12).divide(B11.add(B12)).rename("NBR2")
    bsi = (
        B11.add(B4).subtract(B8.add(B2))
        .divide(B11.add(B4).add(B8).add(B2))
        .rename("BSI")
    )
    clay = B11.divide(B12).rename("Clay")
    swird = B11.subtract(B12).rename("SWIRd")

    return (
        ndvi.addBands(evi).addBands(gndvi).addBands(rendvi)
        .addBands(s2wi).addBands(nbr2).addBands(bsi)
        .addBands(clay).addBands(swird)
    )


def main():
    """CLI entry point for computing spectral indices from a 10-band GeoTIFF."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute spectral indices from a 10-band Sentinel-2 GeoTIFF."
    )
    parser.add_argument("--input", required=True, help="Input 10-band GeoTIFF (B2-B12)")
    parser.add_argument("--output-dir", required=True, help="Output directory for index GeoTIFFs")
    parser.add_argument("--indices", nargs="+", default=None,
                        choices=ALL_INDEX_NAMES,
                        help=f"Indices to compute (default: all). Choices: {', '.join(ALL_INDEX_NAMES)}")
    args = parser.parse_args()

    import rasterio
    from pathlib import Path

    with rasterio.open(args.input) as src:
        spectral = src.read()
        profile = src.profile.copy()
        print(f"Input: {args.input}")
        print(f"  Shape: {spectral.shape}, CRS: {src.crs}")

    if spectral.shape[0] != 10:
        print(f"Error: Expected 10 bands, got {spectral.shape[0]}")
        return

    indices_to_compute = args.indices or ALL_INDEX_NAMES
    results = compute_spectral_indices(spectral, indices=indices_to_compute)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profile.update(count=1, dtype="float32")

    for name, arr in results.items():
        out_path = out_dir / f"{name}.tif"
        with rasterio.open(str(out_path), "w", **profile) as dst:
            dst.write(arr.astype(np.float32), 1)
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        print(f"  {name}: [{vmin:.4f}, {vmax:.4f}] -> {out_path.name}")

    print(f"\n{len(results)} indices saved to {out_dir}")


if __name__ == "__main__":
    main()
