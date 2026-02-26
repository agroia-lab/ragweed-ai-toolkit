"""
Satellite Module
================

Sentinel-2 and PRESTO foundation model integration for satellite-scale
weed density estimation. Includes Earth Engine export, NDVI time series,
and spectral index computation.

Key components:
    - composites: Sentinel-2 NDVI/RGB composite export via Earth Engine
    - presto: PRESTO foundation model embedding extraction
    - ndvi: NDVI time series statistics (min, max, std, slope)
    - indices: 9 spectral indices (NDVI, EVI, GNDVI, SWIRd, NBR2, SAVI, NDMI, BSI, NDWI)

Requires: pip install ragweed-ai-toolkit[satellite]
"""

__all__ = [
    # composites
    "get_sentinel2_collection",
    "mask_clouds_scl",
    "export_ndvi_max",
    "export_rgb_median",
    "export_composites",
    # ndvi
    "compute_ndvi_statistics",
    "compute_ndvi_slope",
    "compute_date_of_max",
    "export_ndvi_stats",
    "extract_raster_to_grid",
    "create_risk_zones",
    # indices
    "compute_spectral_indices",
    "compute_spectral_indices_ee",
    # presto
    "extract_presto_embeddings",
    "raster_to_grid",
    "apply_clustering",
    "apply_pca",
]

try:
    from ragweed_toolkit.satellite.composites import (
        export_composites,
        export_ndvi_max,
        export_rgb_median,
        get_sentinel2_collection,
        mask_clouds_scl,
    )
except ImportError:
    pass

try:
    from ragweed_toolkit.satellite.ndvi import (
        compute_date_of_max,
        compute_ndvi_slope,
        compute_ndvi_statistics,
        create_risk_zones,
        export_ndvi_stats,
        extract_raster_to_grid,
    )
except ImportError:
    pass

try:
    from ragweed_toolkit.satellite.indices import (
        compute_spectral_indices,
        compute_spectral_indices_ee,
    )
except ImportError:
    pass

try:
    from ragweed_toolkit.satellite.presto import (
        apply_clustering,
        apply_pca,
        extract_presto_embeddings,
        raster_to_grid,
    )
except ImportError:
    pass
