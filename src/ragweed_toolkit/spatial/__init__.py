"""
Spatial Statistics Module
=========================

Spatial autocorrelation analysis for identifying weed hot spots and evaluating
satellite-weed density relationships. Implements bivariate Moran's I, LISA
cluster decomposition, and Geographically Weighted Regression (GWR).

Key components:
    - morans: Global and bivariate Moran's I with permutation testing
    - lisa: Local Indicators of Spatial Association (HH/LL/HL/LH clusters)
    - gwr: Geographically Weighted Regression with adaptive bandwidth
    - cross_variograms: Cross-variogram computation and model fitting
"""

# cross_variograms only needs numpy + scipy (always available)
from .cross_variograms import (
    CrossVariogramResult,
    compute_cross_variogram,
    fit_cross_variogram_model,
)

# morans and lisa require esda, libpysal, geopandas
try:
    from .morans import (
        BivariateMoranResult,
        build_distance_weights,
        compute_bivariate_morans,
    )
    from .lisa import (
        LISA_COLORS,
        QUADRANT_MAP,
        compute_bivariate_lisa,
        lisa_summary,
    )
except ImportError:
    BivariateMoranResult = None  # type: ignore[assignment,misc]
    build_distance_weights = None  # type: ignore[assignment,misc]
    compute_bivariate_morans = None  # type: ignore[assignment,misc]
    LISA_COLORS = None  # type: ignore[assignment,misc]
    QUADRANT_MAP = None  # type: ignore[assignment,misc]
    compute_bivariate_lisa = None  # type: ignore[assignment,misc]
    lisa_summary = None  # type: ignore[assignment,misc]

# gwr requires mgwr, spreg (optional heavy dependencies)
try:
    from .gwr import (
        OLSResult,
        GWRResult,
        ModelComparison,
        fit_ols,
        fit_gwr,
        compare_ols_gwr,
    )
except ImportError:
    OLSResult = None  # type: ignore[assignment,misc]
    GWRResult = None  # type: ignore[assignment,misc]
    ModelComparison = None  # type: ignore[assignment,misc]
    fit_ols = None  # type: ignore[assignment,misc]
    fit_gwr = None  # type: ignore[assignment,misc]
    compare_ols_gwr = None  # type: ignore[assignment,misc]

__all__ = [
    # morans
    "BivariateMoranResult",
    "build_distance_weights",
    "compute_bivariate_morans",
    # lisa
    "LISA_COLORS",
    "QUADRANT_MAP",
    "compute_bivariate_lisa",
    "lisa_summary",
    # gwr
    "OLSResult",
    "GWRResult",
    "ModelComparison",
    "fit_ols",
    "fit_gwr",
    "compare_ols_gwr",
    # cross_variograms
    "CrossVariogramResult",
    "compute_cross_variogram",
    "fit_cross_variogram_model",
]
