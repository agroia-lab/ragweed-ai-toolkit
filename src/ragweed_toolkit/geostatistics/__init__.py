"""
Geostatistics Module
====================

Ordinary kriging interpolation and semivariogram analysis for generating
continuous density surfaces from point-based weed detections.

Key components:
    - variograms: Experimental and model semivariogram fitting (exponential)
    - kriging: Ordinary kriging at configurable grid resolution
"""

from .variograms import (
    VariogramModel,
    compute_experimental_variogram,
    fit_exponential_model,
    plot_variogram,
)

# kriging requires geopandas -- defer import so pure-numpy variogram code
# remains usable even when geopandas is not installed.
try:
    from .kriging import ordinary_kriging
except ImportError:
    ordinary_kriging = None  # type: ignore[assignment,misc]

__all__ = [
    "VariogramModel",
    "compute_experimental_variogram",
    "fit_exponential_model",
    "plot_variogram",
    "ordinary_kriging",
]
