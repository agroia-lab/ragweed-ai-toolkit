"""
Visualization Module
====================

Publication-quality figures for weed detection and spatial analysis results.
Generates maps, scatter plots, heatmaps, and comparison panels at 300 DPI.

Submodules
----------
style     : Shared colour palettes, fonts, and rcParams
maps      : Kriged density, LISA cluster maps, GWR coefficient maps
scatter   : UMAP and PCA scatter/biplot
heatmaps  : MMD pairwise matrices, correlation heatmaps
panels    : Detection comparison panels (GT vs predictions)

Quick usage::

    from ragweed_toolkit.viz import set_publication_style, plot_lisa_clusters
    set_publication_style()
    fig = plot_lisa_clusters(x, y, clusters, output_path="lisa.png")
"""

from __future__ import annotations

# Eagerly export style constants (lightweight, no heavy imports)
from .style import (
    DPI,
    FONT_FAMILY,
    FONT_SIZE,
    LISA_COLORS,
    LISA_LABELS,
    LISA_ORDER,
    MMD_THRESHOLDS,
    MMD_TIER_COLORS,
    ORARA_COLORS,
    ORARA_RANGES,
    SPECIES_COLORS,
    VARIOGRAM_COLORS,
    set_publication_style,
)

# Lazy imports: plotting functions are only loaded when accessed,
# keeping `import ragweed_toolkit.viz` fast.

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # maps
    "plot_kriged_density":       (".maps", "plot_kriged_density"),
    "plot_kriged_density_panels": (".maps", "plot_kriged_density_panels"),
    "plot_lisa_clusters":        (".maps", "plot_lisa_clusters"),
    "plot_lisa_panels":          (".maps", "plot_lisa_panels"),
    "plot_gwr_coefficients":     (".maps", "plot_gwr_coefficients"),
    # scatter
    "plot_umap":                 (".scatter", "plot_umap"),
    "plot_pca_biplot":           (".scatter", "plot_pca_biplot"),
    # heatmaps
    "plot_mmd_matrix":           (".heatmaps", "plot_mmd_matrix"),
    "plot_correlation_heatmap":  (".heatmaps", "plot_correlation_heatmap"),
    # panels
    "plot_detection_panel":      (".panels", "plot_detection_panel"),
    "plot_detection_grid":       (".panels", "plot_detection_grid"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # style
    "DPI",
    "FONT_FAMILY",
    "FONT_SIZE",
    "LISA_COLORS",
    "LISA_ORDER",
    "LISA_LABELS",
    "ORARA_COLORS",
    "ORARA_RANGES",
    "MMD_THRESHOLDS",
    "MMD_TIER_COLORS",
    "SPECIES_COLORS",
    "VARIOGRAM_COLORS",
    "set_publication_style",
    # maps
    "plot_kriged_density",
    "plot_kriged_density_panels",
    "plot_lisa_clusters",
    "plot_lisa_panels",
    "plot_gwr_coefficients",
    # scatter
    "plot_umap",
    "plot_pca_biplot",
    # heatmaps
    "plot_mmd_matrix",
    "plot_correlation_heatmap",
    # panels
    "plot_detection_panel",
    "plot_detection_grid",
]
