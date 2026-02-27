"""
Shared Style Constants
======================

Centralized color palettes, font settings, and matplotlib rcParams
for publication-quality figures (300 DPI, Arial/DejaVu Sans).

Usage::

    from ragweed_toolkit.viz.style import set_publication_style, LISA_COLORS

    set_publication_style()  # call once before plotting
"""

from __future__ import annotations

import matplotlib as mpl

# ---------------------------------------------------------------------------
# DPI and font defaults
# ---------------------------------------------------------------------------
DPI: int = 300
FONT_FAMILY: str = "Arial"
FONT_SIZE: int = 10
PANEL_LABEL_SIZE: int = 14

# ---------------------------------------------------------------------------
# LISA cluster colours (Local Moran's I)
# ---------------------------------------------------------------------------
LISA_COLORS: dict[str, str] = {
    "HH": "#e31a1c",  # red — hot spot
    "LL": "#1f78b4",  # blue — cold spot
    "HL": "#ff7f00",  # orange — high outlier
    "LH": "#6a3d9a",  # purple — low outlier
    "NS": "#cccccc",  # gray — not significant
}

# Ordered for legend rendering (significant first, NS last)
LISA_ORDER: list[str] = ["HH", "LL", "HL", "LH", "NS"]

LISA_LABELS: dict[str, str] = {
    "HH": "High-High (hot spot)",
    "LL": "Low-Low (cold spot)",
    "HL": "High-Low (outlier)",
    "LH": "Low-High (outlier)",
    "NS": "Not significant",
}

# ---------------------------------------------------------------------------
# Orobanche severity colours (nr_orara field)
# ---------------------------------------------------------------------------
ORARA_COLORS: dict[str, str] = {
    "Ausente":  "#CCCCCC",  # gray
    "Bajo":     "#2ECC71",  # green
    "Medio":    "#F1C40F",  # yellow
    "Alto":     "#E67E22",  # orange
    "Muy Alto": "#E74C3C",  # red
}

ORARA_RANGES: dict[str, tuple[int, int]] = {
    "Ausente":  (0, 0),
    "Bajo":     (1, 5),
    "Medio":    (6, 10),
    "Alto":     (11, 30),
    "Muy Alto": (31, 9999),
}

# ---------------------------------------------------------------------------
# MMD deployment-gate thresholds
# ---------------------------------------------------------------------------
MMD_THRESHOLDS: dict[str, float] = {
    "safe":     0.15,
    "caution":  0.30,
    "warning":  0.45,
}

MMD_TIER_COLORS: list[str] = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

# ---------------------------------------------------------------------------
# Species palette (chapter colour scheme)
# ---------------------------------------------------------------------------
SPECIES_COLORS: dict[str, str] = {
    "AMBEL": "#E74C3C",  # red
    "LENCU": "#2980B9",  # blue
    "POLAV": "#27AE60",  # green
    "POLPE": "#F39C12",  # orange
}

# ---------------------------------------------------------------------------
# Variogram fitting colours
# ---------------------------------------------------------------------------
VARIOGRAM_COLORS: dict[str, str] = {
    "cross":    "#8E44AD",  # purple
    "range":    "#C0392B",  # dark red
    "sill":     "#7F8C8D",  # gray
}

# ---------------------------------------------------------------------------
# Dark background (LISA-style maps)
# ---------------------------------------------------------------------------
DARK_BG: str = "#2C3E50"
DARK_FG: str = "white"
DARK_FACE: str = "#1A252F"
DARK_LEGEND_BG: str = "#34495E"

# ---------------------------------------------------------------------------
# Boundary style
# ---------------------------------------------------------------------------
BOUNDARY_COLOR: str = "white"
BOUNDARY_LW: float = 1.5
BOUNDARY_LS: str = "--"


def set_publication_style() -> None:
    """Configure matplotlib rcParams for publication-quality output.

    Call once at the start of a script or notebook.  Sets Arial font,
    300 DPI, and sensible tick / label sizes.
    """
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "DejaVu Sans", "Helvetica"],
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 1,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
