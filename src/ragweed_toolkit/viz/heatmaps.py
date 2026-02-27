"""
Matrix Visualizations (Heatmaps)
================================

Publication-quality heatmap plots for pairwise distances and correlations:

- :func:`plot_mmd_matrix` -- Pairwise MMD heatmap with deployment-gate tiers
- :func:`plot_correlation_heatmap` -- Feature-vs-target correlation matrix

All functions return a :class:`matplotlib.figure.Figure` and optionally
save to file at 300 DPI.

Usage::

    from ragweed_toolkit.viz.heatmaps import plot_mmd_matrix
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

from .style import (
    DPI,
    FONT_SIZE,
    MMD_THRESHOLDS,
    MMD_TIER_COLORS,
    set_publication_style,
)


def plot_mmd_matrix(
    mmd_matrix: np.ndarray,
    labels: Sequence[str],
    *,
    title: str = "Pairwise MMD Matrix",
    annotate: bool = True,
    threshold_lines: bool = True,
    cmap: str = "YlOrRd",
    figsize: tuple = (10, 8),
    output_path: Optional[Union[str, Path]] = None,
) -> matplotlib.figure.Figure:
    """Create a pairwise MMD heatmap with deployment-gate colour bands.

    When *threshold_lines* is True, cells are colour-coded using four tiers:
    green (< 0.15, safe), yellow (0.15--0.30, caution), orange (0.30--0.45,
    warning), red (> 0.45, reject).

    Parameters
    ----------
    mmd_matrix : ndarray
        Symmetric ``(K, K)`` matrix of pairwise MMD values.
    labels : sequence of str
        Names for each row/column (length K).
    title : str
        Figure title.
    annotate : bool
        Write MMD values inside each cell.
    threshold_lines : bool
        Use deployment-gate tier colouring instead of a continuous cmap.
    cmap : str
        Continuous colormap (used only when *threshold_lines* is False).
    figsize : tuple
        Figure size.
    output_path : path-like, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    if threshold_lines:
        max_val = float(mmd_matrix.max())
        boundaries = [
            0,
            MMD_THRESHOLDS["safe"],
            MMD_THRESHOLDS["caution"],
            MMD_THRESHOLDS["warning"],
            max(max_val * 1.1, MMD_THRESHOLDS["warning"] + 0.01),
        ]
        cmap_obj = ListedColormap(MMD_TIER_COLORS)
        norm = BoundaryNorm(boundaries, cmap_obj.N)
        im = ax.imshow(mmd_matrix, cmap=cmap_obj, norm=norm, aspect="auto")
    else:
        im = ax.imshow(mmd_matrix, cmap=cmap, aspect="auto")

    # Tick labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONT_SIZE - 1)
    ax.set_yticklabels(labels, fontsize=FONT_SIZE - 1)

    # Cell annotations
    if annotate:
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = mmd_matrix[i, j]
                text_color = "white" if val > 0.30 else "black"
                ax.text(
                    j, i, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=max(6, FONT_SIZE - 2), color=text_color,
                )

    ax.set_title(title, fontsize=FONT_SIZE + 4, fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="MMD")

    plt.tight_layout()
    _maybe_save(fig, output_path)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    var_cols: Sequence[str],
    target_cols: Sequence[str],
    *,
    method: str = "pearson",
    title: str = "Correlation Matrix",
    annotate: bool = True,
    cmap: str = "RdBu_r",
    figsize: Optional[tuple] = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    output_path: Optional[Union[str, Path]] = None,
) -> matplotlib.figure.Figure:
    """Feature-vs-target correlation heatmap.

    Computes pairwise correlations between *var_cols* (rows) and
    *target_cols* (columns) from the supplied DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Data with numeric columns.
    var_cols : sequence of str
        Feature / predictor column names (displayed as rows).
    target_cols : sequence of str
        Target / response column names (displayed as columns).
    method : str
        Correlation method (``"pearson"``, ``"spearman"``, ``"kendall"``).
    title : str
        Figure title.
    annotate : bool
        Print correlation values inside cells.
    cmap : str
        Diverging colormap.
    figsize : tuple, optional
        Figure size.  Defaults to a size based on the matrix dimensions.
    vmin, vmax : float
        Colour range.
    output_path : path-like, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    set_publication_style()

    # Compute correlation sub-matrix
    corr_full = df[list(var_cols) + list(target_cols)].corr(method=method)
    corr = corr_full.loc[var_cols, target_cols].values.astype(float)

    n_rows, n_cols = len(var_cols), len(target_cols)
    if figsize is None:
        figsize = (max(6, n_cols * 1.5 + 2), max(4, n_rows * 0.8 + 2))

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(corr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(target_cols, rotation=45, ha="right",
                       fontsize=FONT_SIZE - 1)
    ax.set_yticklabels(var_cols, fontsize=FONT_SIZE - 1)

    if annotate:
        for i in range(n_rows):
            for j in range(n_cols):
                val = corr[i, j]
                text_color = "white" if abs(val) > 0.6 else "black"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=max(6, FONT_SIZE - 2), color=text_color,
                )

    ax.set_title(title, fontsize=FONT_SIZE + 2, fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label=f"{method.title()} r")

    plt.tight_layout()
    _maybe_save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _maybe_save(
    fig: matplotlib.figure.Figure,
    output_path: Optional[Union[str, Path]],
) -> None:
    """Save figure if output_path is provided."""
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=DPI, bbox_inches="tight",
                    facecolor="white")
        print(f"  Saved: {output_path}")
