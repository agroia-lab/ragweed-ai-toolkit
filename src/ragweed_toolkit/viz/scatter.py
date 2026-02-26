"""
Scatter / Embedding Plots
=========================

Publication-quality scatter plots for dimensionality reduction output:

- :func:`plot_umap` -- UMAP scatter coloured by category
- :func:`plot_pca_biplot` -- PCA biplot with optional loadings arrows

All functions return a :class:`matplotlib.figure.Figure` and optionally
save to file at 300 DPI.

Usage::

    from ragweed_toolkit.viz.scatter import plot_umap, plot_pca_biplot
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from .style import (
    DPI,
    FONT_SIZE,
    PANEL_LABEL_SIZE,
    SPECIES_COLORS,
    set_publication_style,
)


def plot_umap(
    df: pd.DataFrame,
    *,
    x: str = "UMAP1",
    y: str = "UMAP2",
    color_col: str = "database",
    title: Optional[str] = None,
    color_map: Optional[Dict[str, str]] = None,
    marker_size: float = 15,
    alpha: float = 0.7,
    figsize: tuple = (8, 6),
    legend_loc: str = "best",
    output_path: Optional[Union[str, Path]] = None,
) -> matplotlib.figure.Figure:
    """Create a publication-quality UMAP scatter plot.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with coordinate and category columns.
    x, y : str
        Column names for UMAP coordinates.
    color_col : str
        Column to colour points by.
    title : str, optional
        Figure title.
    color_map : dict, optional
        Explicit mapping of category value to hex colour.
    marker_size : float
        Marker size.
    alpha : float
        Marker opacity.
    figsize : tuple
        Figure size.
    legend_loc : str
        Legend location.
    output_path : path-like, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    categories = df[color_col].unique()
    # Sort for reproducible legend order
    categories = sorted(categories, key=str)

    if color_map is None:
        cmap = plt.cm.get_cmap("tab10", len(categories))
        color_map = {cat: matplotlib.colors.to_hex(cmap(i))
                     for i, cat in enumerate(categories)}

    for cat in categories:
        mask = df[color_col] == cat
        ax.scatter(
            df.loc[mask, x], df.loc[mask, y],
            c=color_map.get(cat, "#999999"),
            s=marker_size, alpha=alpha, label=str(cat),
            edgecolors="none",
        )

    ax.set_xlabel(x, fontsize=FONT_SIZE)
    ax.set_ylabel(y, fontsize=FONT_SIZE)
    if title:
        ax.set_title(title, fontsize=FONT_SIZE + 2, pad=10)

    ax.legend(
        loc=legend_loc, fontsize=FONT_SIZE - 1,
        framealpha=0.9, markerscale=1.5,
    )

    plt.tight_layout()
    _maybe_save(fig, output_path)
    return fig


def plot_pca_biplot(
    df: pd.DataFrame,
    *,
    x: str = "PC1",
    y: str = "PC2",
    color_col: str = "species",
    loadings: Optional[pd.DataFrame] = None,
    n_loadings: int = 5,
    var_explained: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    color_map: Optional[Dict[str, str]] = None,
    marker_size: float = 20,
    alpha: float = 0.6,
    figsize: tuple = (8, 7),
    output_path: Optional[Union[str, Path]] = None,
) -> matplotlib.figure.Figure:
    """Create a PCA biplot with optional loadings arrows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with PC score columns and a category column.
    x, y : str
        Column names for PC1 and PC2 scores.
    color_col : str
        Column to colour points by.
    loadings : pd.DataFrame, optional
        Loadings matrix with rows as features and columns matching *x*, *y*.
        If provided, arrows are drawn for the top *n_loadings* features.
    n_loadings : int
        Number of loading arrows to show (by L2 norm).
    var_explained : tuple of float, optional
        Variance explained fractions for PC1 and PC2, used in axis labels.
    title : str, optional
        Figure title.
    color_map : dict, optional
        Category-to-colour mapping.  Defaults to :data:`SPECIES_COLORS`.
    marker_size : float
        Marker size.
    alpha : float
        Marker opacity.
    figsize : tuple
        Figure size.
    output_path : path-like, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    set_publication_style()

    if color_map is None:
        color_map = SPECIES_COLORS

    fig, ax = plt.subplots(figsize=figsize)

    categories = sorted(df[color_col].unique(), key=str)
    cmap_fallback = plt.cm.get_cmap("tab10", len(categories))

    for i, cat in enumerate(categories):
        mask = df[color_col] == cat
        c = color_map.get(cat, matplotlib.colors.to_hex(cmap_fallback(i)))
        ax.scatter(
            df.loc[mask, x], df.loc[mask, y],
            c=c, s=marker_size, alpha=alpha, label=str(cat),
            edgecolors="white", linewidths=0.3,
        )

    # Loadings arrows
    if loadings is not None and x in loadings.columns and y in loadings.columns:
        # Select top features by magnitude
        norms = np.sqrt(loadings[x] ** 2 + loadings[y] ** 2)
        top_idx = norms.nlargest(n_loadings).index

        # Scale arrows to data range
        x_range = df[x].max() - df[x].min()
        y_range = df[y].max() - df[y].min()
        scale = 0.4 * min(x_range, y_range) / norms.max()

        for feat in top_idx:
            lx = loadings.loc[feat, x] * scale
            ly = loadings.loc[feat, y] * scale
            ax.annotate(
                "", xy=(lx, ly), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5),
            )
            ax.text(
                lx * 1.1, ly * 1.1, str(feat),
                fontsize=FONT_SIZE - 2, color="#c0392b",
                ha="center", va="center",
            )

    # Axis labels
    xlabel = x
    ylabel = y
    if var_explained is not None:
        xlabel = f"{x} ({var_explained[0]:.1%} var.)"
        ylabel = f"{y} ({var_explained[1]:.1%} var.)"
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)

    if title:
        ax.set_title(title, fontsize=FONT_SIZE + 2, pad=10)

    ax.legend(
        loc="best", fontsize=FONT_SIZE - 1,
        framealpha=0.9, markerscale=1.5,
    )
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

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
