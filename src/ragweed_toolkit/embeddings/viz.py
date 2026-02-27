"""
Embedding Visualization Functions
====================================

Interactive and static visualizations for embedding analysis:

- :func:`umap_scatter` -- Interactive Plotly scatter plot colored by group
  (database, source type, etc.), with optional image click viewer.
- :func:`mmd_heatmap` -- Pairwise MMD matrix as a matplotlib heatmap, with
  deployment-gate color coding.

Usage::

    from ragweed_toolkit.embeddings import umap_scatter, mmd_heatmap

    # Interactive scatter
    fig = umap_scatter(df, color_col="database", output_path="umap.html")

    # MMD heatmap
    fig = mmd_heatmap(mmd_matrix, labels, output_path="mmd_heatmap.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

if TYPE_CHECKING:
    import matplotlib.figure
    import plotly.graph_objects

import numpy as np
import pandas as pd


def umap_scatter(
    df: pd.DataFrame,
    *,
    x_col: str = "umap_x",
    y_col: str = "umap_y",
    z_col: Optional[str] = None,
    color_col: str = "database",
    hover_cols: Optional[List[str]] = None,
    title: str = "Embedding Visualization",
    output_path: Optional[Union[str, Path]] = None,
    color_map: Optional[Dict[str, str]] = None,
    width: int = 1200,
    height: int = 800,
    marker_size: int = 5,
    opacity: float = 0.7,
    template: str = "plotly_white",
) -> "plotly.graph_objects.Figure":
    """Create an interactive Plotly scatter plot of embeddings.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with coordinate columns and metadata.
    x_col, y_col : str
        Column names for X and Y coordinates.
    z_col : str, optional
        Column name for Z coordinate (creates 3D plot if given).
    color_col : str
        Column to color points by.
    hover_cols : list[str], optional
        Additional columns to show on hover. Defaults to all non-coordinate
        columns present in *df*.
    title : str
        Plot title.
    output_path : path-like, optional
        If given, save the plot as an HTML file.
    color_map : dict, optional
        Explicit mapping of category -> hex color.
    width, height : int
        Figure dimensions in pixels.
    marker_size : int
        Marker size.
    opacity : float
        Marker opacity.
    template : str
        Plotly template name.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.express as px

    # Auto-detect hover columns
    if hover_cols is None:
        coord_cols = {x_col, y_col}
        if z_col:
            coord_cols.add(z_col)
        hover_cols = [c for c in df.columns if c not in coord_cols and c != color_col]

    plot_kwargs = dict(
        data_frame=df,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_data=hover_cols,
        title=title,
        opacity=opacity,
    )
    if color_map:
        plot_kwargs["color_discrete_map"] = color_map

    if z_col is not None:
        fig = px.scatter_3d(**plot_kwargs, z=z_col)
        fig.update_traces(marker=dict(size=max(2, marker_size - 2)))
    else:
        fig = px.scatter(**plot_kwargs)
        fig.update_traces(marker=dict(size=marker_size))

    fig.update_layout(
        template=template,
        width=width,
        height=height,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        print(f"  Saved: {output_path}")

        # Try static PNG
        png_path = output_path.with_suffix(".png")
        try:
            fig.write_image(str(png_path), scale=2)
            print(f"  Saved: {png_path}")
        except Exception:
            pass  # kaleido may not be installed

    return fig


def mmd_heatmap(
    mmd_matrix: np.ndarray,
    labels: Sequence[str],
    *,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Pairwise MMD Matrix",
    figsize: tuple = (10, 8),
    cmap: str = "YlOrRd",
    annotate: bool = True,
    threshold_lines: bool = True,
) -> "matplotlib.figure.Figure":
    """Create a pairwise MMD heatmap with deployment-gate color coding.

    Parameters
    ----------
    mmd_matrix : np.ndarray
        Symmetric matrix of shape ``(K, K)`` with pairwise MMD values.
    labels : sequence of str
        Names for each row/column (length K).
    output_path : path-like, optional
        If given, save the figure as a PNG.
    title : str
        Figure title.
    figsize : tuple
        Matplotlib figure size.
    cmap : str
        Colormap name.
    annotate : bool
        If True, write MMD values inside each cell.
    threshold_lines : bool
        If True, draw contour lines at deployment gate thresholds (0.15, 0.30, 0.45).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    fig, ax = plt.subplots(figsize=figsize)

    if threshold_lines:
        # Custom colormap matching deployment gate tiers
        boundaries = [0, 0.15, 0.30, 0.45, mmd_matrix.max() * 1.1 or 1.0]
        colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]  # green/yellow/orange/red
        cmap_obj = ListedColormap(colors)
        norm = BoundaryNorm(boundaries, cmap_obj.N)
        im = ax.imshow(mmd_matrix, cmap=cmap_obj, norm=norm, aspect="auto")
    else:
        im = ax.imshow(mmd_matrix, cmap=cmap, aspect="auto")

    # Labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotations
    if annotate:
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = mmd_matrix[i, j]
                text_color = "white" if val > 0.3 else "black"
                ax.text(
                    j, i, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=8, color=text_color,
                )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="MMD")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            str(output_path), dpi=200, bbox_inches="tight", facecolor="white"
        )
        plt.close(fig)
        print(f"  Saved: {output_path}")

    return fig


def mmd_bar_chart(
    database_mmds: Dict[str, float],
    *,
    overall_mmd: Optional[float] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Domain Shift: MMD by Database",
    figsize: tuple = (10, 6),
    chilean_dbs: Optional[set] = None,
) -> "matplotlib.figure.Figure":
    """Create a horizontal bar chart of MMD values per reference database.

    Parameters
    ----------
    database_mmds : dict
        Mapping of database name to MMD value.
    overall_mmd : float, optional
        If given, draw a vertical line for the overall MMD.
    output_path : path-like, optional
        If given, save the figure as a PNG.
    title : str
        Figure title.
    figsize : tuple
        Matplotlib figure size.
    chilean_dbs : set, optional
        Set of database names to highlight as Chilean (blue). Others are
        colored orange.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if chilean_dbs is None:
        chilean_dbs = set()

    # Sort by MMD ascending
    sorted_items = sorted(database_mmds.items(), key=lambda x: x[1])
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = ["#2171b5" if n in chilean_dbs else "#e6550d" for n in names]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5, height=0.6)

    # Annotate bars
    for i, val in enumerate(values):
        ax.text(val + 0.008, i, f"{val:.3f}", va="center", ha="left", fontsize=9)

    # Deployment threshold lines
    for threshold in [0.15, 0.30, 0.45]:
        ax.axvline(threshold, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(threshold, len(names) - 0.3, f"{threshold}", fontsize=8,
                color="#666666", ha="center")

    # Overall MMD line
    if overall_mmd is not None:
        ax.axvline(overall_mmd, color="#c0392b", linestyle="-", linewidth=2, alpha=0.8)
        ax.text(overall_mmd + 0.01, -0.5, f"Overall={overall_mmd:.3f}",
                fontsize=10, color="#c0392b", fontweight="bold")

    ax.set_xlabel("MMD (Maximum Mean Discrepancy)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, max(values) * 1.3 if values else 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    if chilean_dbs:
        handles = [
            mpatches.Patch(color="#2171b5", label="Chilean"),
            mpatches.Patch(color="#e6550d", label="International"),
        ]
        ax.legend(handles=handles, fontsize=10, loc="lower right", framealpha=0.9)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {output_path}")

    return fig
