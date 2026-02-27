"""
Spatial Map Plotting
====================

Publication-quality spatial maps for weed density analysis:

- :func:`plot_kriged_density` -- Choropleth density map from kriging grids
- :func:`plot_lisa_clusters` -- LISA cluster map (HH/LL/HL/LH/NS)
- :func:`plot_gwr_coefficients` -- Multi-panel GWR spatial coefficient maps

All functions return a :class:`matplotlib.figure.Figure` and optionally
save to file at 300 DPI.

Usage::

    from ragweed_toolkit.viz.maps import plot_kriged_density, plot_lisa_clusters
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon

from .style import (
    BOUNDARY_COLOR,
    BOUNDARY_LS,
    BOUNDARY_LW,
    DARK_BG,
    DARK_FACE,
    DARK_LEGEND_BG,
    DPI,
    FONT_SIZE,
    LISA_COLORS,
    LISA_ORDER,
    PANEL_LABEL_SIZE,
    set_publication_style,
)

# ---------------------------------------------------------------------------
# Kriged density maps
# ---------------------------------------------------------------------------

def plot_kriged_density(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    title: str = "Kriged density",
    cbar_label: str = "Detections per cell",
    cmap: str = "YlOrRd",
    boundary_coords: Optional[np.ndarray] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    marker_size: float = 18,
    figsize: tuple = (6, 5),
    panel_label: Optional[str] = None,
    annotation: Optional[str] = None,
    scale_bar_m: Optional[float] = 50.0,
    output_path: Optional[Union[str, Path]] = None,
) -> matplotlib.figure.Figure:
    """Plot a kriged density surface as a coloured scatter map.

    Parameters
    ----------
    x, y : array-like
        Easting and northing coordinates (same CRS, typically UTM metres).
    values : array-like
        Density values at each point.
    title : str
        Axes title.
    cbar_label : str
        Colorbar label text.
    cmap : str
        Matplotlib colormap name.
    boundary_coords : ndarray, optional
        Nx2 array of paddock boundary vertices for overlay.
    vmin, vmax : float, optional
        Colorbar range.  Defaults to data range.
    marker_size : float
        Scatter marker size in points**2.
    figsize : tuple
        Figure size ``(width, height)`` in inches.
    panel_label : str, optional
        Panel label, e.g. ``"(A)"``, placed in upper-left corner.
    annotation : str, optional
        Text annotation placed in upper-right corner (e.g. R-squared).
    scale_bar_m : float, optional
        Length of the manual scale bar in map units (metres).  Set to
        ``None`` to omit.
    output_path : path-like, optional
        If given, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    sc = ax.scatter(
        x, y, c=values, cmap=cmap, s=marker_size,
        edgecolors="none", marker="s", rasterized=True,
        vmin=vmin, vmax=vmax,
    )

    # Boundary overlay
    if boundary_coords is not None:
        poly = MplPolygon(
            boundary_coords, closed=True, fill=False,
            edgecolor=BOUNDARY_COLOR, linewidth=BOUNDARY_LW,
            linestyle=BOUNDARY_LS, zorder=1,
        )
        ax.add_patch(poly)

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02, aspect=25)
    cbar.set_label(cbar_label, fontsize=FONT_SIZE - 1)
    cbar.ax.tick_params(labelsize=FONT_SIZE - 2)

    # Panel label
    if panel_label:
        ax.text(
            0.03, 0.97, panel_label,
            transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE,
            fontweight="bold", va="top", ha="left",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white",
                alpha=0.9, edgecolor="gray",
            ),
        )

    # Annotation
    if annotation:
        ax.text(
            0.97, 0.97, annotation,
            transform=ax.transAxes, fontsize=FONT_SIZE - 1,
            va="top", ha="right", family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="white",
                alpha=0.9, edgecolor="gray",
            ),
        )

    # Scale bar
    if scale_bar_m is not None:
        _add_scale_bar(ax, x, y, scale_bar_m)

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(title, pad=8)
    ax.ticklabel_format(useOffset=True, style="plain")
    ax.set_aspect("equal")

    plt.tight_layout()
    _maybe_save(fig, output_path)
    return fig


def plot_kriged_density_panels(
    panels: Sequence[dict],
    *,
    shared_cbar: bool = True,
    cmap: str = "viridis",
    figsize: Optional[tuple] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> matplotlib.figure.Figure:
    """Multi-panel kriged density maps with optional shared colorbar range.

    Parameters
    ----------
    panels : sequence of dict
        Each dict must contain ``x``, ``y``, ``values`` arrays and may
        optionally include ``title``, ``panel_label``, ``cbar_label``,
        ``annotation``, and ``boundary_coords``.
    shared_cbar : bool
        If True, use a shared vmin/vmax across all panels.
    cmap : str
        Colormap name.
    figsize : tuple, optional
        Figure size.  Defaults to ``(5*n, 5)``.
    output_path : path-like, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    set_publication_style()
    n = len(panels)
    if figsize is None:
        figsize = (5 * n, 5)

    fig, axes = plt.subplots(1, n, figsize=figsize, constrained_layout=True)
    if n == 1:
        axes = [axes]

    vmin = vmax = None
    if shared_cbar:
        vmin = min(p["values"].min() for p in panels)
        vmax = max(p["values"].max() for p in panels)

    for ax, p in zip(axes, panels):
        sc = ax.scatter(
            p["x"], p["y"], c=p["values"], cmap=cmap,
            s=p.get("marker_size", 18), edgecolors="none", marker="s",
            rasterized=True, vmin=vmin, vmax=vmax,
        )

        bnd = p.get("boundary_coords")
        if bnd is not None:
            poly = MplPolygon(
                bnd, closed=True, fill=False,
                edgecolor=BOUNDARY_COLOR, linewidth=BOUNDARY_LW,
                linestyle=BOUNDARY_LS, zorder=1,
            )
            ax.add_patch(poly)

        cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02, aspect=25)
        cbar.set_label(p.get("cbar_label", ""), fontsize=FONT_SIZE - 1)
        cbar.ax.tick_params(labelsize=FONT_SIZE - 2)

        label = p.get("panel_label")
        if label:
            ax.text(
                0.03, 0.97, label,
                transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE,
                fontweight="bold", va="top", ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="white",
                    alpha=0.9, edgecolor="gray",
                ),
            )

        ann = p.get("annotation")
        if ann:
            ax.text(
                0.97, 0.97, ann,
                transform=ax.transAxes, fontsize=FONT_SIZE - 1,
                va="top", ha="right", family="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.4", facecolor="white",
                    alpha=0.9, edgecolor="gray",
                ),
            )

        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_title(p.get("title", ""), pad=8)
        ax.ticklabel_format(useOffset=True, style="plain")
        ax.set_aspect("equal")

    # Remove redundant y-labels on non-first panels
    for ax in axes[1:]:
        ax.set_ylabel("")

    _maybe_save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# LISA cluster maps
# ---------------------------------------------------------------------------

def plot_lisa_clusters(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    *,
    title: str = "LISA Cluster Map",
    boundary_coords: Optional[np.ndarray] = None,
    panel_label: Optional[str] = None,
    dark_theme: bool = True,
    significant_size: float = 40,
    ns_size: float = 15,
    ns_alpha: float = 0.3,
    figsize: tuple = (6, 5.5),
    output_path: Optional[Union[str, Path]] = None,
) -> matplotlib.figure.Figure:
    """Plot a LISA (Local Moran's I) cluster map.

    Points are coloured by cluster type: HH (red), LL (blue), HL (orange),
    LH (purple), NS (gray).

    Parameters
    ----------
    x, y : array-like
        Easting and northing coordinates.
    clusters : array-like of str
        Cluster labels per point: ``"HH"``, ``"LL"``, ``"HL"``, ``"LH"``,
        or ``"NS"``.
    title : str
        Axes title (rendered in italics).
    boundary_coords : ndarray, optional
        Nx2 array of paddock boundary vertices.
    panel_label : str, optional
        Panel label, e.g. ``"(A)"``.
    dark_theme : bool
        If True, use dark background (as in chapter Fig 7).
    significant_size, ns_size : float
        Marker sizes for significant and not-significant points.
    ns_alpha : float
        Alpha for NS points.
    figsize : tuple
        Figure size.
    output_path : path-like, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    set_publication_style()
    clusters = np.asarray(clusters)
    x = np.asarray(x)
    y = np.asarray(y)

    face_color = DARK_FACE if dark_theme else "white"
    text_color = "white" if dark_theme else "black"
    spine_color = "white" if dark_theme else "black"

    fig, ax = plt.subplots(figsize=figsize, facecolor=face_color)
    ax.set_facecolor(DARK_BG if dark_theme else "white")

    # Boundary overlay
    if boundary_coords is not None:
        poly = MplPolygon(
            boundary_coords, closed=True, fill=False,
            edgecolor="white" if dark_theme else "black",
            linewidth=BOUNDARY_LW, linestyle=BOUNDARY_LS, zorder=1,
        )
        ax.add_patch(poly)

    # Plot each cluster type in order (NS first so significant on top)
    actual_counts: dict[str, int] = {}
    for q in LISA_ORDER:
        mask = clusters == q
        cnt = int(mask.sum())
        actual_counts[q] = cnt
        if cnt == 0:
            continue
        if q == "NS":
            ax.scatter(
                x[mask], y[mask], c=LISA_COLORS[q],
                s=ns_size, alpha=ns_alpha, linewidths=0, zorder=2,
            )
        else:
            ax.scatter(
                x[mask], y[mask], c=LISA_COLORS[q],
                s=significant_size, alpha=0.9,
                edgecolors="black", linewidths=0.5, zorder=3,
            )

    # Panel label
    if panel_label:
        ax.text(
            0.03, 0.97, panel_label,
            transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE,
            fontweight="bold", color=text_color, va="top", ha="left",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")]
            if dark_theme else [],
        )

    # Legend
    legend_elements = []
    for q in LISA_ORDER:
        cnt = actual_counts.get(q, 0)
        mkw = dict(marker="o", color=LISA_COLORS[q], linestyle="None",
                    markersize=7 if q != "NS" else 4)
        if q != "NS":
            mkw["markeredgecolor"] = "black"
            mkw["markeredgewidth"] = 0.5
        legend_elements.append(
            Line2D([0], [0], label=f"{q} (n={cnt})", **mkw)
        )

    leg_kw: dict = dict(
        handles=legend_elements, loc="lower left",
        fontsize=8, framealpha=0.85, handletextpad=0.5, borderpad=0.6,
    )
    if dark_theme:
        leg_kw.update(facecolor=DARK_LEGEND_BG, edgecolor="white",
                      labelcolor="white")
    leg = ax.legend(**leg_kw)
    leg.get_frame().set_linewidth(0.5)

    # Axes styling
    ax.set_title(title, fontsize=FONT_SIZE + 1, color=text_color,
                 pad=8, style="italic")
    ax.set_xlabel("Easting (m)", fontsize=FONT_SIZE - 2, color=text_color)
    ax.set_ylabel("Northing (m)", fontsize=FONT_SIZE - 2, color=text_color)
    ax.tick_params(colors=text_color, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(0.5)
        spine.set_visible(True)
    ax.set_aspect("equal")

    plt.tight_layout()
    _maybe_save(fig, output_path, facecolor=fig.get_facecolor())
    return fig


def plot_lisa_panels(
    panels: Sequence[dict],
    *,
    dark_theme: bool = True,
    figsize: Optional[tuple] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> matplotlib.figure.Figure:
    """Multi-panel LISA cluster maps (e.g. species A and B side by side).

    Parameters
    ----------
    panels : sequence of dict
        Each dict must contain ``x``, ``y``, ``clusters`` and may include
        ``title``, ``panel_label``, ``boundary_coords``.
    dark_theme : bool
        Use dark background.
    figsize : tuple, optional
        Figure size.  Defaults to ``(5.5*n, 5.5)``.
    output_path : path-like, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    set_publication_style()
    n = len(panels)
    if figsize is None:
        figsize = (5.5 * n, 5.5)

    face_color = DARK_FACE if dark_theme else "white"
    text_color = "white" if dark_theme else "black"
    spine_color = "white" if dark_theme else "black"

    fig, axes = plt.subplots(
        1, n, figsize=figsize, facecolor=face_color,
    )
    fig.subplots_adjust(wspace=0.25, left=0.06, right=0.97,
                        top=0.92, bottom=0.08)
    if n == 1:
        axes = [axes]

    for ax, p in zip(axes, panels):
        clusters = np.asarray(p["clusters"])
        px = np.asarray(p["x"])
        py = np.asarray(p["y"])

        ax.set_facecolor(DARK_BG if dark_theme else "white")

        bnd = p.get("boundary_coords")
        if bnd is not None:
            poly = MplPolygon(
                bnd, closed=True, fill=False,
                edgecolor="white" if dark_theme else "black",
                linewidth=BOUNDARY_LW, linestyle=BOUNDARY_LS, zorder=1,
            )
            ax.add_patch(poly)

        actual_counts: dict[str, int] = {}
        for q in LISA_ORDER:
            mask = clusters == q
            cnt = int(mask.sum())
            actual_counts[q] = cnt
            if cnt == 0:
                continue
            if q == "NS":
                ax.scatter(px[mask], py[mask], c=LISA_COLORS[q],
                           s=15, alpha=0.3, linewidths=0, zorder=2)
            else:
                ax.scatter(px[mask], py[mask], c=LISA_COLORS[q],
                           s=40, alpha=0.9, edgecolors="black",
                           linewidths=0.5, zorder=3)

        label = p.get("panel_label")
        if label:
            ax.text(
                0.03, 0.97, label,
                transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE,
                fontweight="bold", color=text_color, va="top", ha="left",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")]
                if dark_theme else [],
            )

        legend_elements = []
        for q in LISA_ORDER:
            cnt = actual_counts.get(q, 0)
            mkw = dict(marker="o", color=LISA_COLORS[q], linestyle="None",
                        markersize=7 if q != "NS" else 4)
            if q != "NS":
                mkw["markeredgecolor"] = "black"
                mkw["markeredgewidth"] = 0.5
            legend_elements.append(
                Line2D([0], [0], label=f"{q} (n={cnt})", **mkw)
            )
        leg_kw: dict = dict(
            handles=legend_elements, loc="lower left",
            fontsize=8, framealpha=0.85, handletextpad=0.5, borderpad=0.6,
        )
        if dark_theme:
            leg_kw.update(facecolor=DARK_LEGEND_BG, edgecolor="white",
                          labelcolor="white")
        leg = ax.legend(**leg_kw)
        leg.get_frame().set_linewidth(0.5)

        ax.set_title(p.get("title", ""), fontsize=FONT_SIZE + 1,
                     color=text_color, pad=8, style="italic")
        ax.set_xlabel("Easting (m)", fontsize=FONT_SIZE - 2,
                      color=text_color)
        ax.set_ylabel("Northing (m)", fontsize=FONT_SIZE - 2,
                      color=text_color)
        ax.tick_params(colors=text_color, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(spine_color)
            spine.set_linewidth(0.5)
            spine.set_visible(True)
        ax.set_aspect("equal")

    _maybe_save(fig, output_path, facecolor=fig.get_facecolor())
    return fig


# ---------------------------------------------------------------------------
# GWR coefficient maps
# ---------------------------------------------------------------------------

def plot_gwr_coefficients(
    x: np.ndarray,
    y: np.ndarray,
    coef_dict: dict[str, np.ndarray],
    *,
    boundary_coords: Optional[np.ndarray] = None,
    cmap: str = "viridis",
    marker_size: float = 18,
    figsize: Optional[tuple] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> matplotlib.figure.Figure:
    """Multi-panel GWR spatial coefficient maps.

    Parameters
    ----------
    x, y : array-like
        Easting and northing coordinates.
    coef_dict : dict
        Mapping of coefficient name to array of values.  One panel per key.
        Example: ``{"PC1 proxy": values_a, "AMBEL": values_b}``.
    boundary_coords : ndarray, optional
        Paddock boundary vertices.
    cmap : str
        Colormap.
    marker_size : float
        Scatter marker size.
    figsize : tuple, optional
        Figure size.  Defaults to ``(5*n, 5.5)``.
    output_path : path-like, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    set_publication_style()
    names = list(coef_dict.keys())
    n = len(names)
    if figsize is None:
        figsize = (5 * n, 5.5)

    fig, axes = plt.subplots(1, n, figsize=figsize, constrained_layout=True)
    if n == 1:
        axes = [axes]

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for i, (ax, name) in enumerate(zip(axes, names)):
        values = np.asarray(coef_dict[name])
        sc = ax.scatter(
            x, y, c=values, cmap=cmap, s=marker_size,
            edgecolors="none", marker="s", rasterized=True,
        )

        if boundary_coords is not None:
            poly = MplPolygon(
                boundary_coords, closed=True, fill=False,
                edgecolor=BOUNDARY_COLOR, linewidth=BOUNDARY_LW,
                linestyle=BOUNDARY_LS, zorder=1,
            )
            ax.add_patch(poly)

        cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02, aspect=25)
        cbar.set_label(name, fontsize=FONT_SIZE - 1)
        cbar.ax.tick_params(labelsize=FONT_SIZE - 2)

        ax.text(
            0.03, 0.97, f"({letters[i]})",
            transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE,
            fontweight="bold", va="top", ha="left",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white",
                alpha=0.9, edgecolor="gray",
            ),
        )

        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)" if i == 0 else "")
        ax.ticklabel_format(useOffset=True, style="plain")
        ax.set_aspect("equal")

    _maybe_save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_scale_bar(
    ax: matplotlib.axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    length_m: float,
) -> None:
    """Add a manual scale bar to lower-right of axes."""
    _x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min = float(np.min(y))
    y_range = float(np.max(y)) - y_min

    sb_x0 = x_max - length_m - 10
    sb_y0 = y_min + 0.06 * y_range

    ax.plot(
        [sb_x0, sb_x0 + length_m], [sb_y0, sb_y0],
        color="black", linewidth=2.5, solid_capstyle="butt",
    )
    ax.text(
        sb_x0 + length_m / 2, sb_y0 + 0.02 * y_range,
        f"{length_m:.0f} m",
        ha="center", va="bottom", fontsize=FONT_SIZE - 1, fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white",
            alpha=0.8, edgecolor="none",
        ),
    )


def _maybe_save(
    fig: matplotlib.figure.Figure,
    output_path: Optional[Union[str, Path]],
    **savefig_kwargs,
) -> None:
    """Save figure if output_path is provided."""
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        kw = dict(dpi=DPI, bbox_inches="tight", facecolor="white")
        kw.update(savefig_kwargs)
        fig.savefig(str(output_path), **kw)
        print(f"  Saved: {output_path}")
