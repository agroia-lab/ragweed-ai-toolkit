#!/usr/bin/env python3
"""
Generate spatial maps of Presto temporal window embeddings for documentation.

This script creates publication-quality spatial maps comparing 4 Presto
temporal windows (Growth, Pre-Season, Planting, Standard 12m) using both
PCA false-color composites and K=20 cluster coloring.

Figures generated:
  1. presto_windows_pca_rgb_all.png      - 2x2 panel, PCA false-color, all paddocks
  2. presto_windows_clusters_all.png     - 2x2 panel, cluster coloring, all paddocks
  3. presto_windows_single_paddock.png   - 4x3 grid, PCA per window+paddock
  4. presto_windows_cluster_comparison.png - 4x3 grid, clusters per window+paddock

Output:
  /home/malezainia1/dev/INIA_DeepLearning_Ubuntu_mod_lleon/outputs/presto_cross_window_comparison/
  (also copied to docs/latex/figures/)

Usage:
    cd /home/malezainia1/dev/INIA_DeepLearning_Ubuntu_mod_lleon
    /home/malezainia1/anaconda3/envs/yolov8_custom/bin/python scripts/satellite/map_presto_windows.py
"""

import os
import sys
import shutil
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap, Normalize
import numpy as np
import geopandas as gpd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

EMB_DIR = Path("/media/malezainia1/LORENZO/outputs_sugal_25-26/satellite_embeddings")
BOUNDARIES_FILE = EMB_DIR / "paddock_boundaries.gpkg"

PROJECT_ROOT = Path("/home/malezainia1/dev/INIA_DeepLearning_Ubuntu_mod_lleon")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "presto_cross_window_comparison"
FIGURES_DIR = PROJECT_ROOT / "docs" / "latex" / "figures"

# The 4 temporal windows
WINDOWS = [
    {
        "file": EMB_DIR / "embeddings_24-25_presto_growth_complete.gpkg",
        "name": "Growth",
        "dates": "Dec 2024 - Feb 2025",
        "label": "Presto Growth (Dec24-Feb25)",
    },
    {
        "file": EMB_DIR / "embeddings_25-26_presto_pre_season_complete.gpkg",
        "name": "Pre-Season",
        "dates": "Jun - Aug 2025",
        "label": "Presto Pre-Season (Jun-Aug25)",
    },
    {
        "file": EMB_DIR / "embeddings_25-26_presto_planting_complete.gpkg",
        "name": "Planting",
        "dates": "Sep - Nov 2025",
        "label": "Presto Planting (Sep-Nov25)",
    },
    {
        "file": EMB_DIR / "embeddings_25-26_presto_nov-oct.gpkg",
        "name": "Standard 12m",
        "dates": "Nov 2024 - Oct 2025",
        "label": "Presto Standard 12m (Nov24-Oct25)",
    },
]

# Paddocks for the single-paddock grid (3 representative ones)
SELECTED_PADDOCKS = ["INIA_Rayentue", "Apalta", "Santa_Ines"]

# Embedding columns
EMB_COLS = [f"A{i:02d}" for i in range(128)]

# K=20 cluster colors (matching QGIS script)
CLUSTER_COLORS_20 = {
    0:  "#e6194b",
    1:  "#3cb44b",
    2:  "#4363d8",
    3:  "#f58231",
    4:  "#911eb4",
    5:  "#42d4f4",
    6:  "#f032e6",
    7:  "#bfef45",
    8:  "#fabebe",
    9:  "#469990",
    10: "#e6beff",
    11: "#9a6324",
    12: "#fffac8",
    13: "#800000",
    14: "#aaffc3",
    15: "#808000",
    16: "#ffd8b1",
    17: "#000075",
    18: "#a9a9a9",
    19: "#000000",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_window_data(window_info):
    """Load a GeoPackage and return the GeoDataFrame."""
    filepath = window_info["file"]
    if not filepath.exists():
        print(f"  WARNING: File not found: {filepath}")
        return None
    print(f"  Loading: {window_info['name']} ({filepath.name})...")
    gdf = gpd.read_file(filepath)
    print(f"    {len(gdf)} cells, {gdf['paddock'].nunique()} paddocks")
    return gdf


def compute_pca_rgb(gdf, emb_cols):
    """
    Compute PCA on embedding columns and return RGB values (0-1 scale).

    Uses the pre-computed pca_global columns if available, otherwise
    computes fresh PCA.

    Returns:
        numpy array of shape (n_cells, 3) with R, G, B in [0, 1]
    """
    # Check if pre-computed PCA RGB columns exist
    has_precomputed = all(
        col in gdf.columns for col in ['pca_global_1_rgb', 'pca_global_2_rgb', 'pca_global_3_rgb']
    )

    if has_precomputed:
        r = gdf['pca_global_1_rgb'].values.astype(float)
        g = gdf['pca_global_2_rgb'].values.astype(float)
        b = gdf['pca_global_3_rgb'].values.astype(float)

        # Handle NaN values
        nan_mask = np.isnan(r) | np.isnan(g) | np.isnan(b)
        r[nan_mask] = 128
        g[nan_mask] = 128
        b[nan_mask] = 128

        rgb = np.column_stack([r / 255.0, g / 255.0, b / 255.0])
        return rgb

    # Fallback: compute PCA from scratch
    available_cols = [c for c in emb_cols if c in gdf.columns]
    if len(available_cols) < 3:
        return np.full((len(gdf), 3), 0.5)

    data = gdf[available_cols].values.astype(float)
    nan_mask = np.isnan(data).any(axis=1)
    data[nan_mask] = 0

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data_scaled)

    # Min-max normalize each component to [0, 1]
    rgb = np.zeros((len(gdf), 3))
    for i in range(3):
        pc = pca_result[:, i]
        pc_min, pc_max = pc.min(), pc.max()
        if pc_max > pc_min:
            rgb[:, i] = (pc - pc_min) / (pc_max - pc_min)
        else:
            rgb[:, i] = 0.5

    return rgb


def plot_gdf_rgb(ax, gdf, rgb_values, boundaries=None, title=None):
    """
    Plot GeoDataFrame cells colored by RGB values on a given axes.

    Each cell is a polygon; we plot them individually with their assigned color.
    """
    # Plot each cell with its RGB color
    for idx in range(len(gdf)):
        color = rgb_values[idx]
        geom = gdf.geometry.iloc[idx]
        if geom is None or geom.is_empty:
            continue
        ax.fill(
            *geom.exterior.xy,
            color=color,
            edgecolor='none',
            linewidth=0,
        )

    # Add paddock boundaries as black outlines
    if boundaries is not None:
        boundaries.boundary.plot(ax=ax, color='black', linewidth=0.8)

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')

    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=7)
    ax.ticklabel_format(style='plain', axis='both')


def plot_gdf_rgb_fast(ax, gdf, rgb_values, boundaries=None, title=None):
    """
    Fast plotting of GeoDataFrame cells using PatchCollection.

    Much faster than plotting each polygon individually for large datasets.
    """
    from matplotlib.patches import Polygon as MplPolygon

    patches = []
    colors = []

    for idx in range(len(gdf)):
        geom = gdf.geometry.iloc[idx]
        if geom is None or geom.is_empty:
            continue

        # Handle MultiPolygon; take only x,y (ignore z if 3D)
        if geom.geom_type == 'MultiPolygon':
            for part in geom.geoms:
                coords = np.array(part.exterior.coords)[:, :2]
                patches.append(MplPolygon(coords, closed=True))
                colors.append(rgb_values[idx])
        else:
            coords = np.array(geom.exterior.coords)[:, :2]
            patches.append(MplPolygon(coords, closed=True))
            colors.append(rgb_values[idx])

    pc = PatchCollection(patches, edgecolors='none', linewidths=0)
    pc.set_facecolor(colors)
    ax.add_collection(pc)

    # Auto-scale axes to data extent
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    margin_x = (bounds[2] - bounds[0]) * 0.05
    margin_y = (bounds[3] - bounds[1]) * 0.05
    ax.set_xlim(bounds[0] - margin_x, bounds[2] + margin_x)
    ax.set_ylim(bounds[1] - margin_y, bounds[3] + margin_y)

    # Add paddock boundaries as black outlines
    if boundaries is not None:
        # Only plot boundaries that overlap with the current extent
        boundaries.boundary.plot(ax=ax, color='black', linewidth=0.8)

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')

    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=7)
    ax.ticklabel_format(style='plain', axis='both')


def plot_gdf_clusters_fast(ax, gdf, cluster_col, boundaries=None, title=None, cmap_dict=None):
    """
    Fast plotting of GeoDataFrame cells colored by cluster ID.
    """
    from matplotlib.patches import Polygon as MplPolygon

    if cmap_dict is None:
        cmap_dict = CLUSTER_COLORS_20

    patches = []
    colors = []

    cluster_vals = gdf[cluster_col].values

    for idx in range(len(gdf)):
        geom = gdf.geometry.iloc[idx]
        if geom is None or geom.is_empty:
            continue

        cluster_id = cluster_vals[idx]
        if np.isnan(cluster_id):
            color = '#cccccc'
        else:
            color = cmap_dict.get(int(cluster_id), '#cccccc')

        if geom.geom_type == 'MultiPolygon':
            for part in geom.geoms:
                coords = np.array(part.exterior.coords)[:, :2]
                patches.append(MplPolygon(coords, closed=True))
                colors.append(color)
        else:
            coords = np.array(geom.exterior.coords)[:, :2]
            patches.append(MplPolygon(coords, closed=True))
            colors.append(color)

    pc = PatchCollection(patches, edgecolors='none', linewidths=0)
    pc.set_facecolor(colors)
    ax.add_collection(pc)

    # Auto-scale
    bounds = gdf.total_bounds
    margin_x = (bounds[2] - bounds[0]) * 0.05
    margin_y = (bounds[3] - bounds[1]) * 0.05
    ax.set_xlim(bounds[0] - margin_x, bounds[2] + margin_x)
    ax.set_ylim(bounds[1] - margin_y, bounds[3] + margin_y)

    if boundaries is not None:
        boundaries.boundary.plot(ax=ax, color='black', linewidth=0.8)

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')

    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=7)
    ax.ticklabel_format(style='plain', axis='both')


def plot_paddock_subset(ax, gdf, paddock_name, values_func, boundaries, title=None, mode='rgb'):
    """
    Plot a single paddock from a GeoDataFrame.

    Args:
        values_func: either RGB array (mode='rgb') or cluster column name (mode='cluster')
    """
    from matplotlib.patches import Polygon as MplPolygon

    # Filter to paddock
    mask = gdf['paddock'] == paddock_name
    subset = gdf[mask].copy().reset_index(drop=True)

    if len(subset) == 0:
        ax.text(0.5, 0.5, f"No data\n{paddock_name}",
                ha='center', va='center', transform=ax.transAxes, fontsize=9)
        if title:
            ax.set_title(title, fontsize=9, fontweight='bold')
        return

    # Get boundary for this paddock
    bnd_mask = boundaries['paddock'] == paddock_name
    bnd_subset = boundaries[bnd_mask]

    patches = []
    colors = []

    if mode == 'rgb':
        # values_func is an RGB array matching gdf index
        rgb_full = values_func
        rgb_subset = rgb_full[mask.values] if hasattr(mask, 'values') else rgb_full[mask]

        for idx in range(len(subset)):
            geom = subset.geometry.iloc[idx]
            if geom is None or geom.is_empty:
                continue
            color = rgb_subset[idx]
            if geom.geom_type == 'MultiPolygon':
                for part in geom.geoms:
                    coords = np.array(part.exterior.coords)[:, :2]
                    patches.append(MplPolygon(coords, closed=True))
                    colors.append(color)
            else:
                coords = np.array(geom.exterior.coords)[:, :2]
                patches.append(MplPolygon(coords, closed=True))
                colors.append(color)

    elif mode == 'cluster':
        cluster_col = values_func
        cluster_vals = subset[cluster_col].values
        for idx in range(len(subset)):
            geom = subset.geometry.iloc[idx]
            if geom is None or geom.is_empty:
                continue
            cid = cluster_vals[idx]
            color = '#cccccc' if np.isnan(cid) else CLUSTER_COLORS_20.get(int(cid), '#cccccc')
            if geom.geom_type == 'MultiPolygon':
                for part in geom.geoms:
                    coords = np.array(part.exterior.coords)[:, :2]
                    patches.append(MplPolygon(coords, closed=True))
                    colors.append(color)
            else:
                coords = np.array(geom.exterior.coords)[:, :2]
                patches.append(MplPolygon(coords, closed=True))
                colors.append(color)

    if patches:
        pc = PatchCollection(patches, edgecolors='none', linewidths=0)
        pc.set_facecolor(colors)
        ax.add_collection(pc)

    # Auto-scale to paddock extent
    bounds = subset.total_bounds
    margin_x = max((bounds[2] - bounds[0]) * 0.08, 10)
    margin_y = max((bounds[3] - bounds[1]) * 0.08, 10)
    ax.set_xlim(bounds[0] - margin_x, bounds[2] + margin_x)
    ax.set_ylim(bounds[1] - margin_y, bounds[3] + margin_y)

    if len(bnd_subset) > 0:
        bnd_subset.boundary.plot(ax=ax, color='black', linewidth=1.0)

    if title:
        ax.set_title(title, fontsize=9, fontweight='bold')

    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=6)
    ax.ticklabel_format(style='plain', axis='both')


def create_cluster_legend(fig, n_clusters=20):
    """Add a cluster color legend to a figure."""
    legend_patches = []
    for i in range(n_clusters):
        color = CLUSTER_COLORS_20.get(i, '#cccccc')
        legend_patches.append(mpatches.Patch(facecolor=color, label=f"C{i}"))

    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=10,
        fontsize=7,
        frameon=True,
        title="Cluster ID (K=20)",
        title_fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )


# ============================================================
# FIGURE GENERATION
# ============================================================

def _get_paddock_order(gdf):
    """Get a consistent paddock ordering for grid layout (by area, descending)."""
    paddock_areas = {}
    for paddock in gdf['paddock'].unique():
        mask = gdf['paddock'] == paddock
        bounds = gdf[mask].total_bounds
        area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        paddock_areas[paddock] = area
    return sorted(paddock_areas.keys(), key=lambda p: paddock_areas[p], reverse=True)


def figure_1_pca_rgb_all(all_data, boundaries, output_dir):
    """
    Figure 1: 2x2 panel with PCA false-color for all paddocks.

    Since paddocks are spread across a wide region, each panel shows a
    multi-paddock composite with individual paddock sub-views arranged in
    a grid-like layout (using separate inset axes per geographic cluster).
    """
    print("\n--- Figure 1: PCA RGB All Paddocks (2x2) ---")

    # Get paddock list from the first window (largest dataset)
    all_paddocks = _get_paddock_order(all_data[0][1])
    n_paddocks = len(all_paddocks)
    # Arrange paddocks in a grid within each panel
    ncols_inner = 4
    nrows_inner = int(np.ceil(n_paddocks / ncols_inner))

    fig, outer_axes = plt.subplots(2, 2, figsize=(16, 14))
    outer_axes = outer_axes.flatten()

    for win_idx, (win_info, gdf, rgb) in enumerate(all_data):
        # Remove the outer axes (we will use inner gridspec)
        outer_axes[win_idx].set_visible(False)

    # Use gridspec for nested layout
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    fig.clf()
    outer_gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.1,
                        top=0.93, bottom=0.02, left=0.02, right=0.98)

    for win_idx, (win_info, gdf, rgb) in enumerate(all_data):
        inner_gs = GridSpecFromSubplotSpec(
            nrows_inner, ncols_inner, subplot_spec=outer_gs[win_idx],
            hspace=0.3, wspace=0.15
        )

        # Window title as text
        row = win_idx // 2
        col = win_idx % 2
        title_y = outer_gs[win_idx].get_position(fig).y1 + 0.01
        title_x = (outer_gs[win_idx].get_position(fig).x0 +
                    outer_gs[win_idx].get_position(fig).x1) / 2
        fig.text(title_x, title_y,
                 f"{win_info['name']} ({win_info['dates']})",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

        for p_idx, paddock in enumerate(all_paddocks):
            p_row = p_idx // ncols_inner
            p_col = p_idx % ncols_inner

            ax = fig.add_subplot(inner_gs[p_row, p_col])

            mask = gdf['paddock'] == paddock
            if mask.sum() == 0:
                ax.text(0.5, 0.5, f"{paddock}\n(no data)",
                        ha='center', va='center', transform=ax.transAxes, fontsize=6)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            plot_paddock_subset(ax, gdf, paddock, rgb, boundaries,
                                title=paddock.replace('_', ' '), mode='rgb')
            ax.set_title(paddock.replace('_', ' '), fontsize=7, pad=2)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Presto Temporal Windows: PCA False-Color Composite\n(R=PC1, G=PC2, B=PC3 per window)",
        fontsize=14, fontweight='bold', y=0.99,
    )

    outpath = output_dir / "presto_windows_pca_rgb_all.png"
    fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


def figure_2_clusters_all(all_data, boundaries, output_dir):
    """
    Figure 2: 2x2 panel with K=20 cluster coloring for all paddocks.

    Each panel shows individual paddock sub-views with cluster colors.
    """
    print("\n--- Figure 2: Clusters All Paddocks (2x2) ---")

    all_paddocks = _get_paddock_order(all_data[0][1])
    n_paddocks = len(all_paddocks)
    ncols_inner = 4
    nrows_inner = int(np.ceil(n_paddocks / ncols_inner))

    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    fig = plt.figure(figsize=(16, 16))
    outer_gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.1,
                        top=0.92, bottom=0.06, left=0.02, right=0.98)

    for win_idx, (win_info, gdf, _) in enumerate(all_data):
        inner_gs = GridSpecFromSubplotSpec(
            nrows_inner, ncols_inner, subplot_spec=outer_gs[win_idx],
            hspace=0.3, wspace=0.15
        )

        title_y = outer_gs[win_idx].get_position(fig).y1 + 0.01
        title_x = (outer_gs[win_idx].get_position(fig).x0 +
                    outer_gs[win_idx].get_position(fig).x1) / 2
        fig.text(title_x, title_y,
                 f"{win_info['name']} ({win_info['dates']})",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

        for p_idx, paddock in enumerate(all_paddocks):
            p_row = p_idx // ncols_inner
            p_col = p_idx % ncols_inner

            ax = fig.add_subplot(inner_gs[p_row, p_col])

            mask = gdf['paddock'] == paddock
            if mask.sum() == 0:
                ax.text(0.5, 0.5, f"{paddock}\n(no data)",
                        ha='center', va='center', transform=ax.transAxes, fontsize=6)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            plot_paddock_subset(ax, gdf, paddock, 'cluster_global_20', boundaries,
                                title=paddock.replace('_', ' '), mode='cluster')
            ax.set_title(paddock.replace('_', ' '), fontsize=7, pad=2)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Presto Temporal Windows: Global Clusters (K=20)",
        fontsize=14, fontweight='bold', y=0.99,
    )

    create_cluster_legend(fig)

    outpath = output_dir / "presto_windows_clusters_all.png"
    fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


def figure_3_single_paddock_pca(all_data, boundaries, selected_paddocks, output_dir):
    """
    Figure 3: 4-row x 3-column grid.
    Rows = temporal windows, Columns = selected paddocks.
    PCA false-color per cell.
    """
    print("\n--- Figure 3: Single Paddock PCA Grid (4x3) ---")

    n_rows = len(all_data)
    n_cols = len(selected_paddocks)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 18))

    for row_idx, (win_info, gdf, rgb) in enumerate(all_data):
        for col_idx, paddock in enumerate(selected_paddocks):
            ax = axes[row_idx, col_idx]

            # Only show row label on first column
            if col_idx == 0:
                row_label = f"{win_info['name']}\n({win_info['dates']})"
            else:
                row_label = None

            # Only show column label on first row
            if row_idx == 0:
                col_label = paddock.replace('_', ' ')
            else:
                col_label = None

            # Build title
            parts = []
            if col_label:
                parts.append(col_label)
            if row_label:
                parts.append(row_label)
            title = "\n".join(parts) if parts else None

            # Simpler title: just paddock on top row, window on left
            if row_idx == 0 and col_idx == 0:
                title = f"{paddock.replace('_', ' ')}\n{win_info['name']} ({win_info['dates']})"
            elif row_idx == 0:
                title = paddock.replace('_', ' ')
            elif col_idx == 0:
                title = f"{win_info['name']} ({win_info['dates']})"
            else:
                title = None

            plot_paddock_subset(
                ax, gdf, paddock, rgb, boundaries,
                title=title, mode='rgb'
            )

    plt.suptitle(
        "Presto Temporal Windows: PCA False-Color by Paddock\n"
        "(Rows = temporal windows, Columns = paddocks)",
        fontsize=13, fontweight='bold', y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    outpath = output_dir / "presto_windows_single_paddock.png"
    fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


def figure_4_single_paddock_clusters(all_data, boundaries, selected_paddocks, output_dir):
    """
    Figure 4: 4-row x 3-column grid.
    Rows = temporal windows, Columns = selected paddocks.
    Cluster coloring per cell.
    """
    print("\n--- Figure 4: Single Paddock Clusters Grid (4x3) ---")

    n_rows = len(all_data)
    n_cols = len(selected_paddocks)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 18))

    for row_idx, (win_info, gdf, _) in enumerate(all_data):
        for col_idx, paddock in enumerate(selected_paddocks):
            ax = axes[row_idx, col_idx]

            if row_idx == 0 and col_idx == 0:
                title = f"{paddock.replace('_', ' ')}\n{win_info['name']} ({win_info['dates']})"
            elif row_idx == 0:
                title = paddock.replace('_', ' ')
            elif col_idx == 0:
                title = f"{win_info['name']} ({win_info['dates']})"
            else:
                title = None

            plot_paddock_subset(
                ax, gdf, paddock, 'cluster_global_20', boundaries,
                title=title, mode='cluster'
            )

    plt.suptitle(
        "Presto Temporal Windows: Global Clusters (K=20) by Paddock\n"
        "(Rows = temporal windows, Columns = paddocks)",
        fontsize=13, fontweight='bold', y=0.99,
    )

    create_cluster_legend(fig)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    outpath = output_dir / "presto_windows_cluster_comparison.png"
    fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PRESTO TEMPORAL WINDOWS - SPATIAL MAP GENERATOR")
    print("=" * 60)
    print()

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print()

    # Load boundaries
    print("Loading paddock boundaries...")
    if not BOUNDARIES_FILE.exists():
        print(f"ERROR: Boundaries not found: {BOUNDARIES_FILE}")
        sys.exit(1)
    boundaries = gpd.read_file(BOUNDARIES_FILE)
    print(f"  {len(boundaries)} polygons, {boundaries['paddock'].nunique()} paddocks")
    print()

    # Load all window data and compute PCA RGB
    all_data = []  # List of (window_info, gdf, rgb_array)

    for win_info in WINDOWS:
        gdf = load_window_data(win_info)
        if gdf is None:
            print(f"  Skipping {win_info['name']}")
            continue

        print(f"  Computing PCA RGB for {win_info['name']}...")
        rgb = compute_pca_rgb(gdf, EMB_COLS)
        print(f"    RGB shape: {rgb.shape}, range: [{rgb.min():.2f}, {rgb.max():.2f}]")

        all_data.append((win_info, gdf, rgb))
        print()

    if len(all_data) == 0:
        print("ERROR: No window data loaded. Check file paths.")
        sys.exit(1)

    print(f"Loaded {len(all_data)} temporal windows.")
    print()

    # Check which selected paddocks are available in all windows
    available_paddocks = set(all_data[0][1]['paddock'].unique())
    for _, gdf, _ in all_data[1:]:
        available_paddocks &= set(gdf['paddock'].unique())

    selected = [p for p in SELECTED_PADDOCKS if p in available_paddocks]
    if len(selected) < len(SELECTED_PADDOCKS):
        missing = set(SELECTED_PADDOCKS) - set(selected)
        print(f"  Note: Paddock(s) {missing} not in all windows, using: {selected}")

    # Generate figures
    generated_files = []

    # Figure 1: PCA RGB all paddocks
    f1 = figure_1_pca_rgb_all(all_data, boundaries, OUTPUT_DIR)
    generated_files.append(f1)

    # Figure 2: Clusters all paddocks
    f2 = figure_2_clusters_all(all_data, boundaries, OUTPUT_DIR)
    generated_files.append(f2)

    # Figure 3: Single paddock PCA grid
    if selected:
        f3 = figure_3_single_paddock_pca(all_data, boundaries, selected, OUTPUT_DIR)
        generated_files.append(f3)
    else:
        print("  Skipping Figure 3: no common paddocks found")

    # Figure 4: Single paddock clusters grid
    if selected:
        f4 = figure_4_single_paddock_clusters(all_data, boundaries, selected, OUTPUT_DIR)
        generated_files.append(f4)
    else:
        print("  Skipping Figure 4: no common paddocks found")

    # Copy to figures directory
    print()
    print("Copying figures to docs/latex/figures/...")
    for fpath in generated_files:
        if fpath and fpath.exists():
            dest = FIGURES_DIR / fpath.name
            shutil.copy2(fpath, dest)
            print(f"  Copied: {dest.name}")

    # Summary
    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print()
    print(f"Generated {len(generated_files)} figures:")
    for fpath in generated_files:
        if fpath and fpath.exists():
            size_mb = fpath.stat().st_size / 1024 / 1024
            print(f"  {fpath.name} ({size_mb:.1f} MB)")
    print()
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")
    print()
    print("Windows analyzed:")
    for win_info, gdf, _ in all_data:
        print(f"  {win_info['name']:15s}  {win_info['dates']:25s}  {len(gdf):6d} cells")


if __name__ == "__main__":
    main()
