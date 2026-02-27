#!/usr/bin/env python3
"""
Spatial correlation analysis between Presto PCA embeddings and weed species counts.

Analyses:
  1. Bivariate Global Moran's I
  2. Bivariate Local LISA (with GeoPackage export)
  3. Cross-variograms (manual computation with scipy)
  4. Summary statistics table

Output: outputs/lencu_presto/spatial_analysis/
"""

import sys
import warnings

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from esda.moran import Moran_BV, Moran_Local_BV
from libpysal.weights import DistanceBand
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist
from shapely.geometry import Point

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---- Portable path resolution ----
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

# --- Library imports (replacing internal duplicates) ---
from ragweed_toolkit.spatial import (
    LISA_COLORS,
    QUADRANT_MAP,
)
from ragweed_toolkit.spatial import (
    compute_cross_variogram as _lib_compute_cross_variogram,
)

# Note: build_distance_weights in the library takes a GeoDataFrame, while
# this script's build_weights takes a DataFrame with x/y columns directly.
# Keeping the script-specific version for backward compatibility.

# Note: bivariate_global_moran uses esda's Moran_BV directly (similar to
# library's compute_bivariate_morans, but the library version takes a
# GeoDataFrame + column names and returns BivariateMoranResult dataclass).
# Keeping script-specific version since it processes multiple pairs at once.

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = "/home/malezainia1/dev/INIA_DeepLearning_Ubuntu_mod_lleon/outputs/lencu_presto/analysis_results.csv"
OUT_DIR = "/home/malezainia1/dev/INIA_DeepLearning_Ubuntu_mod_lleon/outputs/lencu_presto/spatial_analysis"

import os

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 0. Load and filter data
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df["valid_mask"] == 1.0].copy()
    df.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(df)} valid points")
    return df


# ---------------------------------------------------------------------------
# 1. Spatial weights
# ---------------------------------------------------------------------------
def build_weights(df, threshold=15.0):
    """Build DistanceBand weight matrix. Increase threshold if islands exist."""
    coords = list(zip(df["pixel_x_utm"], df["pixel_y_utm"]))
    w = DistanceBand(coords, threshold=threshold, binary=True, silence_warnings=True)
    w.transform = "r"  # row-standardize
    n_islands = len(w.islands)
    if n_islands > 0:
        print(f"  WARNING: {n_islands} islands at threshold={threshold}m, increasing to {threshold + 5}m")
        return build_weights(df, threshold=threshold + 5)
    print(f"  Weight matrix: threshold={threshold}m, mean neighbours={w.mean_neighbors:.1f}, islands={n_islands}")
    return w


# ---------------------------------------------------------------------------
# 2. Bivariate Global Moran's I
# ---------------------------------------------------------------------------
def bivariate_global_moran(df, w):
    print("\n" + "=" * 70)
    print("BIVARIATE GLOBAL MORAN'S I")
    print("=" * 70)

    pairs = [
        ("PC1", "AMBEL"),
        ("PC1", "LENCU"),
        ("PC2", "AMBEL"),
        ("PC2", "LENCU"),
        ("PC5", "LENCU"),
    ]

    results = []
    for x_col, y_col in pairs:
        x = df[x_col].values
        y = df[y_col].values
        bv = Moran_BV(x, y, w, permutations=999)
        sig = "***" if bv.p_sim <= 0.001 else "**" if bv.p_sim < 0.01 else "*" if bv.p_sim < 0.05 else "ns"
        results.append({
            "Variable_X": x_col,
            "Variable_Y": y_col,
            "Morans_I": round(bv.I, 4),
            "p_sim": round(bv.p_sim, 4),
            "Significance": sig,
        })

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    return res_df


# ---------------------------------------------------------------------------
# 3. Bivariate Local LISA
# ---------------------------------------------------------------------------
def bivariate_local_lisa(df, w):
    """Compute bivariate local LISA using library constants for colors/quadrants."""
    print("\n" + "=" * 70)
    print("BIVARIATE LOCAL LISA")
    print("=" * 70)

    pairs = [("PC1", "AMBEL"), ("PC1", "LENCU")]
    gdf_all = gpd.GeoDataFrame(
        df[["pixel_x_utm", "pixel_y_utm", "PC1", "AMBEL", "LENCU"]].copy(),
        geometry=[Point(x, y) for x, y in zip(df["pixel_x_utm"], df["pixel_y_utm"])],
        crs="EPSG:32719",
    )

    lisa_results = {}
    for x_col, y_col in pairs:
        x = df[x_col].values
        y = df[y_col].values
        lm = Moran_Local_BV(x, y, w, permutations=999)

        labels = []
        for i in range(len(df)):
            if lm.p_sim[i] < 0.05:
                labels.append(QUADRANT_MAP.get(lm.q[i], "NS"))
            else:
                labels.append("NS")

        col_name = f"LISA_{x_col}_{y_col}"
        gdf_all[col_name] = labels
        gdf_all[f"Ii_{x_col}_{y_col}"] = lm.Is
        gdf_all[f"p_{x_col}_{y_col}"] = lm.p_sim

        counts = pd.Series(labels).value_counts()
        sig_count = sum(1 for l in labels if l != "NS")
        print(f"\n  {x_col} vs {y_col}: {sig_count} significant clusters ({sig_count/len(df)*100:.1f}%)")
        for cat in ["HH", "HL", "LH", "LL", "NS"]:
            print(f"    {cat}: {counts.get(cat, 0)}")
        lisa_results[(x_col, y_col)] = labels

    # Save GeoPackage
    gpkg_path = os.path.join(OUT_DIR, "bivariate_lisa_clusters.gpkg")
    gdf_all.to_file(gpkg_path, driver="GPKG")
    print(f"\n  Saved: {gpkg_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (pair, labels) in zip(axes, lisa_results.items()):
        x_col, y_col = pair
        colors = [LISA_COLORS[l] for l in labels]
        ax.scatter(
            df["pixel_x_utm"], df["pixel_y_utm"],
            c=colors, s=35, alpha=0.85, edgecolors="k", linewidths=0.3,
        )
        ax.set_title(f"Bivariate LISA: {x_col} vs {y_col}", fontsize=13, fontweight="bold")
        ax.set_xlabel("UTM Easting (m)")
        ax.set_ylabel("UTM Northing (m)")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=9)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LISA_COLORS["HH"], markersize=9, label="HH (High-High)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LISA_COLORS["HL"], markersize=9, label="HL (High-Low)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LISA_COLORS["LH"], markersize=9, label="LH (Low-High)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LISA_COLORS["LL"], markersize=9, label="LL (Low-Low)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LISA_COLORS["NS"], markersize=9, label="Not significant"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig_path = os.path.join(OUT_DIR, "bivariate_lisa_map.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    return gdf_all


# ---------------------------------------------------------------------------
# 4. Cross-variogram (uses library function)
# ---------------------------------------------------------------------------
def cross_variogram_analysis(df):
    """Compute cross-variograms using the library's compute_cross_variogram."""
    print("\n" + "=" * 70)
    print("CROSS-VARIOGRAMS")
    print("=" * 70)

    coords = df[["pixel_x_utm", "pixel_y_utm"]].values
    pc1 = df["PC1"].values
    ambel = df["AMBEL"].values
    lencu = df["LENCU"].values

    dist_flat = pdist(coords)
    maxlag = np.median(dist_flat)
    print(f"  Median pairwise distance: {maxlag:.1f}m (used as maxlag)")

    # Cross-variograms using library function
    result_ambel = _lib_compute_cross_variogram(coords, pc1, ambel, max_lag=maxlag)
    result_lencu = _lib_compute_cross_variogram(coords, pc1, lencu, max_lag=maxlag)
    # Auto-variogram of PC1
    result_pc1 = _lib_compute_cross_variogram(coords, pc1, pc1, max_lag=maxlag)

    range_ambel = result_ambel.approx_range
    range_lencu = result_lencu.approx_range
    range_pc1 = result_pc1.approx_range

    print(f"  PC1 auto-variogram approx. range: {range_pc1:.1f}m")
    print(f"  PC1 x AMBEL cross-variogram approx. range: {range_ambel:.1f}m")
    print(f"  PC1 x LENCU cross-variogram approx. range: {range_lencu:.1f}m")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: cross-variograms
    ax = axes[0]
    valid_a = ~np.isnan(result_ambel.gamma)
    valid_l = ~np.isnan(result_lencu.gamma)
    ax.plot(result_ambel.lag_centers[valid_a], result_ambel.gamma[valid_a], "o-", color="#e74c3c", linewidth=2, markersize=6, label="PC1 x AMBEL")
    ax.plot(result_lencu.lag_centers[valid_l], result_lencu.gamma[valid_l], "s-", color="#2ecc71", linewidth=2, markersize=6, label="PC1 x LENCU")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Lag distance (m)", fontsize=11)
    ax.set_ylabel("Cross-semivariance", fontsize=11)
    ax.set_title("Cross-variograms: PC1 vs Weed Species", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right panel: PC1 auto-variogram
    ax2 = axes[1]
    valid_p = ~np.isnan(result_pc1.gamma)
    ax2.plot(result_pc1.lag_centers[valid_p], result_pc1.gamma[valid_p], "D-", color="#3498db", linewidth=2, markersize=6, label="PC1 auto-variogram")
    ax2.set_xlabel("Lag distance (m)", fontsize=11)
    ax2.set_ylabel("Semivariance", fontsize=11)
    ax2.set_title("PC1 Auto-variogram (reference)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "cross_variograms.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    return {
        "PC1_auto_range_m": round(range_pc1, 1),
        "PC1xAMBEL_range_m": round(range_ambel, 1),
        "PC1xLENCU_range_m": round(range_lencu, 1),
    }


# ---------------------------------------------------------------------------
# 5. Summary table
# ---------------------------------------------------------------------------
def save_summary(moran_df, vario_stats):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Add variogram info as extra rows
    vario_rows = pd.DataFrame([
        {"Variable_X": "PC1", "Variable_Y": "PC1 (auto)", "Morans_I": np.nan,
         "p_sim": np.nan, "Significance": f"range={vario_stats['PC1_auto_range_m']}m"},
        {"Variable_X": "PC1", "Variable_Y": "AMBEL (cross-vario)", "Morans_I": np.nan,
         "p_sim": np.nan, "Significance": f"range={vario_stats['PC1xAMBEL_range_m']}m"},
        {"Variable_X": "PC1", "Variable_Y": "LENCU (cross-vario)", "Morans_I": np.nan,
         "p_sim": np.nan, "Significance": f"range={vario_stats['PC1xLENCU_range_m']}m"},
    ])

    summary = pd.concat([moran_df, vario_rows], ignore_index=True)
    csv_path = os.path.join(OUT_DIR, "correlation_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("SPATIAL CORRELATION ANALYSIS: Presto PCA vs Weed Counts")
    print("=" * 70)

    df = load_data()

    print("\nBuilding spatial weight matrix...")
    w = build_weights(df)

    moran_df = bivariate_global_moran(df, w)
    bivariate_local_lisa(df, w)
    vario_stats = cross_variogram_analysis(df)
    save_summary(moran_df, vario_stats)

    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
