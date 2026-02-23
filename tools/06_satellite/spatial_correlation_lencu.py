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
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import Point
from scipy.spatial.distance import pdist, squareform
from libpysal.weights import DistanceBand
from esda.moran import Moran_BV, Moran_Local_BV

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
LISA_COLORS = {
    "HH": "#e74c3c",   # red
    "LH": "#9b59b6",   # purple
    "LL": "#3498db",   # blue
    "HL": "#e67e22",   # orange
    "NS": "#d5d5d5",   # light gray
}

QUADRANT_MAP = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}


def bivariate_local_lisa(df, w):
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
# 4. Cross-variogram (manual computation)
# ---------------------------------------------------------------------------
def compute_cross_variogram(coords, z1, z2, n_lags=15, maxlag=None):
    """
    Cross-semivariogram:
      gamma(h) = 1/(2*N(h)) * sum[ (z1(i)-z1(j)) * (z2(i)-z2(j)) ]
    for all pairs (i,j) within lag bin h.
    """
    dist_matrix = squareform(pdist(coords))
    if maxlag is None:
        maxlag = np.median(dist_matrix[dist_matrix > 0])

    lag_edges = np.linspace(0, maxlag, n_lags + 1)
    lag_centers = (lag_edges[:-1] + lag_edges[1:]) / 2
    gammas = np.full(n_lags, np.nan)
    counts = np.zeros(n_lags, dtype=int)

    n = len(z1)
    for k in range(n_lags):
        lo, hi = lag_edges[k], lag_edges[k + 1]
        mask = (dist_matrix > lo) & (dist_matrix <= hi)
        # upper triangle only to avoid double counting
        mask = np.triu(mask, k=1)
        idx = np.where(mask)
        npairs = len(idx[0])
        counts[k] = npairs
        if npairs > 0:
            diffs = (z1[idx[0]] - z1[idx[1]]) * (z2[idx[0]] - z2[idx[1]])
            gammas[k] = np.sum(diffs) / (2.0 * npairs)

    return lag_centers, gammas, counts


def cross_variogram_analysis(df):
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

    # Cross-variograms
    lags_ambel, gamma_ambel, cnt_ambel = compute_cross_variogram(coords, pc1, ambel, maxlag=maxlag)
    lags_lencu, gamma_lencu, cnt_lencu = compute_cross_variogram(coords, pc1, lencu, maxlag=maxlag)
    # Auto-variogram of PC1
    lags_pc1, gamma_pc1, cnt_pc1 = compute_cross_variogram(coords, pc1, pc1, maxlag=maxlag)

    # Find approximate range (where 90% of sill is reached)
    def approx_range(lags, gammas):
        valid = ~np.isnan(gammas)
        if not valid.any():
            return np.nan
        sill = np.nanmax(np.abs(gammas))
        for i, g in enumerate(gammas):
            if not np.isnan(g) and np.abs(g) >= 0.9 * sill:
                return lags[i]
        return lags[valid][-1]

    range_ambel = approx_range(lags_ambel, gamma_ambel)
    range_lencu = approx_range(lags_lencu, gamma_lencu)
    range_pc1 = approx_range(lags_pc1, gamma_pc1)

    print(f"  PC1 auto-variogram approx. range: {range_pc1:.1f}m")
    print(f"  PC1 x AMBEL cross-variogram approx. range: {range_ambel:.1f}m")
    print(f"  PC1 x LENCU cross-variogram approx. range: {range_lencu:.1f}m")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: cross-variograms
    ax = axes[0]
    valid_a = ~np.isnan(gamma_ambel)
    valid_l = ~np.isnan(gamma_lencu)
    ax.plot(lags_ambel[valid_a], gamma_ambel[valid_a], "o-", color="#e74c3c", linewidth=2, markersize=6, label="PC1 x AMBEL")
    ax.plot(lags_lencu[valid_l], gamma_lencu[valid_l], "s-", color="#2ecc71", linewidth=2, markersize=6, label="PC1 x LENCU")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Lag distance (m)", fontsize=11)
    ax.set_ylabel("Cross-semivariance", fontsize=11)
    ax.set_title("Cross-variograms: PC1 vs Weed Species", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right panel: PC1 auto-variogram
    ax2 = axes[1]
    valid_p = ~np.isnan(gamma_pc1)
    ax2.plot(lags_pc1[valid_p], gamma_pc1[valid_p], "D-", color="#3498db", linewidth=2, markersize=6, label="PC1 auto-variogram")
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
    gdf_lisa = bivariate_local_lisa(df, w)
    vario_stats = cross_variogram_analysis(df)
    save_summary(moran_df, vario_stats)

    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
