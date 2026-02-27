#!/usr/bin/env python3
"""
Cross-Window Comparison of Presto Embeddings
=============================================

Compares 4 temporal windows of Presto satellite embeddings to understand
how the embedding space changes with different observation periods.

Windows:
  1. Growth     (Dec 2024 - Feb 2025, 3 months)
  2. Pre-season (Jun - Aug 2025, 2 months)  [winter/bare soil]
  3. Planting   (Sep - Nov 2025, 2 months)
  4. Standard   (Nov 2024 - Oct 2025, 12 months)

Analyses:
  1. Cosine similarity between matched cells across windows
  2. Combined PCA space comparison
  3. Cluster agreement (ARI, NMI) for K=20
  4. Temporal sensitivity by paddock
  5. Unique information per window (PCA cross-correlation)

Output: plots and summary text in
  outputs/presto_cross_window_comparison/

Author: Claude Code
Date: 2026-02-06
"""

import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---- Portable path resolution ----
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

# TODO: This script is entirely bespoke analysis code (cosine similarity,
# combined PCA, cluster agreement ARI/NMI, temporal sensitivity, PCA
# cross-correlation). No direct library equivalents exist yet.
# Future candidates for ragweed_toolkit.satellite.comparison module:
#   - row_cosine_similarities -> ragweed_toolkit.embeddings.similarity
#   - match_datasets -> ragweed_toolkit.satellite.presto.match_cells
#   - analysis_3_cluster_agreement -> ragweed_toolkit.spatial.cluster_agreement

# ── Configuration ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path("/home/malezainia1/dev/INIA_DeepLearning_Ubuntu_mod_lleon")
EMBED_DIR = Path("/media/malezainia1/LORENZO/outputs_sugal_25-26/satellite_embeddings")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "presto_cross_window_comparison"

WINDOW_FILES = {
    "Growth\n(Dec24-Feb25)": EMBED_DIR / "embeddings_24-25_presto_growth_complete.gpkg",
    "Pre-season\n(Jun-Aug25)": EMBED_DIR / "embeddings_25-26_presto_pre_season_complete.gpkg",
    "Planting\n(Sep-Nov25)": EMBED_DIR / "embeddings_25-26_presto_planting_complete.gpkg",
    "Standard 12m\n(Nov24-Oct25)": EMBED_DIR / "embeddings_25-26_presto_nov-oct.gpkg",
}

# Short labels for filenames and compact displays
SHORT_LABELS = {
    "Growth\n(Dec24-Feb25)": "growth",
    "Pre-season\n(Jun-Aug25)": "pre_season",
    "Planting\n(Sep-Nov25)": "planting",
    "Standard 12m\n(Nov24-Oct25)": "standard_12m",
}

# Embedding columns: A00 .. A127
EMBED_COLS = [f"A{i:02d}" for i in range(128)]

# Cluster column used across all files
CLUSTER_COL = "cluster_global_20"

# Plotting style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

# Consistent color palette for the 4 windows
WINDOW_COLORS = {
    "Growth\n(Dec24-Feb25)": "#E74C3C",
    "Pre-season\n(Jun-Aug25)": "#3498DB",
    "Planting\n(Sep-Nov25)": "#2ECC71",
    "Standard 12m\n(Nov24-Oct25)": "#9B59B6",
}

# ── Helper functions ───────────────────────────────────────────────────────

def load_all_windows():
    """Load all 4 GeoPackage files and return dict of GeoDataFrames."""
    datasets = {}
    for label, path in WINDOW_FILES.items():
        print(f"  Loading {SHORT_LABELS[label]:15s} ... ", end="", flush=True)
        t0 = time.time()
        gdf = gpd.read_file(path)
        dt = time.time() - t0
        print(f"{len(gdf):,} cells  ({dt:.1f}s)")
        datasets[label] = gdf
    return datasets


def get_embedding_matrix(gdf):
    """Extract the 128-dim embedding matrix as a numpy array."""
    return gdf[EMBED_COLS].values.astype(np.float64)


def match_datasets(gdf_a, gdf_b):
    """
    Match rows between two GeoDataFrames using cell_id + paddock.
    Returns aligned DataFrames (same rows in same order).
    """
    key_cols = ["cell_id", "paddock"]
    merged = gdf_a[key_cols].merge(gdf_b[key_cols], on=key_cols, how="inner")
    # Reindex both to matched keys
    a_indexed = gdf_a.set_index(key_cols).loc[
        merged.set_index(key_cols).index
    ].reset_index()
    b_indexed = gdf_b.set_index(key_cols).loc[
        merged.set_index(key_cols).index
    ].reset_index()
    return a_indexed, b_indexed


def row_cosine_similarities(mat_a, mat_b):
    """Compute per-row cosine similarity between two aligned matrices."""
    # Normalize rows
    norm_a = np.linalg.norm(mat_a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(mat_b, axis=1, keepdims=True)
    # Avoid division by zero
    norm_a = np.where(norm_a == 0, 1, norm_a)
    norm_b = np.where(norm_b == 0, 1, norm_b)
    dot = np.sum((mat_a / norm_a) * (mat_b / norm_b), axis=1)
    return dot


# ── Analysis 1: Embedding Space Comparison ─────────────────────────────────

def analysis_1_cosine_similarity(datasets):
    """Compute pairwise cosine similarity between all window pairs."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Embedding Space Comparison (Cosine Similarity)")
    print("=" * 70)

    window_names = list(datasets.keys())
    n = len(window_names)
    pairs = list(combinations(range(n), 2))

    # Store results
    sim_matrix = np.ones((n, n))  # diagonal = 1
    pair_results = {}  # (i,j) -> {overall, per_paddock}

    for i, j in pairs:
        w_a, w_b = window_names[i], window_names[j]
        short_a, short_b = SHORT_LABELS[w_a], SHORT_LABELS[w_b]
        print(f"\n  Comparing {short_a} vs {short_b} ... ", end="", flush=True)

        gdf_a, gdf_b = match_datasets(datasets[w_a], datasets[w_b])
        mat_a = get_embedding_matrix(gdf_a)
        mat_b = get_embedding_matrix(gdf_b)
        sims = row_cosine_similarities(mat_a, mat_b)

        overall_mean = np.mean(sims)
        sim_matrix[i, j] = overall_mean
        sim_matrix[j, i] = overall_mean

        # Per-paddock breakdown
        paddocks = sorted(gdf_a["paddock"].unique())
        per_paddock = {}
        for p in paddocks:
            mask = gdf_a["paddock"].values == p
            if mask.sum() > 0:
                per_paddock[p] = {
                    "mean": float(np.mean(sims[mask])),
                    "std": float(np.std(sims[mask])),
                    "median": float(np.median(sims[mask])),
                    "min": float(np.min(sims[mask])),
                    "n": int(mask.sum()),
                }

        pair_results[(i, j)] = {
            "overall_mean": overall_mean,
            "overall_std": float(np.std(sims)),
            "overall_median": float(np.median(sims)),
            "per_paddock": per_paddock,
            "sims": sims,
            "paddock_labels": gdf_a["paddock"].values,
        }
        print(f"mean={overall_mean:.4f}, std={np.std(sims):.4f}, "
              f"matched={len(sims):,} cells")

    # ── Plot 1a: Heatmap of mean cosine similarity ──
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    # Annotate cells
    for ii in range(n):
        for jj in range(n):
            color = "white" if sim_matrix[ii, jj] < 0.5 else "black"
            ax.text(jj, ii, f"{sim_matrix[ii, jj]:.3f}",
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color=color)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([SHORT_LABELS[w] for w in window_names], fontsize=10)
    ax.set_yticklabels([SHORT_LABELS[w] for w in window_names], fontsize=10)
    ax.set_title("Mean Cosine Similarity Between Presto Windows\n"
                 "(1.0 = identical embeddings, 0.0 = orthogonal)", fontsize=13)
    plt.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_cosine_similarity_heatmap.png")
    plt.close(fig)
    print("\n  Saved: 01_cosine_similarity_heatmap.png")

    # ── Plot 1b: Distribution of cosine similarities per pair ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        sims = pair_results[(i, j)]["sims"]
        short_a = SHORT_LABELS[window_names[i]]
        short_b = SHORT_LABELS[window_names[j]]
        ax.hist(sims, bins=80, color=WINDOW_COLORS[window_names[i]], alpha=0.7,
                edgecolor="white", linewidth=0.3)
        ax.axvline(np.mean(sims), color="red", linestyle="--", linewidth=2,
                   label=f"mean={np.mean(sims):.3f}")
        ax.axvline(np.median(sims), color="blue", linestyle=":", linewidth=2,
                   label=f"median={np.median(sims):.3f}")
        ax.set_title(f"{short_a} vs {short_b}", fontsize=11)
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Cell Count")
        ax.legend(fontsize=9)
        ax.set_xlim(-0.1, 1.05)
    fig.suptitle("Distribution of Per-Cell Cosine Similarities Between Windows",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_cosine_similarity_distributions.png")
    plt.close(fig)
    print("  Saved: 02_cosine_similarity_distributions.png")

    # ── Plot 1c: Per-paddock boxplots for each pair ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        sims = pair_results[(i, j)]["sims"]
        plabels = pair_results[(i, j)]["paddock_labels"]
        short_a = SHORT_LABELS[window_names[i]]
        short_b = SHORT_LABELS[window_names[j]]

        paddocks = sorted(set(plabels))
        data_per_paddock = [sims[plabels == p] for p in paddocks]
        bp = ax.boxplot(data_per_paddock, labels=paddocks, patch_artist=True,
                        showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(WINDOW_COLORS[window_names[i]])
            patch.set_alpha(0.6)
        ax.set_title(f"{short_a} vs {short_b}", fontsize=11)
        ax.set_ylabel("Cosine Similarity")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.set_ylim(-0.1, 1.05)
    fig.suptitle("Cosine Similarity by Paddock (Per Window Pair)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_cosine_similarity_by_paddock.png")
    plt.close(fig)
    print("  Saved: 03_cosine_similarity_by_paddock.png")

    return sim_matrix, pair_results


# ── Analysis 2: PCA Space Comparison ───────────────────────────────────────

def analysis_2_pca_comparison(datasets):
    """Project all windows into a shared PCA space and compare."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: PCA Space Comparison (Combined Projection)")
    print("=" * 70)

    window_names = list(datasets.keys())

    # Stack all embeddings and track window membership
    all_embeddings = []
    all_labels = []
    all_paddocks = []
    for w in window_names:
        mat = get_embedding_matrix(datasets[w])
        all_embeddings.append(mat)
        all_labels.extend([w] * len(mat))
        all_paddocks.extend(datasets[w]["paddock"].values)

    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)
    all_paddocks = np.array(all_paddocks)
    print(f"  Combined matrix: {all_embeddings.shape[0]:,} cells x "
          f"{all_embeddings.shape[1]} dims")

    # Fit PCA on combined data
    print("  Fitting PCA on combined space ... ", end="", flush=True)
    t0 = time.time()
    pca_combined = PCA(n_components=10)
    all_pcs = pca_combined.fit_transform(all_embeddings)
    dt = time.time() - t0
    print(f"done ({dt:.1f}s)")

    explained = pca_combined.explained_variance_ratio_
    print(f"  Variance explained by PC1..PC10: "
          f"{', '.join(f'{v:.1%}' for v in explained[:5])}, ...")
    print(f"  Cumulative (PC1-3): {sum(explained[:3]):.1%}, "
          f"(PC1-5): {sum(explained[:5]):.1%}")

    # ── Plot 2a: PC1 vs PC2 colored by window ──
    fig, ax = plt.subplots(figsize=(12, 10))
    # Plot in reverse order so first window is on top
    for w in reversed(window_names):
        mask = all_labels == w
        ax.scatter(all_pcs[mask, 0], all_pcs[mask, 1],
                   c=WINDOW_COLORS[w], label=SHORT_LABELS[w],
                   alpha=0.15, s=3, rasterized=True)
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance)")
    ax.set_title("Combined PCA: All 4 Windows Projected Together\n"
                 "(each dot = one 10m cell)", fontsize=13)
    ax.legend(markerscale=5, fontsize=11, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_pca_combined_pc1_pc2.png")
    plt.close(fig)
    print("  Saved: 04_pca_combined_pc1_pc2.png")

    # ── Plot 2b: PC1 vs PC3 colored by window ──
    fig, ax = plt.subplots(figsize=(12, 10))
    for w in reversed(window_names):
        mask = all_labels == w
        ax.scatter(all_pcs[mask, 0], all_pcs[mask, 2],
                   c=WINDOW_COLORS[w], label=SHORT_LABELS[w],
                   alpha=0.15, s=3, rasterized=True)
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC3 ({explained[2]:.1%} variance)")
    ax.set_title("Combined PCA: PC1 vs PC3", fontsize=13)
    ax.legend(markerscale=5, fontsize=11, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_pca_combined_pc1_pc3.png")
    plt.close(fig)
    print("  Saved: 05_pca_combined_pc1_pc3.png")

    # ── Per-window variance in combined PCA space ──
    print("\n  Variance captured per window in combined PCA space:")
    window_var_stats = {}
    for w in window_names:
        mask = all_labels == w
        pcs_w = all_pcs[mask]
        total_var = np.var(pcs_w, axis=0).sum()
        pc1_var = np.var(pcs_w[:, 0])
        pc2_var = np.var(pcs_w[:, 1])
        window_var_stats[w] = {
            "total_var": total_var,
            "pc1_var": pc1_var,
            "pc2_var": pc2_var,
            "pc1_share": pc1_var / total_var,
            "pc2_share": pc2_var / total_var,
        }
        print(f"    {SHORT_LABELS[w]:15s}: total_var={total_var:.4f}  "
              f"PC1={pc1_var:.4f} ({pc1_var/total_var:.1%})  "
              f"PC2={pc2_var:.4f} ({pc2_var/total_var:.1%})")

    # ── Plot 2c: Variance explained bar chart ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Scree plot
    ax1.bar(range(1, 11), explained[:10] * 100, color="#3498DB", edgecolor="white")
    ax1.plot(range(1, 11), np.cumsum(explained[:10]) * 100, "r-o",
             linewidth=2, markersize=6)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("Combined PCA: Scree Plot")
    ax1.set_xticks(range(1, 11))

    # Per-window spread (total variance in combined PCA space)
    shorts = [SHORT_LABELS[w] for w in window_names]
    total_vars = [window_var_stats[w]["total_var"] for w in window_names]
    colors = [WINDOW_COLORS[w] for w in window_names]
    ax2.bar(shorts, total_vars, color=colors, edgecolor="white")
    ax2.set_ylabel("Total Variance in Combined PCA Space")
    ax2.set_title("Embedding Spread per Window\n(higher = more diverse)")
    ax2.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_pca_variance_analysis.png")
    plt.close(fig)
    print("  Saved: 06_pca_variance_analysis.png")

    # ── Plot 2d: Per-window density in PC1-PC2 (2x2 grid) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for idx, w in enumerate(window_names):
        ax = axes[idx]
        mask = all_labels == w
        pcs_w = all_pcs[mask]
        ax.scatter(pcs_w[:, 0], pcs_w[:, 1], c=WINDOW_COLORS[w],
                   alpha=0.1, s=2, rasterized=True)
        ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
        ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
        ax.set_title(f"{SHORT_LABELS[w]} ({len(pcs_w):,} cells)")
        # Set same limits for all panels
        ax.set_xlim(all_pcs[:, 0].min() - 0.5, all_pcs[:, 0].max() + 0.5)
        ax.set_ylim(all_pcs[:, 1].min() - 0.5, all_pcs[:, 1].max() + 0.5)
    fig.suptitle("PCA Distributions per Window (Same Axes)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_pca_per_window_panels.png")
    plt.close(fig)
    print("  Saved: 07_pca_per_window_panels.png")

    return pca_combined, all_pcs, all_labels, explained, window_var_stats


# ── Analysis 3: Cluster Agreement ──────────────────────────────────────────

def analysis_3_cluster_agreement(datasets):
    """Compute ARI and NMI for cluster_global_20 between all window pairs."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Cluster Agreement (ARI & NMI for K=20)")
    print("=" * 70)

    window_names = list(datasets.keys())
    n = len(window_names)
    ari_matrix = np.ones((n, n))
    nmi_matrix = np.ones((n, n))

    for i, j in combinations(range(n), 2):
        w_a, w_b = window_names[i], window_names[j]
        short_a, short_b = SHORT_LABELS[w_a], SHORT_LABELS[w_b]
        print(f"  {short_a} vs {short_b} ... ", end="", flush=True)

        gdf_a, gdf_b = match_datasets(datasets[w_a], datasets[w_b])

        if CLUSTER_COL not in gdf_a.columns or CLUSTER_COL not in gdf_b.columns:
            print(f"SKIP (missing {CLUSTER_COL})")
            ari_matrix[i, j] = ari_matrix[j, i] = np.nan
            nmi_matrix[i, j] = nmi_matrix[j, i] = np.nan
            continue

        labels_a = gdf_a[CLUSTER_COL].values
        labels_b = gdf_b[CLUSTER_COL].values

        # Drop rows where either is NaN
        valid = ~(pd.isna(labels_a) | pd.isna(labels_b))
        labels_a = labels_a[valid].astype(int)
        labels_b = labels_b[valid].astype(int)

        ari = adjusted_rand_score(labels_a, labels_b)
        nmi = normalized_mutual_info_score(labels_a, labels_b)

        ari_matrix[i, j] = ari_matrix[j, i] = ari
        nmi_matrix[i, j] = nmi_matrix[j, i] = nmi
        print(f"ARI={ari:.4f}, NMI={nmi:.4f} (n={len(labels_a):,})")

    # ── Plot 3a: ARI heatmap ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    short_names = [SHORT_LABELS[w] for w in window_names]

    for ax, matrix, title, cmap in [
        (ax1, ari_matrix, "Adjusted Rand Index (ARI)", "YlOrRd"),
        (ax2, nmi_matrix, "Normalized Mutual Information (NMI)", "YlGnBu"),
    ]:
        im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0)
        for ii in range(n):
            for jj in range(n):
                val = matrix[ii, jj]
                if np.isnan(val):
                    txt = "N/A"
                else:
                    txt = f"{val:.3f}"
                color = "white" if val < 0.5 else "black"
                ax.text(jj, ii, txt, ha="center", va="center",
                        fontsize=12, fontweight="bold", color=color)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_names, fontsize=10)
        ax.set_yticklabels(short_names, fontsize=10)
        ax.set_title(title, fontsize=13)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Cluster Agreement Between Windows (K=20 Clustering)\n"
                 "ARI: chance=0, perfect=1  |  NMI: no info=0, perfect=1",
                 fontsize=13, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_cluster_agreement_heatmaps.png")
    plt.close(fig)
    print("\n  Saved: 08_cluster_agreement_heatmaps.png")

    return ari_matrix, nmi_matrix


# ── Analysis 4: Temporal Sensitivity by Paddock ────────────────────────────

def analysis_4_temporal_sensitivity(datasets, pair_results):
    """Identify paddocks with high vs low temporal sensitivity."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Temporal Sensitivity by Paddock")
    print("=" * 70)

    window_names = list(datasets.keys())
    pairs = list(combinations(range(len(window_names)), 2))

    # Collect all paddocks across all pairs
    all_paddocks = set()
    for key in pair_results:
        all_paddocks.update(pair_results[key]["per_paddock"].keys())
    all_paddocks = sorted(all_paddocks)

    # Build paddock x pair matrix of mean cosine similarities
    paddock_pair_sims = {}
    for p in all_paddocks:
        paddock_pair_sims[p] = {}
        for key in pair_results:
            pp = pair_results[key]["per_paddock"]
            if p in pp:
                paddock_pair_sims[p][key] = pp[p]["mean"]

    # Compute temporal sensitivity = 1 - mean cosine similarity across all pairs
    # (Higher value = embeddings change more across windows = more sensitive)
    sensitivity = {}
    for p in all_paddocks:
        sims = list(paddock_pair_sims[p].values())
        if sims:
            sensitivity[p] = {
                "mean_sim": np.mean(sims),
                "min_sim": np.min(sims),
                "max_sim": np.max(sims),
                "std_sim": np.std(sims),
                "sensitivity": 1 - np.mean(sims),
                "n_pairs": len(sims),
            }

    # Sort by sensitivity (highest first)
    sorted_paddocks = sorted(sensitivity.keys(),
                             key=lambda p: sensitivity[p]["sensitivity"],
                             reverse=True)

    print("\n  Temporal Sensitivity Ranking (1 - mean cosine similarity):")
    print(f"  {'Paddock':<20s} {'Sensitivity':>12s} {'Mean Sim':>10s} "
          f"{'Min Sim':>10s} {'Max Sim':>10s}")
    print("  " + "-" * 64)
    for p in sorted_paddocks:
        s = sensitivity[p]
        print(f"  {p:<20s} {s['sensitivity']:>12.4f} {s['mean_sim']:>10.4f} "
              f"{s['min_sim']:>10.4f} {s['max_sim']:>10.4f}")

    # ── Plot 4a: Sensitivity bar chart ──
    fig, ax = plt.subplots(figsize=(14, 7))
    sens_vals = [sensitivity[p]["sensitivity"] for p in sorted_paddocks]
    mean_sims = [sensitivity[p]["mean_sim"] for p in sorted_paddocks]

    # Color bars by sensitivity level
    colors = []
    for s in sens_vals:
        if s > 0.5:
            colors.append("#E74C3C")  # High sensitivity (red)
        elif s > 0.3:
            colors.append("#F39C12")  # Medium (orange)
        else:
            colors.append("#2ECC71")  # Low sensitivity (green)

    bars = ax.bar(range(len(sorted_paddocks)), sens_vals, color=colors,
                  edgecolor="white", linewidth=0.5)

    # Add mean similarity text on bars
    for idx, (bar, ms) in enumerate(zip(bars, mean_sims)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"sim={ms:.2f}", ha="center", va="bottom", fontsize=8,
                rotation=45)

    ax.set_xticks(range(len(sorted_paddocks)))
    ax.set_xticklabels(sorted_paddocks, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Temporal Sensitivity\n(1 - mean cosine similarity)")
    ax.set_title("Temporal Sensitivity by Paddock\n"
                 "Red = HIGH (embeddings change a lot between windows)\n"
                 "Green = LOW (stable embeddings across windows)", fontsize=13)
    ax.set_ylim(0, max(sens_vals) * 1.2 if sens_vals else 1)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "09_temporal_sensitivity_by_paddock.png")
    plt.close(fig)
    print("\n  Saved: 09_temporal_sensitivity_by_paddock.png")

    # ── Plot 4b: Detailed per-pair per-paddock heatmap ──
    pair_labels = []
    for i, j in pairs:
        pair_labels.append(f"{SHORT_LABELS[window_names[i]]} vs\n"
                           f"{SHORT_LABELS[window_names[j]]}")

    sim_heatmap = np.full((len(all_paddocks), len(pairs)), np.nan)
    for pidx, p in enumerate(sorted_paddocks):
        for qidx, key in enumerate(pairs):
            if key in paddock_pair_sims[p]:
                sim_heatmap[pidx, qidx] = paddock_pair_sims[p][key]

    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.imshow(sim_heatmap, cmap="RdYlGn", aspect="auto",
                   vmin=0.0, vmax=1.0)
    # Annotate
    for ii in range(sim_heatmap.shape[0]):
        for jj in range(sim_heatmap.shape[1]):
            val = sim_heatmap[ii, jj]
            if not np.isnan(val):
                color = "white" if val < 0.4 else "black"
                ax.text(jj, ii, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pair_labels, fontsize=9)
    ax.set_yticks(range(len(sorted_paddocks)))
    ax.set_yticklabels(sorted_paddocks, fontsize=10)
    ax.set_title("Mean Cosine Similarity: Paddock x Window Pair\n"
                 "(green = stable, red = different)", fontsize=13)
    plt.colorbar(im, ax=ax, label="Mean Cosine Similarity", shrink=0.7)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "10_paddock_x_pair_heatmap.png")
    plt.close(fig)
    print("  Saved: 10_paddock_x_pair_heatmap.png")

    return sensitivity, sorted_paddocks


# ── Analysis 5: Unique Information per Window ──────────────────────────────

def analysis_5_unique_information(datasets):
    """Fit PCA per window, then correlate PCA components across windows."""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Unique Information per Window (PCA Cross-Correlation)")
    print("=" * 70)

    window_names = list(datasets.keys())
    n_components = 5  # Compare first 5 PCs per window

    # Fit PCA per window
    per_window_pca = {}
    per_window_components = {}
    per_window_explained = {}

    for w in window_names:
        mat = get_embedding_matrix(datasets[w])
        pca = PCA(n_components=n_components)
        pca.fit_transform(mat)
        per_window_pca[w] = pca
        per_window_components[w] = pca.components_  # shape (n_components, 128)
        per_window_explained[w] = pca.explained_variance_ratio_

        total = sum(pca.explained_variance_ratio_[:n_components])
        print(f"  {SHORT_LABELS[w]:15s}: PC1-5 explain "
              f"{total:.1%} of window variance")
        print(f"    Per-component: "
              f"{', '.join(f'{v:.1%}' for v in pca.explained_variance_ratio_[:n_components])}")

    # Cross-correlate PCA components between windows
    # For each pair of windows, compute |correlation| between their PC components
    print("\n  Cross-correlation of PCA loadings between windows:")
    cross_corr_results = {}

    for i, j in combinations(range(len(window_names)), 2):
        w_a, w_b = window_names[i], window_names[j]
        comp_a = per_window_components[w_a]  # (5, 128)
        comp_b = per_window_components[w_b]  # (5, 128)

        # Correlation matrix between PCs of window A and PCs of window B
        corr_mat = np.zeros((n_components, n_components))
        for ii in range(n_components):
            for jj in range(n_components):
                r, _ = pearsonr(comp_a[ii], comp_b[jj])
                corr_mat[ii, jj] = abs(r)

        cross_corr_results[(i, j)] = corr_mat
        # Maximum correlation for each PC in window A with any PC in window B
        corr_mat.max(axis=1)
        short_a, short_b = SHORT_LABELS[w_a], SHORT_LABELS[w_b]
        print(f"\n    {short_a} PCs -> best match in {short_b}:")
        for pc in range(n_components):
            best_match = corr_mat[pc].argmax()
            print(f"      PC{pc+1} -> PC{best_match+1} "
                  f"(|r|={corr_mat[pc, best_match]:.3f})")

    # ── Plot 5a: Cross-correlation matrices ──
    len(list(combinations(range(len(window_names)), 2)))
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for idx, (i, j) in enumerate(combinations(range(len(window_names)), 2)):
        ax = axes[idx]
        corr_mat = cross_corr_results[(i, j)]
        im = ax.imshow(corr_mat, cmap="YlOrRd", vmin=0, vmax=1)
        for ii in range(n_components):
            for jj in range(n_components):
                color = "white" if corr_mat[ii, jj] > 0.6 else "black"
                ax.text(jj, ii, f"{corr_mat[ii, jj]:.2f}",
                        ha="center", va="center", fontsize=9, color=color)
        ax.set_xticks(range(n_components))
        ax.set_yticks(range(n_components))
        ax.set_xticklabels([f"PC{k+1}" for k in range(n_components)], fontsize=9)
        ax.set_yticklabels([f"PC{k+1}" for k in range(n_components)], fontsize=9)
        short_a = SHORT_LABELS[window_names[i]]
        short_b = SHORT_LABELS[window_names[j]]
        ax.set_xlabel(short_b, fontsize=10)
        ax.set_ylabel(short_a, fontsize=10)
        ax.set_title(f"{short_a} vs {short_b}", fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("PCA Loading Cross-Correlation (|r|) Between Windows\n"
                 "High values = PCs capture same directions; "
                 "Low = unique information",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "11_pca_cross_correlation.png")
    plt.close(fig)
    print("\n  Saved: 11_pca_cross_correlation.png")

    # ── Compute uniqueness score per window ──
    # For each window, how much of its PC1 variance is NOT explained by any
    # other window's PCs? (1 - max |r| of PC1 with any PC of other windows)
    print("\n  Uniqueness of each window's principal components:")
    uniqueness_scores = {}
    for i, w in enumerate(window_names):
        max_corrs = []
        for j in range(len(window_names)):
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            if key in cross_corr_results:
                corr_mat = cross_corr_results[key]
                if i < j:
                    # Rows are window i's PCs
                    max_corrs.append(corr_mat[0].max())
                else:
                    # Columns are window i's PCs
                    max_corrs.append(corr_mat[:, 0].max())
        avg_max_corr_pc1 = np.mean(max_corrs) if max_corrs else 0
        uniqueness = 1 - avg_max_corr_pc1
        uniqueness_scores[w] = {
            "pc1_uniqueness": uniqueness,
            "pc1_avg_max_corr": avg_max_corr_pc1,
        }
        print(f"    {SHORT_LABELS[w]:15s}: PC1 uniqueness = {uniqueness:.3f} "
              f"(avg max |r| with other windows' PCs = {avg_max_corr_pc1:.3f})")

    # ── Plot 5b: Variance explained per window comparison ──
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_components)
    width = 0.2
    for idx, w in enumerate(window_names):
        offset = (idx - 1.5) * width
        evr = per_window_explained[w][:n_components]
        ax.bar(x + offset, evr * 100, width, label=SHORT_LABELS[w],
               color=WINDOW_COLORS[w], edgecolor="white")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("Per-Window PCA: Variance Explained by Top Components",
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{k+1}" for k in range(n_components)])
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "12_per_window_pca_variance.png")
    plt.close(fig)
    print("  Saved: 12_per_window_pca_variance.png")

    # ── Plot 5c: Uniqueness summary ──
    fig, ax = plt.subplots(figsize=(10, 6))
    shorts = [SHORT_LABELS[w] for w in window_names]
    unique_vals = [uniqueness_scores[w]["pc1_uniqueness"] for w in window_names]
    colors = [WINDOW_COLORS[w] for w in window_names]
    bars = ax.bar(shorts, unique_vals, color=colors, edgecolor="white")
    for bar, val in zip(bars, unique_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")
    ax.set_ylabel("PC1 Uniqueness Score\n(1 - avg max correlation with other windows)")
    ax.set_title("Unique Information per Window\n"
                 "Higher = captures directions not found in other windows",
                 fontsize=13)
    ax.set_ylim(0, max(unique_vals) * 1.3 if unique_vals else 1)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "13_window_uniqueness.png")
    plt.close(fig)
    print("  Saved: 13_window_uniqueness.png")

    return per_window_pca, per_window_explained, uniqueness_scores, cross_corr_results


# ── Summary Report ─────────────────────────────────────────────────────────

def write_summary(sim_matrix, pair_results, ari_matrix, nmi_matrix,
                  sensitivity, sorted_paddocks, explained,
                  window_var_stats, uniqueness_scores, datasets):
    """Write a text summary of all analyses."""
    window_names = list(datasets.keys())
    summary_path = OUTPUT_DIR / "SUMMARY.txt"

    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("PRESTO CROSS-WINDOW COMPARISON SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write("Generated: 2026-02-06\n")
        f.write(f"Number of windows: {len(window_names)}\n\n")

        f.write("WINDOWS:\n")
        for w in window_names:
            n_cells = len(datasets[w])
            n_paddocks = datasets[w]["paddock"].nunique()
            tw = datasets[w]["temporal_window"].iloc[0] if "temporal_window" in datasets[w].columns else "N/A"
            f.write(f"  {SHORT_LABELS[w]:15s}: {n_cells:,} cells, "
                    f"{n_paddocks} paddocks, window: {tw}\n")

        # Analysis 1
        f.write("\n" + "=" * 70 + "\n")
        f.write("ANALYSIS 1: COSINE SIMILARITY (Embedding Space)\n")
        f.write("=" * 70 + "\n\n")
        f.write("Mean cosine similarity matrix:\n\n")
        header = f"{'':15s}" + "  ".join(f"{SHORT_LABELS[w]:>12s}" for w in window_names)
        f.write(header + "\n")
        for i, w in enumerate(window_names):
            row = f"{SHORT_LABELS[w]:15s}"
            for j in range(len(window_names)):
                row += f"{sim_matrix[i, j]:>14.4f}"
            f.write(row + "\n")

        f.write("\nPairwise details:\n")
        for (i, j), res in pair_results.items():
            short_a = SHORT_LABELS[window_names[i]]
            short_b = SHORT_LABELS[window_names[j]]
            f.write(f"\n  {short_a} vs {short_b}:\n")
            f.write(f"    Mean: {res['overall_mean']:.4f}\n")
            f.write(f"    Std:  {res['overall_std']:.4f}\n")
            f.write(f"    Median: {res['overall_median']:.4f}\n")
            f.write(f"    Matched cells: {len(res['sims']):,}\n")

        # Analysis 3
        f.write("\n" + "=" * 70 + "\n")
        f.write("ANALYSIS 3: CLUSTER AGREEMENT (K=20)\n")
        f.write("=" * 70 + "\n\n")
        f.write("ARI matrix:\n\n")
        header = f"{'':15s}" + "  ".join(f"{SHORT_LABELS[w]:>12s}" for w in window_names)
        f.write(header + "\n")
        for i, w in enumerate(window_names):
            row = f"{SHORT_LABELS[w]:15s}"
            for j in range(len(window_names)):
                if np.isnan(ari_matrix[i, j]):
                    row += f"{'N/A':>14s}"
                else:
                    row += f"{ari_matrix[i, j]:>14.4f}"
            f.write(row + "\n")

        f.write("\nNMI matrix:\n\n")
        f.write(header + "\n")
        for i, w in enumerate(window_names):
            row = f"{SHORT_LABELS[w]:15s}"
            for j in range(len(window_names)):
                if np.isnan(nmi_matrix[i, j]):
                    row += f"{'N/A':>14s}"
                else:
                    row += f"{nmi_matrix[i, j]:>14.4f}"
            f.write(row + "\n")

        # Analysis 4
        f.write("\n" + "=" * 70 + "\n")
        f.write("ANALYSIS 4: TEMPORAL SENSITIVITY BY PADDOCK\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Paddock':<20s} {'Sensitivity':>12s} {'Mean Sim':>10s} "
                f"{'Min Sim':>10s} {'Max Sim':>10s} {'Std':>8s}\n")
        f.write("-" * 70 + "\n")
        for p in sorted_paddocks:
            s = sensitivity[p]
            f.write(f"{p:<20s} {s['sensitivity']:>12.4f} {s['mean_sim']:>10.4f} "
                    f"{s['min_sim']:>10.4f} {s['max_sim']:>10.4f} "
                    f"{s['std_sim']:>8.4f}\n")

        # Interpretation
        f.write("\nInterpretation:\n")
        high_sens = [p for p in sorted_paddocks
                     if sensitivity[p]["sensitivity"] > 0.5]
        low_sens = [p for p in sorted_paddocks
                    if sensitivity[p]["sensitivity"] < 0.3]
        f.write(f"  HIGH sensitivity (>0.5): {', '.join(high_sens) if high_sens else 'None'}\n")
        f.write("    -> Embeddings change significantly across windows.\n")
        f.write("    -> These paddocks benefit most from multi-window analysis.\n")
        f.write(f"  LOW sensitivity (<0.3): {', '.join(low_sens) if low_sens else 'None'}\n")
        f.write("    -> Stable embeddings; a single window may suffice.\n")

        # Analysis 2 & 5
        f.write("\n" + "=" * 70 + "\n")
        f.write("ANALYSIS 2 & 5: PCA & UNIQUENESS\n")
        f.write("=" * 70 + "\n\n")
        f.write("Combined PCA variance explained (PC1-5):\n")
        f.write(f"  {', '.join(f'PC{i+1}={v:.1%}' for i, v in enumerate(explained[:5]))}\n")
        f.write(f"  Cumulative (PC1-3): {sum(explained[:3]):.1%}\n")
        f.write(f"  Cumulative (PC1-5): {sum(explained[:5]):.1%}\n\n")

        f.write("Per-window spread in combined PCA:\n")
        for w in window_names:
            vs = window_var_stats[w]
            f.write(f"  {SHORT_LABELS[w]:15s}: total_var={vs['total_var']:.4f}\n")

        f.write("\nWindow uniqueness (PC1):\n")
        for w in window_names:
            u = uniqueness_scores[w]
            f.write(f"  {SHORT_LABELS[w]:15s}: uniqueness={u['pc1_uniqueness']:.3f} "
                    f"(avg_max_corr={u['pc1_avg_max_corr']:.3f})\n")

        # Key conclusions
        f.write("\n" + "=" * 70 + "\n")
        f.write("KEY CONCLUSIONS\n")
        f.write("=" * 70 + "\n\n")

        # Find most and least similar pair
        pairs = list(combinations(range(len(window_names)), 2))
        most_sim = max(pairs, key=lambda p: sim_matrix[p[0], p[1]])
        least_sim = min(pairs, key=lambda p: sim_matrix[p[0], p[1]])
        f.write(f"1. Most similar windows: "
                f"{SHORT_LABELS[window_names[most_sim[0]]]} & "
                f"{SHORT_LABELS[window_names[most_sim[1]]]} "
                f"(cosine={sim_matrix[most_sim[0], most_sim[1]]:.4f})\n\n")
        f.write(f"2. Most different windows: "
                f"{SHORT_LABELS[window_names[least_sim[0]]]} & "
                f"{SHORT_LABELS[window_names[least_sim[1]]]} "
                f"(cosine={sim_matrix[least_sim[0], least_sim[1]]:.4f})\n\n")

        # Most unique window
        most_unique = max(uniqueness_scores, key=lambda w: uniqueness_scores[w]["pc1_uniqueness"])
        f.write(f"3. Most unique window: {SHORT_LABELS[most_unique]} "
                f"(uniqueness={uniqueness_scores[most_unique]['pc1_uniqueness']:.3f})\n\n")

        # Most temporally sensitive paddock
        if sorted_paddocks:
            f.write(f"4. Most sensitive paddock: {sorted_paddocks[0]} "
                    f"(sensitivity={sensitivity[sorted_paddocks[0]]['sensitivity']:.4f})\n")
            f.write(f"   Least sensitive paddock: {sorted_paddocks[-1]} "
                    f"(sensitivity={sensitivity[sorted_paddocks[-1]]['sensitivity']:.4f})\n\n")

        f.write("=" * 70 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("=" * 70 + "\n\n")
        for fname in sorted(OUTPUT_DIR.glob("*.png")):
            f.write(f"  {fname.name}\n")
        f.write("  SUMMARY.txt\n")

    print(f"\n  Summary written to: {summary_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PRESTO CROSS-WINDOW COMPARISON")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check all files exist
    print("\nChecking input files:")
    for label, path in WINDOW_FILES.items():
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {SHORT_LABELS[label]:15s}: {path.name}")
        if not exists:
            print(f"\nERROR: File not found: {path}")
            sys.exit(1)

    # Load all datasets
    print("\nLoading datasets:")
    t0_total = time.time()
    datasets = load_all_windows()

    # Run analyses
    sim_matrix, pair_results = analysis_1_cosine_similarity(datasets)
    pca_combined, all_pcs, all_labels, explained, window_var_stats = \
        analysis_2_pca_comparison(datasets)
    ari_matrix, nmi_matrix = analysis_3_cluster_agreement(datasets)
    sensitivity, sorted_paddocks = \
        analysis_4_temporal_sensitivity(datasets, pair_results)
    per_window_pca, per_window_explained, uniqueness_scores, cross_corr = \
        analysis_5_unique_information(datasets)

    # Write summary
    print("\n" + "=" * 70)
    print("WRITING SUMMARY REPORT")
    print("=" * 70)
    write_summary(sim_matrix, pair_results, ari_matrix, nmi_matrix,
                  sensitivity, sorted_paddocks, explained,
                  window_var_stats, uniqueness_scores, datasets)

    dt_total = time.time() - t0_total
    print(f"\nTotal execution time: {dt_total:.1f}s")
    print(f"\nAll outputs saved to:\n  {OUTPUT_DIR}/")
    print("\nFiles generated:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:50s} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
