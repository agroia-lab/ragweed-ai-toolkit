#!/usr/bin/env python3
"""
Orthomosaic Domain Shift Analysis (MMD)
========================================

Computes domain shift between orthomosaic tile embeddings and existing
ragweed training database embeddings using Maximum Mean Discrepancy (MMD).

This implements the deployment gate framework from the ragweed domain shift
analysis (malezainia2/resultados/05_domain_shift_generalization_analysis.md):

    MMD < 0.15  -> Deploy directly, minimal domain shift
    0.15 - 0.30 -> Deploy with augmentation recommended
    0.30 - 0.45 -> Active learning required
    > 0.45      -> Collect substantial new training data

The script loads tile embeddings (from embed_ortho_tiles.py, Phase 1) and
reference ragweed database embeddings (9 databases, 5338 images, ResNet50
2048-dim), then computes MMD between the tile distribution and each reference
database.

Additionally, it ranks individual tiles by their distance to the reference
centroid (most distant = most novel = best candidates for active learning).

Usage:
    python scripts/cli/compute_ortho_domain_shift.py \\
        --tile-embeddings /path/to/ortho_embeddings/embeddings.npz \\
        --reference-embeddings /path/to/ragweed_embeddings/embeddings.npz \\
        --output /path/to/domain_shift_report/ \\
        --device cuda:0

    # With subsampling for faster computation
    python scripts/cli/compute_ortho_domain_shift.py \\
        --tile-embeddings /path/to/embeddings.npz \\
        --reference-embeddings /path/to/ragweed_embeddings/embeddings.npz \\
        --output /path/to/domain_shift_report/ \\
        --subsample 500

Output:
    domain_shift_report.json   - Full results: MMD per database, recommendation
    tile_distances.csv         - Per-tile distance ranking
    joint_umap.html            - Interactive: tiles + reference databases
    joint_umap.png             - Static version
    mmd_comparison.png         - Bar chart: MMD by database
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Short display names for the 9 reference databases
SHORT_NAMES = {
    "CL_Seba_Labelled": "CL_Seba",
    "01_NorthDakota_Individual": "ND_Indiv",
    "03_Michigan_3Season": "MI_3Season",
    "CL_Alberto_Labelled": "CL_Alberto",
    "07_WeedCrop_PrecAg": "WeedCrop",
    "01_NorthDakota_Aerial": "ND_Aerial",
    "05_Purdue_4Weed": "Purdue",
    "CL_StaRosa_Unlabelled": "CL_StaRosa",
    "06_USDA_WeedCube": "WeedCube",
}

# Deployment gate thresholds (from domain shift analysis)
DEPLOYMENT_THRESHOLDS = [
    (0.15, "Deploy directly - minimal domain shift"),
    (0.30, "Deploy with augmentation recommended"),
    (0.45, "Active learning required"),
    (float("inf"), "Collect substantial new training data"),
]


# ============================================================================
# MMD COMPUTATION
# ============================================================================


def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> float:
    """
    Compute MMD (Maximum Mean Discrepancy) between two sets of embeddings.

    Uses a Gaussian RBF kernel with the median heuristic for bandwidth
    selection (if gamma not provided). Returns MMD (not squared).

    Args:
        X: First set of embeddings, shape (n, d)
        Y: Second set of embeddings, shape (m, d)
        gamma: RBF kernel bandwidth. If None, uses median heuristic.

    Returns:
        MMD value (float, >= 0). Higher means more different distributions.
    """
    from sklearn.metrics.pairwise import rbf_kernel

    if gamma is None:
        from sklearn.metrics.pairwise import euclidean_distances

        XY = np.vstack([X, Y])
        dists = euclidean_distances(XY, XY)
        median_dist = np.median(dists[dists > 0])
        gamma = 1.0 / (2 * median_dist ** 2)

    n = len(X)
    m = len(Y)

    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)

    # Unbiased MMD^2 estimator
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)

    mmd2 = (
        Kxx.sum() / (n * (n - 1))
        + Kyy.sum() / (m * (m - 1))
        - 2 * Kxy.sum() / (n * m)
    )

    return max(0, mmd2) ** 0.5  # Return MMD (not squared)


def compute_mmd_with_permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: Optional[float] = None,
    n_permutations: int = 1000,
) -> Tuple[float, float]:
    """
    Compute MMD with a permutation test for statistical significance.

    Under H0 (same distribution), the MMD is computed on random shuffles
    of the combined data. The p-value is the fraction of permuted MMDs
    that are >= the observed MMD.

    Args:
        X: First set of embeddings, shape (n, d)
        Y: Second set of embeddings, shape (m, d)
        gamma: RBF kernel bandwidth
        n_permutations: Number of permutations for the significance test

    Returns:
        Tuple of (mmd_value, p_value)
    """
    from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel

    # Compute gamma once using combined data
    if gamma is None:
        XY = np.vstack([X, Y])
        dists = euclidean_distances(XY, XY)
        median_dist = np.median(dists[dists > 0])
        gamma = 1.0 / (2 * median_dist ** 2)

    # Observed MMD
    observed_mmd = compute_mmd(X, Y, gamma=gamma)

    # Permutation test
    combined = np.vstack([X, Y])
    n = len(X)
    total = len(combined)
    rng = np.random.RandomState(42)

    count_ge = 0
    for _ in range(n_permutations):
        perm = rng.permutation(total)
        X_perm = combined[perm[:n]]
        Y_perm = combined[perm[n:]]
        perm_mmd = compute_mmd(X_perm, Y_perm, gamma=gamma)
        if perm_mmd >= observed_mmd:
            count_ge += 1

    p_value = (count_ge + 1) / (n_permutations + 1)  # +1 for continuity correction

    return observed_mmd, p_value


def compute_tile_distances(
    tile_embeddings: np.ndarray, reference_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute L2 distance from each tile to the centroid of reference embeddings.

    Args:
        tile_embeddings: Tile embeddings, shape (n_tiles, d)
        reference_embeddings: Reference embeddings, shape (n_ref, d)

    Returns:
        Array of distances, shape (n_tiles,)
    """
    centroid = reference_embeddings.mean(axis=0)
    distances = np.linalg.norm(tile_embeddings - centroid, axis=1)
    return distances


def get_deployment_recommendation(mmd_value: float) -> Tuple[str, str]:
    """
    Get deployment recommendation based on MMD value.

    Returns:
        Tuple of (recommendation_text, risk_level)
    """
    for threshold, recommendation in DEPLOYMENT_THRESHOLDS:
        if mmd_value < threshold:
            if threshold == 0.15:
                return recommendation, "LOW"
            elif threshold == 0.30:
                return recommendation, "MODERATE"
            elif threshold == 0.45:
                return recommendation, "SIGNIFICANT"
            else:
                return recommendation, "HIGH"
    return DEPLOYMENT_THRESHOLDS[-1][1], "HIGH"


# ============================================================================
# LOADING
# ============================================================================


def load_tile_embeddings(
    npz_path: Path, subsample: int = 0
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Load tile embeddings from .npz file.

    Args:
        npz_path: Path to embeddings.npz
        subsample: If > 0, randomly subsample this many tiles

    Returns:
        Tuple of (embeddings array, metadata list)
    """
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    metadata = data["metadata"].tolist() if "metadata" in data else []

    if subsample > 0 and subsample < len(embeddings):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(embeddings), subsample, replace=False)
        indices.sort()
        embeddings = embeddings[indices]
        if metadata:
            metadata = [metadata[i] for i in indices]

    return embeddings, metadata


def load_reference_embeddings(
    ref_path: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Load reference embeddings from a single .npz file containing all databases.

    The ragweed embeddings file contains:
        - embeddings: (5338, 2048) array
        - metadata: (5338,) array of dicts with 'source_collection' key

    Args:
        ref_path: Path to embeddings.npz or directory containing it

    Returns:
        Tuple of (all_embeddings, source_labels, per_database_dict)
        where per_database_dict maps short_name -> embeddings array
    """
    # Handle directory or direct file path
    if ref_path.is_dir():
        npz_path = ref_path / "embeddings.npz"
    else:
        npz_path = ref_path

    if not npz_path.exists():
        print(f"ERROR: Reference embeddings not found: {npz_path}")
        sys.exit(1)

    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    metadata = data["metadata"]

    # Extract source collection labels
    sources = np.array([m["source_collection"] for m in metadata])

    # Split by database
    per_db = {}
    for source_key, short_name in SHORT_NAMES.items():
        mask = sources == source_key
        if mask.sum() > 0:
            per_db[short_name] = embeddings[mask]

    return embeddings, sources, per_db


# ============================================================================
# VISUALIZATION
# ============================================================================


def create_mmd_bar_chart(
    mmd_results: List[Dict], overall_mmd: float, output_path: Path
) -> None:
    """
    Create a bar chart comparing MMD values across reference databases.

    Args:
        mmd_results: List of dicts with 'database', 'mmd', 'n_images' keys
        overall_mmd: MMD against combined reference set
        output_path: Path to save the PNG
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available, skipping bar chart")
        return

    # Sort by MMD
    results_sorted = sorted(mmd_results, key=lambda x: x["mmd"])

    databases = [r["database"] for r in results_sorted]
    mmd_values = [r["mmd"] for r in results_sorted]
    n_images = [r["n_images"] for r in results_sorted]

    # Color by Chilean vs International
    chilean_dbs = {"CL_Seba", "CL_Alberto", "CL_StaRosa"}
    colors = ["#2171b5" if db in chilean_dbs else "#e6550d" for db in databases]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        databases, mmd_values, color=colors, edgecolor="white", linewidth=0.5, height=0.6
    )

    # Annotate bars with image counts
    for i, (mmd_val, n_img) in enumerate(zip(mmd_values, n_images)):
        ax.text(
            mmd_val + 0.008,
            i,
            f"MMD={mmd_val:.3f} (n={n_img})",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )

    # Threshold lines
    for threshold, label in DEPLOYMENT_THRESHOLDS[:-1]:
        ax.axvline(
            threshold, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7
        )
        ax.text(
            threshold,
            len(databases) - 0.3,
            f"{threshold}",
            fontsize=8,
            color="#666666",
            ha="center",
        )

    # Overall MMD line
    ax.axvline(
        overall_mmd, color="#c0392b", linestyle="-", linewidth=2, alpha=0.8
    )
    ax.text(
        overall_mmd + 0.01,
        -0.5,
        f"Overall MMD = {overall_mmd:.3f}",
        fontsize=10,
        color="#c0392b",
        fontweight="bold",
    )

    ax.set_xlabel("MMD (Maximum Mean Discrepancy)", fontsize=12)
    ax.set_title(
        "Domain Shift: Orthomosaic Tiles vs Reference Databases",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlim(0, max(mmd_values) * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color="#2171b5", label="Chilean"),
        mpatches.Patch(color="#e6550d", label="International"),
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def create_joint_umap(
    tile_embeddings: np.ndarray,
    tile_metadata: List[Dict],
    ref_embeddings: np.ndarray,
    ref_sources: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Create a joint UMAP projection of tile embeddings and reference databases.

    Projects all embeddings into a single 2D space so tiles can be visually
    compared with the existing 9 ragweed databases.

    Args:
        tile_embeddings: Tile embeddings, shape (n_tiles, d)
        tile_metadata: Metadata list for tiles
        ref_embeddings: Reference embeddings, shape (n_ref, d)
        ref_sources: Source labels for reference embeddings
        output_dir: Directory to save UMAP outputs
    """
    try:
        import umap
    except ImportError:
        print("  WARNING: umap-learn not available, skipping UMAP visualization")
        print("  Install with: pip install umap-learn")
        return

    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("  WARNING: scikit-learn not available, skipping UMAP visualization")
        return

    n_tiles = len(tile_embeddings)
    n_ref = len(ref_embeddings)

    # Combine all embeddings
    all_embeddings = np.vstack([tile_embeddings, ref_embeddings])

    # L2 normalize
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    all_normed = all_embeddings / norms

    # Pre-PCA to 50 dims for speed
    print("  Pre-PCA: 2048 -> 50 dimensions...")
    pca = PCA(n_components=min(50, all_normed.shape[1], all_normed.shape[0] - 1), random_state=42)
    all_pca = pca.fit_transform(all_normed)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    # UMAP projection
    print("  UMAP 2D projection...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        verbose=False,
    )
    coords_2d = reducer.fit_transform(all_pca)

    # Build labels
    tile_labels = ["Ortho Tiles"] * n_tiles
    ref_labels = []
    for src in ref_sources:
        short = SHORT_NAMES.get(src, src)
        ref_labels.append(short)

    all_labels = tile_labels + ref_labels

    # Build DataFrame
    df = pd.DataFrame(
        {
            "umap_x": coords_2d[:, 0],
            "umap_y": coords_2d[:, 1],
            "source": all_labels,
            "type": (["tile"] * n_tiles) + (["reference"] * n_ref),
        }
    )

    # Add tile filenames if available
    filenames = []
    for i in range(n_tiles):
        if tile_metadata and i < len(tile_metadata):
            meta = tile_metadata[i]
            if isinstance(meta, dict):
                filenames.append(meta.get("filename", f"tile_{i}"))
            else:
                filenames.append(f"tile_{i}")
        else:
            filenames.append(f"tile_{i}")

    for i in range(n_ref):
        filenames.append("")

    df["filename"] = filenames

    # Save CSV
    csv_path = output_dir / "joint_umap_coords.csv"
    df.to_csv(csv_path, index=False)

    # Create interactive HTML visualization
    try:
        import plotly.express as px

        # Color palette: distinct color for tiles, standard palette for databases
        palette = {
            "Ortho Tiles": "#000000",
            "CL_Seba": "#2171b5",
            "CL_Alberto": "#6baed6",
            "CL_StaRosa": "#08519c",
            "ND_Indiv": "#e6550d",
            "ND_Aerial": "#fd8d3c",
            "MI_3Season": "#31a354",
            "WeedCrop": "#756bb1",
            "Purdue": "#de2d26",
            "WeedCube": "#636363",
        }

        # Tiles get larger marker, references smaller
        fig = px.scatter(
            df,
            x="umap_x",
            y="umap_y",
            color="source",
            color_discrete_map=palette,
            hover_data=["filename", "type"],
            title="Joint UMAP: Orthomosaic Tiles + Reference Databases",
            opacity=0.5,
        )

        # Adjust marker sizes: tiles larger
        for trace in fig.data:
            if trace.name == "Ortho Tiles":
                trace.marker.size = 8
                trace.marker.symbol = "diamond"
                trace.marker.opacity = 0.7
            else:
                trace.marker.size = 4
                trace.marker.opacity = 0.35

        fig.update_layout(
            template="plotly_white",
            width=1200,
            height=800,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        )

        html_path = output_dir / "joint_umap.html"
        fig.write_html(str(html_path))
        print(f"  Saved: {html_path}")

        # Static PNG
        try:
            png_path = output_dir / "joint_umap.png"
            fig.write_image(str(png_path), scale=2)
            print(f"  Saved: {png_path}")
        except Exception:
            # Try matplotlib fallback
            _create_joint_umap_static(df, palette, output_dir / "joint_umap.png")

    except ImportError:
        print("  WARNING: plotly not available, creating static plot only")
        _create_joint_umap_static(df, {}, output_dir / "joint_umap.png")


def _create_joint_umap_static(
    df: pd.DataFrame, palette: Dict[str, str], output_path: Path
) -> None:
    """Matplotlib fallback for joint UMAP visualization."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available, skipping static UMAP plot")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    sources = df["source"].unique()
    default_colors = plt.cm.tab10(np.linspace(0, 1, len(sources)))

    for i, src in enumerate(sources):
        mask = df["source"] == src
        sub = df[mask]
        color = palette.get(src, default_colors[i])
        marker = "D" if src == "Ortho Tiles" else "o"
        size = 15 if src == "Ortho Tiles" else 5
        alpha = 0.7 if src == "Ortho Tiles" else 0.35

        ax.scatter(
            sub["umap_x"],
            sub["umap_y"],
            c=[color],
            s=size,
            marker=marker,
            alpha=alpha,
            label=f"{src} (n={len(sub)})",
            rasterized=True,
        )

    ax.set_xlabel("UMAP-1", fontsize=12)
    ax.set_ylabel("UMAP-2", fontsize=12)
    ax.set_title(
        "Joint UMAP: Orthomosaic Tiles + Reference Databases",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ============================================================================
# BANNER AND OUTPUT HELPERS
# ============================================================================


def print_banner():
    """Print script banner."""
    print("=" * 70)
    print("  ORTHOMOSAIC DOMAIN SHIFT ANALYSIS (MMD)")
    print("=" * 70)
    print()


def print_section(title: str):
    """Print a section header."""
    print()
    print("-" * 70)
    print(f"  {title}")
    print("-" * 70)


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Compute domain shift (MMD) between orthomosaic tiles and reference databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python scripts/cli/compute_ortho_domain_shift.py \\
        --tile-embeddings /path/to/ortho_embeddings/embeddings.npz \\
        --reference-embeddings /path/to/ragweed_embeddings/embeddings.npz \\
        --output /path/to/domain_shift_report/

    # With subsampling for faster computation
    python scripts/cli/compute_ortho_domain_shift.py \\
        --tile-embeddings /path/to/embeddings.npz \\
        --reference-embeddings /path/to/ragweed_embeddings/ \\
        --output /path/to/report/ \\
        --subsample 500 --n-permutations 100
        """,
    )

    parser.add_argument(
        "--tile-embeddings",
        type=str,
        required=True,
        help="Path to tile embeddings .npz file (from embed_ortho_tiles.py)",
    )
    parser.add_argument(
        "--reference-embeddings",
        type=str,
        required=True,
        help="Path to reference .npz file or directory containing embeddings.npz",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for domain shift report",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Reserved for future GPU acceleration (currently unused, all computation is CPU-based)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of permutations for MMD significance test (default: 1000)",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=0,
        help="Subsample tiles for faster computation, 0 = all (default: 0)",
    )

    args = parser.parse_args()

    print_banner()

    # ---- Validate inputs ----
    tile_path = Path(args.tile_embeddings)
    ref_path = Path(args.reference_embeddings)
    output_dir = Path(args.output)

    if not tile_path.exists():
        print(f"ERROR: Tile embeddings not found: {tile_path}")
        sys.exit(1)

    if not ref_path.exists():
        print(f"ERROR: Reference embeddings not found: {ref_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()

    # ====================================================================
    # STEP 1: Load tile embeddings
    # ====================================================================
    print_section("[1/7] LOADING TILE EMBEDDINGS")

    tile_embeddings, tile_metadata = load_tile_embeddings(tile_path, args.subsample)
    print(f"  Loaded: {tile_embeddings.shape[0]} tiles, {tile_embeddings.shape[1]}-dim")
    if args.subsample > 0:
        print(f"  (Subsampled from original to {args.subsample} tiles)")

    # ====================================================================
    # STEP 2: Load reference embeddings
    # ====================================================================
    print_section("[2/7] LOADING REFERENCE EMBEDDINGS")

    ref_all, ref_sources, ref_per_db = load_reference_embeddings(ref_path)
    print(f"  Loaded: {ref_all.shape[0]} images, {ref_all.shape[1]}-dim")
    print(f"  Databases: {len(ref_per_db)}")
    for db_name, db_emb in sorted(ref_per_db.items(), key=lambda x: -len(x[1])):
        print(f"    {db_name:15s}: {len(db_emb):5d} images")

    # ====================================================================
    # STEP 3: L2 normalize all embeddings
    # ====================================================================
    print_section("[3/7] NORMALIZING EMBEDDINGS (L2)")

    tile_norms = np.linalg.norm(tile_embeddings, axis=1, keepdims=True)
    tile_norms[tile_norms == 0] = 1
    tile_normed = tile_embeddings / tile_norms

    ref_norms = np.linalg.norm(ref_all, axis=1, keepdims=True)
    ref_norms[ref_norms == 0] = 1
    ref_normed = ref_all / ref_norms

    ref_per_db_normed = {}
    for db_name, db_emb in ref_per_db.items():
        db_norms = np.linalg.norm(db_emb, axis=1, keepdims=True)
        db_norms[db_norms == 0] = 1
        ref_per_db_normed[db_name] = db_emb / db_norms

    if tile_normed.shape[1] != ref_normed.shape[1]:
        print(f"ERROR: Dimension mismatch! Tile embeddings: {tile_normed.shape[1]}, "
              f"Reference embeddings: {ref_normed.shape[1]}")
        print("Both must use the same feature extractor (e.g., ResNet50 = 2048-dim)")
        sys.exit(1)

    print(f"  Tile embeddings normalized: {tile_normed.shape}")
    print(f"  Reference embeddings normalized: {ref_normed.shape}")

    # ====================================================================
    # STEP 4: Compute global bandwidth (median heuristic)
    # ====================================================================
    print_section("[4/7] COMPUTING MMD PER DATABASE")

    # Global gamma from combined subsample
    print("  Computing global bandwidth (median heuristic)...")
    rng = np.random.RandomState(42)
    combined_for_gamma = np.vstack([tile_normed, ref_normed])
    n_subsample_gamma = min(2000, len(combined_for_gamma))
    sub_idx = rng.choice(len(combined_for_gamma), n_subsample_gamma, replace=False)
    from sklearn.metrics.pairwise import euclidean_distances
    sub_dists = euclidean_distances(
        combined_for_gamma[sub_idx], combined_for_gamma[sub_idx]
    )
    global_gamma = 1.0 / (2 * np.median(sub_dists[sub_dists > 0]) ** 2)
    print(f"  Global gamma: {global_gamma:.6f}")

    # MMD vs each database
    mmd_results = []
    print()
    print(f"  {'Database':15s} {'N_images':>8s} {'MMD':>8s} {'p-value':>10s}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*10}")

    for db_name in sorted(ref_per_db_normed.keys()):
        db_emb = ref_per_db_normed[db_name]

        # Use permutation test for significance
        if args.n_permutations > 0:
            mmd_val, p_val = compute_mmd_with_permutation_test(
                tile_normed, db_emb, gamma=global_gamma, n_permutations=args.n_permutations
            )
        else:
            mmd_val = compute_mmd(tile_normed, db_emb, gamma=global_gamma)
            p_val = None

        p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
        print(f"  {db_name:15s} {len(db_emb):8d} {mmd_val:8.4f} {p_str:>10s}")

        mmd_results.append(
            {
                "database": db_name,
                "mmd": round(float(mmd_val), 4),
                "p_value": round(float(p_val), 4) if p_val is not None else None,
                "n_images": int(len(db_emb)),
                "significant": bool(p_val < 0.05) if p_val is not None else None,
            }
        )

    # Overall MMD (tiles vs ALL reference images combined)
    print()
    print("  Computing overall MMD (tiles vs combined reference)...")
    overall_mmd = compute_mmd(tile_normed, ref_normed, gamma=global_gamma)
    print(f"  Overall MMD: {overall_mmd:.4f}")

    # ====================================================================
    # STEP 5: Per-tile distance ranking
    # ====================================================================
    print_section("[5/7] COMPUTING PER-TILE DISTANCES")

    tile_distances = compute_tile_distances(tile_normed, ref_normed)
    print(f"  Distance statistics:")
    print(f"    Min:    {tile_distances.min():.4f}")
    print(f"    Median: {np.median(tile_distances):.4f}")
    print(f"    Mean:   {tile_distances.mean():.4f}")
    print(f"    Max:    {tile_distances.max():.4f}")
    print(f"    Std:    {tile_distances.std():.4f}")

    # Build distance CSV
    distance_records = []
    for i in range(len(tile_distances)):
        record = {
            "tile_index": i,
            "distance_to_centroid": round(float(tile_distances[i]), 6),
        }
        if tile_metadata and i < len(tile_metadata):
            meta = tile_metadata[i]
            if isinstance(meta, dict):
                record["filename"] = meta.get("filename", "")
                record["row"] = meta.get("row", "")
                record["col"] = meta.get("col", "")
        distance_records.append(record)

    # Sort by distance (most distant first)
    distance_records.sort(key=lambda x: -x["distance_to_centroid"])

    # Add rank
    for rank, rec in enumerate(distance_records, 1):
        rec["rank"] = rank

    # Save CSV
    dist_df = pd.DataFrame(distance_records)
    dist_csv_path = output_dir / "tile_distances.csv"
    dist_df.to_csv(dist_csv_path, index=False)
    print(f"  Saved: {dist_csv_path}")

    # Show top 10 most distant tiles
    print()
    print("  Top 10 most distant tiles (best candidates for active learning):")
    for rec in distance_records[:10]:
        fname = rec.get("filename", f"tile_{rec['tile_index']}")
        print(f"    Rank {rec['rank']:3d}: {fname:35s} dist={rec['distance_to_centroid']:.4f}")

    # ====================================================================
    # STEP 6: Deployment recommendation
    # ====================================================================
    print_section("[6/7] DEPLOYMENT RECOMMENDATION")

    recommendation, risk_level = get_deployment_recommendation(overall_mmd)

    # Find closest and most distant databases
    mmd_sorted = sorted(mmd_results, key=lambda x: x["mmd"])
    closest_db = mmd_sorted[0]
    farthest_db = mmd_sorted[-1]

    print(f"  Overall MMD:        {overall_mmd:.4f}")
    print(f"  Risk level:         {risk_level}")
    print(f"  Recommendation:     {recommendation}")
    print()
    print(f"  Closest database:   {closest_db['database']} (MMD={closest_db['mmd']:.4f})")
    print(f"  Farthest database:  {farthest_db['database']} (MMD={farthest_db['mmd']:.4f})")

    # ====================================================================
    # STEP 7: Generate visualizations and save report
    # ====================================================================
    print_section("[7/7] GENERATING OUTPUTS")

    # MMD comparison bar chart
    print("  Creating MMD comparison bar chart...")
    create_mmd_bar_chart(mmd_results, overall_mmd, output_dir / "mmd_comparison.png")

    # Joint UMAP
    print("  Creating joint UMAP visualization...")
    create_joint_umap(
        tile_embeddings, tile_metadata, ref_all, ref_sources, output_dir
    )

    # Save full report JSON
    elapsed = (datetime.now() - start_time).total_seconds()

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tile_embeddings_path": str(tile_path),
        "reference_embeddings_path": str(ref_path),
        "n_tiles": int(len(tile_embeddings)),
        "n_reference_images": int(len(ref_all)),
        "n_reference_databases": int(len(ref_per_db)),
        "embedding_dim": int(tile_embeddings.shape[1]),
        "global_gamma": round(float(global_gamma), 6),
        "n_permutations": args.n_permutations,
        "subsample": args.subsample,
        "overall_mmd": round(float(overall_mmd), 4),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "closest_database": {
            "name": closest_db["database"],
            "mmd": closest_db["mmd"],
        },
        "farthest_database": {
            "name": farthest_db["database"],
            "mmd": farthest_db["mmd"],
        },
        "per_database_mmd": mmd_results,
        "tile_distance_stats": {
            "min": round(float(tile_distances.min()), 4),
            "median": round(float(np.median(tile_distances)), 4),
            "mean": round(float(tile_distances.mean()), 4),
            "max": round(float(tile_distances.max()), 4),
            "std": round(float(tile_distances.std()), 4),
        },
        "deployment_thresholds": {
            str(t): d for t, d in DEPLOYMENT_THRESHOLDS
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    report_path = output_dir / "domain_shift_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path}")

    # ====================================================================
    # FINAL SUMMARY
    # ====================================================================
    print()
    print("=" * 70)
    print("  DOMAIN SHIFT ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"  Tiles analyzed:    {len(tile_embeddings):,}")
    print(f"  Reference images:  {len(ref_all):,} ({len(ref_per_db)} databases)")
    print(f"  Overall MMD:       {overall_mmd:.4f}")
    print(f"  Risk level:        {risk_level}")
    print(f"  Recommendation:    {recommendation}")
    print(f"  Elapsed time:      {elapsed:.1f} s")
    print(f"  Output directory:  {output_dir}")
    print()
    print("  Output files:")
    for f_path in sorted(output_dir.iterdir()):
        size_kb = f_path.stat().st_size / 1024
        print(f"    {f_path.name:40s} {size_kb:8.1f} KB")
    print("=" * 70)


if __name__ == "__main__":
    main()
