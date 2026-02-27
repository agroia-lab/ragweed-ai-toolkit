#!/usr/bin/env python3
"""
Geographically Weighted Regression (GWR) for AMBEL and LENCU weed counts
against Presto PCA embeddings (PC1-PC3).

Models:
  1. AMBEL ~ PC1 + PC2 + PC3
  2. LENCU ~ PC1 + PC2 + PC3

Compares GWR (spatially varying coefficients) with OLS baseline.
Produces local R2 maps, local coefficient maps, and summary statistics.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
from spreg import OLS

# ---- Portable path resolution ----
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

# TODO: The library has ragweed_toolkit.spatial.gwr.fit_ols and
# ragweed_toolkit.spatial.gwr.fit_gwr, but they take a GeoDataFrame +
# column names as inputs, whereas this script passes raw numpy arrays
# (coords, y, X) directly to mgwr/spreg.  The library API would need
# adapting to support the raw-array workflow used here, or this script
# could be refactored to use GeoDataFrame input.  For now, keep the
# script-specific implementations.

OUTPUT_DIR = "/home/malezainia1/dev/INIA_DeepLearning_Ubuntu_mod_lleon/outputs/lencu_presto/spatial_analysis"
DATA_PATH = "/home/malezainia1/dev/INIA_DeepLearning_Ubuntu_mod_lleon/outputs/lencu_presto/analysis_results.csv"


def load_data():
    """Load CSV, filter to valid points, drop NaN in target columns."""
    df = pd.read_csv(DATA_PATH)
    df = df[df["valid_mask"] == 1.0].copy()
    print(f"Valid points: {len(df)}")

    # Drop rows where AMBEL or LENCU is NaN
    before = len(df)
    df = df.dropna(subset=["AMBEL", "LENCU", "PC1", "PC2", "PC3",
                            "pixel_x_utm", "pixel_y_utm"])
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} rows with NaN values, {after} remaining")

    return df


def fit_ols(y, X, var_names):
    """Fit OLS regression using spreg and return summary dict."""
    ols = OLS(y, X, name_y="y", name_x=var_names)
    return {
        "R2": ols.r2,
        "Adj_R2": ols.ar2,
        "AICc": ols.aic,  # spreg uses AIC; we'll note this
        "n": ols.n,
        "k": ols.k,
        "coefficients": dict(zip(["intercept"] + var_names,
                                  ols.betas.flatten().tolist())),
    }


def fit_gwr(coords, y, X, var_names):
    """Fit GWR with adaptive bisquare kernel, AICc bandwidth selection."""
    print("  Selecting bandwidth (AICc, adaptive bisquare)...")
    selector = Sel_BW(coords, y, X, kernel="bisquare", fixed=False)
    bw = selector.search(criterion="AICc", search_method="golden_section")
    print(f"  Optimal bandwidth: {bw} nearest neighbors")

    print("  Fitting GWR model...")
    model = GWR(coords, y, X, bw=bw, kernel="bisquare", fixed=False)
    results = model.fit()

    print(f"  Global R2:   {results.R2:.4f}")
    print(f"  Adj R2:      {results.adj_R2:.4f}")
    print(f"  AICc:        {results.aicc:.2f}")

    # Local results
    local_R2 = results.localR2.flatten()
    # Columns: intercept, PC1, PC2, PC3
    local_betas = results.params  # shape (n, k)

    summary = {
        "R2": results.R2,
        "Adj_R2": results.adj_R2,
        "AICc": results.aicc,
        "bandwidth": bw,
        "n": len(y),
        "k": X.shape[1],
    }

    local_data = {
        "local_R2": local_R2,
        "local_beta_intercept": local_betas[:, 0],
    }
    for i, name in enumerate(var_names):
        local_data[f"local_beta_{name}"] = local_betas[:, i + 1]

    return summary, local_data


def plot_local_r2(df, ambel_local, lencu_local, outpath):
    """Figure 1: Local R2 maps for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, local_data, title in [
        (axes[0], ambel_local, "AMBEL ~ PC1+PC2+PC3"),
        (axes[1], lencu_local, "LENCU ~ PC1+PC2+PC3"),
    ]:
        sc = ax.scatter(
            df["pixel_x_utm"], df["pixel_y_utm"],
            c=local_data["local_R2"], cmap="viridis",
            s=20, edgecolors="none", vmin=0, vmax=1,
        )
        ax.set_title(f"GWR Local R² — {title}", fontsize=12)
        ax.set_xlabel("UTM Easting (m)")
        ax.set_ylabel("UTM Northing (m)")
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, label="Local R²", shrink=0.8)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


def plot_local_coefs(df, ambel_local, lencu_local, outpath):
    """Figure 2: Local PC1 coefficient maps for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Determine shared color limits for PC1 beta
    all_betas = np.concatenate([
        ambel_local["local_beta_PC1"],
        lencu_local["local_beta_PC1"],
    ])
    vmax = np.percentile(np.abs(all_betas), 95)
    vmin = -vmax

    for ax, local_data, title in [
        (axes[0], ambel_local, "AMBEL"),
        (axes[1], lencu_local, "LENCU"),
    ]:
        sc = ax.scatter(
            df["pixel_x_utm"], df["pixel_y_utm"],
            c=local_data["local_beta_PC1"], cmap="RdBu_r",
            s=20, edgecolors="none", vmin=vmin, vmax=vmax,
        )
        ax.set_title(f"GWR Local β(PC1) — {title}", fontsize=12)
        ax.set_xlabel("UTM Easting (m)")
        ax.set_ylabel("UTM Northing (m)")
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, label="Local β(PC1)", shrink=0.8)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


def plot_summary_table(ols_ambel, ols_lencu, gwr_ambel, gwr_lencu, outpath):
    """Figure 3: Summary comparison table."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    table_data = [
        ["Metric", "OLS (AMBEL)", "GWR (AMBEL)", "OLS (LENCU)", "GWR (LENCU)"],
        ["R²", f"{ols_ambel['R2']:.4f}", f"{gwr_ambel['R2']:.4f}",
         f"{ols_lencu['R2']:.4f}", f"{gwr_lencu['R2']:.4f}"],
        ["Adj R²", f"{ols_ambel['Adj_R2']:.4f}", f"{gwr_ambel['Adj_R2']:.4f}",
         f"{ols_lencu['Adj_R2']:.4f}", f"{gwr_lencu['Adj_R2']:.4f}"],
        ["AIC/AICc", f"{ols_ambel['AICc']:.1f}", f"{gwr_ambel['AICc']:.1f}",
         f"{ols_lencu['AICc']:.1f}", f"{gwr_lencu['AICc']:.1f}"],
        ["Bandwidth", "Global", f"{gwr_ambel['bandwidth']:.0f} nn",
         "Global", f"{gwr_lencu['bandwidth']:.0f} nn"],
    ]

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)

    # Style header row
    for j in range(5):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    plt.title("OLS vs GWR Comparison", fontsize=14, fontweight="bold", pad=20)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


def save_geopkg(df, ambel_local, lencu_local, outpath):
    """Save local GWR results to GeoPackage (EPSG:32719)."""
    geometry = [Point(x, y) for x, y in zip(df["pixel_x_utm"], df["pixel_y_utm"])]
    gdf = gpd.GeoDataFrame(df[["pixel_x_utm", "pixel_y_utm", "AMBEL", "LENCU",
                                "PC1", "PC2", "PC3"]].copy(),
                           geometry=geometry, crs="EPSG:32719")

    # Add AMBEL local results
    gdf["ambel_local_R2"] = ambel_local["local_R2"]
    gdf["ambel_beta_PC1"] = ambel_local["local_beta_PC1"]
    gdf["ambel_beta_PC2"] = ambel_local["local_beta_PC2"]
    gdf["ambel_beta_PC3"] = ambel_local["local_beta_PC3"]

    # Add LENCU local results
    gdf["lencu_local_R2"] = lencu_local["local_R2"]
    gdf["lencu_beta_PC1"] = lencu_local["local_beta_PC1"]
    gdf["lencu_beta_PC2"] = lencu_local["local_beta_PC2"]
    gdf["lencu_beta_PC3"] = lencu_local["local_beta_PC3"]

    gdf.to_file(outpath, driver="GPKG")
    print(f"  Saved: {outpath}")


def save_summary_csv(ols_ambel, ols_lencu, gwr_ambel, gwr_lencu, outpath):
    """Save model comparison summary to CSV."""
    rows = []
    for model_name, stats in [
        ("OLS_AMBEL", ols_ambel), ("GWR_AMBEL", gwr_ambel),
        ("OLS_LENCU", ols_lencu), ("GWR_LENCU", gwr_lencu),
    ]:
        row = {"model": model_name}
        row["R2"] = stats["R2"]
        row["Adj_R2"] = stats["Adj_R2"]
        row["AICc"] = stats["AICc"]
        row["n"] = stats["n"]
        row["k"] = stats["k"]
        if "bandwidth" in stats:
            row["bandwidth"] = stats["bandwidth"]
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(outpath, index=False)
    print(f"  Saved: {outpath}")


def main():
    print("=" * 60)
    print("GWR Analysis: Weed Counts ~ Presto PCA Embeddings")
    print("=" * 60)

    # Load data
    df = load_data()

    # Prepare arrays
    coords = df[["pixel_x_utm", "pixel_y_utm"]].values
    var_names = ["PC1", "PC2", "PC3"]

    # Standardize predictors
    scaler = StandardScaler()
    X_raw = df[var_names].values
    X_scaled = scaler.fit_transform(X_raw)

    # --- AMBEL ---
    print("\n" + "=" * 60)
    print("Model 1: AMBEL ~ PC1 + PC2 + PC3")
    print("=" * 60)

    y_ambel = df["AMBEL"].values.reshape(-1, 1)

    print("\n--- OLS Baseline ---")
    ols_ambel = fit_ols(y_ambel, X_scaled, var_names)
    print(f"  R²: {ols_ambel['R2']:.4f}, Adj R²: {ols_ambel['Adj_R2']:.4f}, "
          f"AIC: {ols_ambel['AICc']:.1f}")
    print(f"  Coefficients: {ols_ambel['coefficients']}")

    print("\n--- GWR ---")
    gwr_ambel, ambel_local = fit_gwr(coords, y_ambel, X_scaled, var_names)

    # --- LENCU ---
    print("\n" + "=" * 60)
    print("Model 2: LENCU ~ PC1 + PC2 + PC3")
    print("=" * 60)

    y_lencu = df["LENCU"].values.reshape(-1, 1)

    print("\n--- OLS Baseline ---")
    ols_lencu = fit_ols(y_lencu, X_scaled, var_names)
    print(f"  R²: {ols_lencu['R2']:.4f}, Adj R²: {ols_lencu['Adj_R2']:.4f}, "
          f"AIC: {ols_lencu['AICc']:.1f}")
    print(f"  Coefficients: {ols_lencu['coefficients']}")

    print("\n--- GWR ---")
    gwr_lencu, lencu_local = fit_gwr(coords, y_lencu, X_scaled, var_names)

    # --- Figures ---
    print("\n" + "=" * 60)
    print("Generating figures...")
    print("=" * 60)

    plot_local_r2(df, ambel_local, lencu_local,
                  f"{OUTPUT_DIR}/fig_gwr_local_r2.png")

    plot_local_coefs(df, ambel_local, lencu_local,
                     f"{OUTPUT_DIR}/fig_gwr_local_coefs.png")

    plot_summary_table(ols_ambel, ols_lencu, gwr_ambel, gwr_lencu,
                       f"{OUTPUT_DIR}/fig_gwr_summary_table.png")

    # --- Save outputs ---
    print("\n" + "=" * 60)
    print("Saving outputs...")
    print("=" * 60)

    save_geopkg(df, ambel_local, lencu_local,
                f"{OUTPUT_DIR}/gwr_results.gpkg")

    save_summary_csv(ols_ambel, ols_lencu, gwr_ambel, gwr_lencu,
                     f"{OUTPUT_DIR}/gwr_summary.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
