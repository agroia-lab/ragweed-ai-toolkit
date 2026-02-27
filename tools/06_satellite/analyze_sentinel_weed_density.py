#!/usr/bin/env python3
"""
Analyze Sentinel-2 Spectral Signatures vs Weed Species Density
Lentil Paddock Santa Rosa — September 24, 2024

=============================================================================
QUE HACE ESTE SCRIPT / WHAT THIS SCRIPT DOES:
=============================================================================
Analiza si la firma espectral de Sentinel-2 (septiembre 24, 2024) se
relaciona con la densidad de malezas medida con el dron (sept 10-17, 2024).

La fecha de la imagen satelital (24-sept) esta a solo 7 dias del ultimo
vuelo dron (17-sept), lo que hace esta comparacion muy valida.

Pasos:
  1. Carga la imagen Sentinel-2 (11 bandas, 5m, ya recortada al potrero)
  2. Carga los mapas de kriging de malezas (AMBEL, LENCU, POLAV, POLPE)
  3. Alinea ambos datasets en la misma grilla de 5m
  4. Calcula indices espectrales: NDVI, EVI, SWIR, BSI, etc.
  5. Ejecuta PCA sobre las bandas espectrales
  6. Visualiza con UMAP (reduccion no lineal)
  7. Genera figuras y tabla de correlaciones

DATOS USADOS:
  Imagen: sentinel sept oct 24 / 2024-09-24, Boundary1.data.tif
    - 11 bandas Sentinel-2 L1C (reflectancia de superficie)
    - Resolucion: 5m (interpolado desde 10m nativo S2)
    - CRS: EPSG:32719 (UTM zona 19S)

  Malezas: smart map outputs / 1_Krig_*_Grid_Map.tiff
    - AMBEL (Ambrosia artemisiifolia)
    - LENCU (Convolvulus arvensis)
    - POLAV (Polygonum aviculare)
    - POLPE (Polygonum persicaria)
    - Resolucion: 5m, CRS: UTM zona 19S (sin etiqueta CRS)

COMO USAR:
  conda activate yolov8_custom
  python scripts/satellite/analyze_sentinel_weed_density.py

SALIDAS (en outputs/lencu_sentinel_analysis/):
  fig_spectral_indices.png       - Mapas espaciales de NDVI, EVI, BSI, SWIR
  fig_pca_by_species.png         - PCA scatter coloreado por especie
  fig_umap_by_species.png        - UMAP scatter coloreado por especie (si disponible)
  fig_spatial_maps.png           - PC1 + densidad AMBEL + densidad LENCU en mapa
  fig_correlation_heatmap.png    - Correlacion bandas/indices vs malezas
  fig_scatter_matrix.png         - Regresion indices vs densidad por especie
  fig_kriging_maps.png           - Mapas de kriging de las 4 especies
  correlation_table.csv          - Tabla de correlaciones completa
  analysis_data.csv              - Datos pixel a pixel completos
"""

import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol as rasterio_rowcol

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

# ---- Portable path resolution ----
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

# --- Library imports (replacing internal duplicate) ---
from ragweed_toolkit.satellite.indices import (
    compute_spectral_indices as _lib_compute_spectral_indices,
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = _project_root
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "lencu_sentinel_analysis"

# Sentinel-2 clipped multispectral (11 bands, already in UTM 19S, 5m)
SENTINEL_TIFF = Path(
    "/media/malezainia1/DRONES/ambel-data/Proyecto_lentejas_v2/"
    "sentinel sept oct 24/lencuSR-sept24PixDfields/2024-09-24, Boundary1.data.tif"
)

# Paddock boundary (UTM 19S)
PADDOCK_SHP = Path(
    "/media/malezainia1/DRONES/ambel-data/Proyecto_lentejas_v2/poligono lentejas v2.shp"
)

# Smart Map kriging density maps (5m, UTM 19S — no CRS tag in TIFF)
SMARTMAP_DIR = Path(
    "/media/malezainia1/DRONES/ambel-data/Proyecto_lentejas_v2/smart map outputs"
)
WEED_SPECIES = ["AMBEL", "LENCU", "POLAV", "POLPE"]
WEED_LABELS  = {
    "AMBEL": "Ambrosia artemisiifolia",
    "LENCU": "Convolvulus arvensis",
    "POLAV": "Polygonum aviculare",
    "POLPE": "Polygonum persicaria",
}
KRIGING_TIFFS = {sp: SMARTMAP_DIR / f"1_Krig_{sp}_Grid_Map.tiff" for sp in WEED_SPECIES}
WEED_CSV      = SMARTMAP_DIR / "0_Dados.csv"

# Kriging rasters are in UTM 19S (Smart Map doesn't write CRS tag)
KRIGING_EPSG = 32719

# Sentinel-2 band assignments for Boundary1.data.tif (11 bands)
# Bands are L1C reflectance (divided by 10000), resampled to 5m
# Order: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, Mask
S2_BAND_NAMES = ["B2_Blue", "B3_Green", "B4_Red", "B5_RE1", "B6_RE2",
                  "B7_RE3", "B8_NIR", "B8A_NIRn", "B11_SWIR1", "B12_SWIR2",
                  "Mask"]
S2_SPECTRAL_IDX = list(range(10))  # Bands 0-9 are spectral (band 10 = mask)

# Color maps per species
SPECIES_COLORS = {"AMBEL": "Reds", "LENCU": "Blues",
                  "POLAV": "Greens", "POLPE": "Oranges"}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD SENTINEL-2 DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_sentinel_data():
    """Load Sentinel-2 multispectral data and assign band names."""
    print("\n" + "="*60)
    print("STEP 1: Load Sentinel-2 Data (Sept 24, 2024)")
    print("="*60)

    with rasterio.open(SENTINEL_TIFF) as src:
        print(f"  File: {SENTINEL_TIFF.name}")
        print(f"  CRS: {src.crs}")
        print(f"  Size: {src.width} x {src.height} pixels")
        print(f"  Pixel size: {src.res[0]:.1f} m")
        print(f"  Bands: {src.count}")
        print(f"  Bounds: {src.bounds}")

        data   = src.read()                         # shape (11, H, W)
        transform = src.transform
        meta   = src.meta.copy()

    # Take spectral bands only (0-9), skip mask (10)
    spectral = data[S2_SPECTRAL_IDX, :, :]          # (10, H, W)
    mask_band = data[10, :, :]                       # quality mask

    # Valid pixel mask (mask_band > 0 = valid in QGIS exports)
    valid = mask_band > 0

    print(f"  Valid pixels: {valid.sum()} / {valid.size} "
          f"({100*valid.mean():.0f}%)")

    # Print band stats
    print("\n  Band statistics (reflectance):")
    for i, name in enumerate(S2_BAND_NAMES[:10]):
        b = spectral[i][valid]
        print(f"    {name:12s}: min={b.min():.3f}  max={b.max():.3f}  "
              f"mean={b.mean():.3f}")

    return spectral, valid, transform, meta


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: COMPUTE SPECTRAL INDICES
# ─────────────────────────────────────────────────────────────────────────────

def compute_spectral_indices(spectral: np.ndarray, valid: np.ndarray) -> dict:
    """Compute NDVI, EVI, BSI, SWIR, clay index, etc.

    Thin wrapper around ragweed_toolkit.satellite.indices.compute_spectral_indices
    that adds logging output for CLI usage.
    """
    print("\n" + "="*60)
    print("STEP 2: Compute Spectral Indices")
    print("="*60)

    # Delegate to library function (same signature: 10-band array + valid mask)
    indices = _lib_compute_spectral_indices(spectral, valid)

    # Print statistics for user feedback
    for name, arr in indices.items():
        v = arr[valid]
        print(f"  {name:8s}: min={np.nanmin(v):.3f}  max={np.nanmax(v):.3f}  "
              f"mean={np.nanmean(v):.3f}")

    return indices


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: LOAD KRIGING WEED DENSITY MAPS
# ─────────────────────────────────────────────────────────────────────────────

def load_kriging_maps():
    """Load kriging density maps for each species (assign UTM 19S CRS)."""
    print("\n" + "="*60)
    print("STEP 3: Load Kriging Weed Density Maps")
    print("="*60)

    kriging = {}
    meta_krig = None

    for sp in WEED_SPECIES:
        path = KRIGING_TIFFS[sp]
        if not path.exists():
            print(f"  WARNING: {path} not found")
            kriging[sp] = None
            continue

        with rasterio.open(path) as src:
            data = src.read(1).astype(float)
            data[data < -900] = np.nan  # nodata sentinel

            if meta_krig is None:
                meta_krig = {
                    "transform": src.transform,
                    "width":  src.width,
                    "height": src.height,
                    "bounds": src.bounds,
                }
            kriging[sp] = data
            v = data[~np.isnan(data)]
            print(f"  {sp}: {src.width}x{src.height} grid, "
                  f"density {v.min():.1f}–{v.max():.1f}, mean={v.mean():.1f}")

    return kriging, meta_krig


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: ALIGN DATASETS — same pixel grid
# ─────────────────────────────────────────────────────────────────────────────

def align_datasets(spectral, valid, s2_transform, indices, kriging, krig_meta):
    """
    Both rasters are in UTM 19S at ~5m resolution but may have slightly
    different extents and transforms. We iterate over Sentinel pixels
    and sample kriging values at the same location.
    """
    print("\n" + "="*60)
    print("STEP 4: Align Sentinel-2 Pixels with Kriging Values")
    print("="*60)

    H, W = valid.shape

    # Get pixel centroids in UTM 19S for all valid Sentinel pixels
    rows, cols = np.where(valid)
    # Pixel center from Sentinel-2 transform (which is already in UTM 19S)
    xs = s2_transform.c + (cols + 0.5) * s2_transform.a
    ys = s2_transform.f + (rows + 0.5) * s2_transform.e

    print(f"  Sentinel pixels: {len(rows)}")
    print(f"  Sentinel UTM range: "
          f"E {xs.min():.0f}–{xs.max():.0f}, N {ys.min():.0f}–{ys.max():.0f}")

    # Build feature matrix
    # Columns: 10 spectral bands + 9 indices + 4 species kriging
    records = []

    krig_tr = krig_meta["transform"]
    krig_H  = krig_meta["height"]
    krig_W  = krig_meta["width"]

    for i, (r, c) in enumerate(zip(rows, cols)):
        x, y = xs[i], ys[i]

        # Sample kriging at this location
        krow, kcol = rasterio_rowcol(krig_tr, x, y)
        krow, kcol = int(krow), int(kcol)

        weed_vals = {}
        for sp, kmap in kriging.items():
            if kmap is None:
                weed_vals[sp] = np.nan
            elif 0 <= krow < krig_H and 0 <= kcol < krig_W:
                v = kmap[krow, kcol]
                weed_vals[sp] = float(v) if not np.isnan(v) else 0.0
            else:
                weed_vals[sp] = np.nan

        # Spectral bands
        band_vals = {S2_BAND_NAMES[j]: float(spectral[j, r, c])
                     for j in range(10)}

        # Spectral indices
        idx_vals = {name: float(arr[r, c])
                    for name, arr in indices.items()
                    if not np.isnan(arr[r, c])}

        rec = {"row": int(r), "col": int(c),
               "x_utm": float(x), "y_utm": float(y),
               **band_vals, **idx_vals, **weed_vals}
        records.append(rec)

    df = pd.DataFrame(records)
    n_valid_weed = (~df[WEED_SPECIES].isnull().all(axis=1)).sum()
    print(f"  Total aligned pixels: {len(df)}")
    print(f"  Pixels with weed data: {n_valid_weed}")
    print("\n  Weed density at pixel level (sampled from kriging):")
    for sp in WEED_SPECIES:
        v = df[sp].dropna()
        print(f"    {sp}: {len(v)} pixels, "
              f"mean={v.mean():.1f}, max={v.max():.1f}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: PCA + UMAP
# ─────────────────────────────────────────────────────────────────────────────

def run_dimensionality_reduction(df: pd.DataFrame):
    """PCA + UMAP on spectral features."""
    print("\n" + "="*60)
    print("STEP 5: Dimension Reduction — PCA + UMAP")
    print("="*60)

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Feature columns: 10 spectral bands + 9 indices
    feature_cols = S2_BAND_NAMES[:10] + list(["NDVI","EVI","GNDVI","RENDVI",
                                               "S2WI","NBR2","BSI","Clay","SWIRd"])
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values
    valid_rows = ~np.isnan(X).any(axis=1)
    # Also require at least one weed value not NaN
    weed_ok = ~df[WEED_SPECIES].isnull().all(axis=1).values
    mask = valid_rows & weed_ok

    print(f"  Feature matrix: {X.shape}, valid: {mask.sum()}")

    X_valid = X[mask]
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_valid)

    # PCA
    n_comp = min(len(feature_cols), X_sc.shape[0] - 1, 10)
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_sc)

    var_exp = pca.explained_variance_ratio_ * 100
    print("  PCA variance explained:")
    cum = 0
    for i, v in enumerate(var_exp[:6]):
        cum += v
        print(f"    PC{i+1}: {v:.1f}%  (cumulative: {cum:.1f}%)")

    df["_valid"] = mask
    for i in range(X_pca.shape[1]):
        df.loc[mask, f"PC{i+1}"] = X_pca[:, i]

    # UMAP
    umap_ok = False
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42,
                            n_neighbors=min(15, mask.sum()-1), min_dist=0.1)
        X_umap = reducer.fit_transform(X_sc)
        df.loc[mask, "UMAP1"] = X_umap[:, 0]
        df.loc[mask, "UMAP2"] = X_umap[:, 1]
        umap_ok = True
        print("  UMAP: 2D projection done")
    except ImportError:
        print("  UMAP not installed — skipping (pip install umap-learn)")

    return df, pca, var_exp, feature_cols, umap_ok


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_correlations(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Pearson correlations: spectral features × weed species."""
    print("\n" + "="*60)
    print("STEP 6: Correlation Analysis")
    print("="*60)

    mask = df["_valid"]
    rows = []

    for feat in feature_cols:
        row = {"Feature": feat}
        for sp in WEED_SPECIES:
            both = mask & ~df[sp].isnull()
            if both.sum() < 5:
                row[sp] = np.nan
                continue
            r, p = stats.pearsonr(df.loc[both, feat], df.loc[both, sp])
            row[sp] = round(r, 3)
        rows.append(row)

    corr_df = pd.DataFrame(rows).set_index("Feature")

    print("\n  Top correlations with AMBEL (Ambrosia):")
    ambel_corr = corr_df["AMBEL"].abs().sort_values(ascending=False)
    for feat, r in ambel_corr.head(8).items():
        sign = "+" if corr_df.loc[feat, "AMBEL"] > 0 else "-"
        print(f"    {feat:12s}: r = {sign}{r:.3f}")

    return corr_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def plot_spectral_indices(indices: dict, valid: np.ndarray, transform):
    """Spatial maps of key spectral indices."""
    print("\n  [Figure 1] Spectral index maps...")
    show = ["NDVI", "EVI", "BSI", "S2WI"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Sentinel-2 Spectral Indices — Santa Rosa Lentil Paddock\n"
                 "September 24, 2024", fontsize=13, fontweight="bold")

    cmaps = {"NDVI": "RdYlGn", "EVI": "RdYlGn", "BSI": "RdBu_r", "S2WI": "Blues"}
    titles = {
        "NDVI":  "NDVI\n(Vegetación / verdor)",
        "EVI":   "EVI\n(Vegetación mejorada)",
        "BSI":   "BSI\n(Índice suelo desnudo)",
        "S2WI":  "S2WI\n(Humedad del suelo)",
    }

    for ax, name in zip(axes.flat, show):
        arr = indices[name]
        vmin = np.nanpercentile(arr[valid], 2)
        vmax = np.nanpercentile(arr[valid], 98)
        im = ax.imshow(arr, cmap=cmaps[name], vmin=vmin, vmax=vmax,
                       origin="upper")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(titles[name], fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_spectral_indices.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out.name}")


def plot_kriging_maps(kriging: dict):
    """Spatial maps of the 4 kriging density layers."""
    print("  [Figure 2] Kriging density maps...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Weed Species Kriging Density — Santa Rosa Lentil Paddock\n"
                 "Drone survey Sept 10-17, 2024", fontsize=13, fontweight="bold")

    for ax, sp in zip(axes.flat, WEED_SPECIES):
        arr = kriging.get(sp)
        if arr is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue
        vmax = np.nanpercentile(arr, 98)
        im = ax.imshow(arr, cmap=SPECIES_COLORS[sp], vmin=0, vmax=vmax,
                       origin="upper")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Detecciones/fotograma")
        ax.set_title(f"{sp} — {WEED_LABELS[sp]}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_kriging_maps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out.name}")


def plot_pca_by_species(df: pd.DataFrame, var_exp: np.ndarray):
    """4-panel PCA scatter colored by each weed species."""
    print("  [Figure 3] PCA scatter by species...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Sentinel-2 Espacio PCA — Coloreado por Densidad de Maleza\n"
        f"PC1 ({var_exp[0]:.0f}%) + PC2 ({var_exp[1]:.0f}%), Sept 24, 2024",
        fontsize=13, fontweight="bold"
    )

    mask = df["_valid"]
    data = df[mask]

    for ax, sp in zip(axes.flat, WEED_SPECIES):
        vals = data[sp].fillna(0)
        vmax = np.percentile(vals, 97)
        sc = ax.scatter(data["PC1"], data["PC2"],
                        c=vals, cmap=SPECIES_COLORS[sp],
                        vmin=0, vmax=vmax, s=40, alpha=0.8,
                        edgecolors="none")
        plt.colorbar(sc, ax=ax, label="Densidad kriging")
        ax.set_xlabel(f"PC1 ({var_exp[0]:.0f}%)")
        ax.set_ylabel(f"PC2 ({var_exp[1]:.0f}%)")
        ax.set_title(f"{sp} — {WEED_LABELS[sp][:28]}", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_pca_by_species.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out.name}")


def plot_umap_by_species(df: pd.DataFrame):
    """4-panel UMAP scatter colored by each weed species."""
    if "UMAP1" not in df.columns:
        print("  [Figure 4] UMAP skipped (not available)")
        return

    print("  [Figure 4] UMAP scatter by species...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Sentinel-2 Espacio UMAP — Coloreado por Densidad de Maleza\n"
        "Sept 24, 2024",
        fontsize=13, fontweight="bold"
    )

    mask = df["_valid"]
    data = df[mask]

    for ax, sp in zip(axes.flat, WEED_SPECIES):
        vals = data[sp].fillna(0)
        vmax = np.percentile(vals, 97)
        sc = ax.scatter(data["UMAP1"], data["UMAP2"],
                        c=vals, cmap=SPECIES_COLORS[sp],
                        vmin=0, vmax=vmax, s=40, alpha=0.8,
                        edgecolors="none")
        plt.colorbar(sc, ax=ax, label="Densidad kriging")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"{sp} — {WEED_LABELS[sp][:28]}", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_umap_by_species.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out.name}")


def plot_correlation_heatmap(corr_df: pd.DataFrame):
    """Heatmap: spectral features × weed species."""
    print("  [Figure 5] Correlation heatmap...")
    fig, ax = plt.subplots(figsize=(7, max(6, len(corr_df)*0.4 + 1)))
    data_m = corr_df.values.astype(float)

    im = ax.imshow(data_m, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")

    ax.set_xticks(range(len(WEED_SPECIES)))
    ax.set_xticklabels(
        [f"{sp}\n({WEED_LABELS[sp][:15]})" for sp in WEED_SPECIES], fontsize=9
    )
    ax.set_yticks(range(len(corr_df)))
    ax.set_yticklabels(corr_df.index, fontsize=8)
    ax.set_title(
        "Correlación Pearson: Firmas Espectrales × Densidad de Maleza\n"
        "Sentinel-2 Sept 24, 2024 — Potrero Lentejas Santa Rosa",
        fontsize=10, fontweight="bold"
    )

    for i in range(data_m.shape[0]):
        for j in range(data_m.shape[1]):
            v = data_m[i, j]
            if not np.isnan(v) and abs(v) > 0.1:
                color = "white" if abs(v) > 0.4 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_correlation_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out.name}")


def plot_spatial_maps(df: pd.DataFrame):
    """Spatial comparison: PC1 vs AMBEL kriging vs LENCU kriging."""
    print("  [Figure 6] Spatial maps...")
    if "PC1" not in df.columns:
        print("    PCA not available — skipping spatial maps")
        return

    mask = df["_valid"]
    data = df[mask]
    cx, cy = data["x_utm"].values, data["y_utm"].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Mapas Espaciales — Potrero Lentejas Santa Rosa\n"
        "Sentinel-2 PC1 vs Densidad de Malezas (Kriging)",
        fontsize=12, fontweight="bold"
    )

    # PC1
    sc1 = axes[0].scatter(cx, cy, c=data["PC1"], cmap="RdYlGn_r", s=50,
                          edgecolors="k", linewidths=0.1)
    plt.colorbar(sc1, ax=axes[0], label="PC1 (componente 1)")
    axes[0].set_title("Sentinel-2 PC1\n(firma espectral)")
    axes[0].ticklabel_format(useOffset=False, style="plain")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_xlabel("UTM Este (m)")
    axes[0].set_ylabel("UTM Norte (m)")

    # AMBEL
    ambel_v = data["AMBEL"].fillna(0)
    sc2 = axes[1].scatter(cx, cy, c=ambel_v, cmap="Reds", s=50,
                          edgecolors="k", linewidths=0.1,
                          vmax=np.percentile(ambel_v, 97))
    plt.colorbar(sc2, ax=axes[1], label="Densidad AMBEL (kriging)")
    axes[1].set_title("AMBEL — Ambrosia artemisiifolia\n(densidad kriging)")
    axes[1].ticklabel_format(useOffset=False, style="plain")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_xlabel("UTM Este (m)")

    # LENCU
    lencu_v = data["LENCU"].fillna(0)
    sc3 = axes[2].scatter(cx, cy, c=lencu_v, cmap="Blues", s=50,
                          edgecolors="k", linewidths=0.1,
                          vmax=np.percentile(lencu_v, 97))
    plt.colorbar(sc3, ax=axes[2], label="Densidad LENCU (kriging)")
    axes[2].set_title("LENCU — Convolvulus arvensis\n(densidad kriging)")
    axes[2].ticklabel_format(useOffset=False, style="plain")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].set_xlabel("UTM Este (m)")

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_spatial_maps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out.name}")


def plot_top_correlations(df: pd.DataFrame, corr_df: pd.DataFrame):
    """Scatter plots for top-correlated features vs AMBEL and LENCU."""
    print("  [Figure 7] Scatter plots — top correlations...")

    top_features = corr_df["AMBEL"].abs().sort_values(ascending=False).head(4).index.tolist()

    fig, axes = plt.subplots(4, 2, figsize=(10, 14))
    fig.suptitle("Top Correlaciones: Índices Espectrales vs Densidad de Maleza\n"
                 "Potrero Lentejas Santa Rosa, Sept 2024",
                 fontsize=12, fontweight="bold")

    mask = df["_valid"]
    data = df[mask].copy()

    for row_axes, feat in zip(axes, top_features):
        for ax, sp in zip(row_axes, ["AMBEL", "LENCU"]):
            both = ~data[sp].isnull() & ~data[feat].isnull()
            x = data.loc[both, feat].values
            y = data.loc[both, sp].values
            ax.scatter(x, y, alpha=0.5, s=20, color="steelblue", edgecolors="none")
            if len(x) >= 5:
                slope, intercept, r, p, _ = stats.linregress(x, y)
                xline = np.linspace(x.min(), x.max(), 100)
                ax.plot(xline, slope * xline + intercept, "r-", linewidth=1.5,
                        label=f"r={r:.2f}, p={p:.3f}")
                ax.legend(fontsize=7)
            ax.set_xlabel(feat)
            ax.set_ylabel(f"{sp} (densidad)")
            ax.set_title(f"{feat} vs {sp}", fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_scatter_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("  SENTINEL-2 × WEED DENSITY ANALYSIS — Santa Rosa Lentil Paddock")
    print("="*70)
    print("  Image:   Sentinel-2, Sept 24, 2024")
    print("  Weeds:   AMBEL, LENCU, POLAV, POLPE (kriging from Sept 10-17 drone)")
    print(f"  Output:  {OUTPUT_DIR}")

    # Load data
    spectral, valid, s2_tr, meta = load_sentinel_data()
    indices                      = compute_spectral_indices(spectral, valid)
    kriging, krig_meta           = load_kriging_maps()

    # Align
    df = align_datasets(spectral, valid, s2_tr, indices, kriging, krig_meta)

    # Save aligned data
    df.to_csv(OUTPUT_DIR / "analysis_data.csv", index=False)

    # PCA + UMAP
    df, pca, var_exp, feat_cols, umap_ok = run_dimensionality_reduction(df)

    # Correlations
    corr_df = compute_correlations(df, feat_cols)
    corr_df.to_csv(OUTPUT_DIR / "correlation_table.csv")

    # Figures
    print("\n" + "="*60)
    print("STEP 7: Generating Figures")
    print("="*60)

    plot_spectral_indices(indices, valid, s2_tr)
    plot_kriging_maps(kriging)
    plot_pca_by_species(df, var_exp)
    plot_umap_by_species(df)
    plot_correlation_heatmap(corr_df)
    plot_spatial_maps(df)
    plot_top_correlations(df, corr_df)

    # Print summary
    print("\n" + "="*70)
    print("  ANALYSIS COMPLETE")
    print("="*70)
    n_pix = df["_valid"].sum()
    print(f"\n  Pixels analyzed: {n_pix}")
    print(f"  Spectral features: {len(feat_cols)}")

    print("\n  Top 5 correlations with AMBEL (Ambrosia):")
    ambel_top = corr_df["AMBEL"].abs().sort_values(ascending=False).head(5)
    for feat, r in ambel_top.items():
        sign = "+" if corr_df.loc[feat, "AMBEL"] > 0 else "-"
        print(f"    {feat:12s}: r = {sign}{r:.3f}")

    print("\n  Top 5 correlations with LENCU (Convolvulus):")
    lencu_top = corr_df["LENCU"].abs().sort_values(ascending=False).head(5)
    for feat, r in lencu_top.items():
        sign = "+" if corr_df.loc[feat, "LENCU"] > 0 else "-"
        print(f"    {feat:12s}: r = {sign}{r:.3f}")

    print(f"\n  All figures saved to: {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("fig_*.png")):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
