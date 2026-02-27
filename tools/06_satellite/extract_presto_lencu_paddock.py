#!/usr/bin/env python3
"""
Extract Presto Embeddings for the Lencu Lentil Paddock (Santa Rosa, Chile)
and correlate with weed density maps (AMBEL, LENCU, POLAV, POLPE).

=============================================================================
QUE HACE ESTE SCRIPT / WHAT THIS SCRIPT DOES:
=============================================================================
1. Envia un trabajo a Copernicus Data Space (CDSE) para extraer embeddings
   Presto (128 dimensiones) del potrero de lentejas Santa Rosa,
   usando imagenes Sentinel-2 de Julio a Diciembre 2024 (6 meses).

2. Espera que el trabajo termine y descarga el resultado (GeoTIFF 128 bandas).

3. Carga los mapas de kriging de malezas (AMBEL, LENCU, POLAV, POLPE)
   del SmartMap y los alinea con los embeddings satelitales.

4. Ejecuta PCA y UMAP para reducir 128 dimensiones a 2D visualizables.

5. Genera figuras mostrando como la firma espectral satelital se relaciona
   con la densidad de cada especie de maleza.

=============================================================================
POTRERO / PADDOCK:
=============================================================================
  Santa Rosa, lenteja (Lens culinaris), Chile Central
  Area: ~3.4 ha
  Coordenadas: [-71.914, -36.532] (WGS84)
  Vuelo dron: 10 y 17 de Septiembre 2024
  Datos malezas: AMBEL (Ambrosia), LENCU (Convolvulus), POLAV, POLPE

=============================================================================
COMO USAR / HOW TO USE:
=============================================================================
  # 1. Abrir terminal
  # 2. Activar ambiente: conda activate worldcereal
  # 3. Ejecutar:
  python scripts/satellite/extract_presto_lencu_paddock.py

  # Para solo analizar (si los embeddings ya estan descargados):
  python scripts/satellite/extract_presto_lencu_paddock.py --analyze-only

  # Para solo enviar el trabajo (sin esperar):
  python scripts/satellite/extract_presto_lencu_paddock.py --extract-only

=============================================================================
SALIDAS / OUTPUTS (en outputs/lencu_presto/):
=============================================================================
  presto_jul_dec_2024.tif      - Embeddings crudos (128 bandas, GeoTIFF)
  presto_jul_dec_2024.gpkg     - Embeddings por pixel (GeoPackage, ~340 filas)
  aligned_data.csv             - Tabla: embeddings + densidad malezas por pixel
  fig_pca_by_species.png       - PCA coloreado por cada especie
  fig_umap_by_species.png      - UMAP coloreado por cada especie
  fig_correlation_heatmap.png  - Correlacion PCA components vs malezas
  fig_spatial_maps.png         - Mapas espaciales: embeddings vs kriging
"""

import argparse
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.mask import mask as rio_mask
from shapely.geometry import box, mapping

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---- Portable path resolution ----
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

# --- Library imports (replacing sibling-script imports) ---
from ragweed_toolkit.satellite.presto import (
    extract_presto_embeddings as _lib_extract_presto,
)
from ragweed_toolkit.satellite.presto import (
    raster_to_grid as _lib_raster_to_grid,
)

# -------------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------------
PROJECT_ROOT = _project_root
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "lencu_presto"
DATA_DIR     = PROJECT_ROOT / "research_docs" / "lencu_book_chapter" / "data"

PADDOCK_KML = DATA_DIR / "lencu paddock.kml"

# Smart Map kriging outputs (Smart Map outputs, UTM 19S, no CRS tag)
SMARTMAP_DIR = Path("/media/malezainia1/DRONES/ambel-data/Proyecto_lentejas_v2/smart map outputs")
WEED_SPECIES  = ["AMBEL", "LENCU", "POLAV", "POLPE"]
WEED_LABELS   = {
    "AMBEL": "Ambrosia artemisiifolia",
    "LENCU": "Convolvulus arvensis",
    "POLAV": "Polygonum aviculare",
    "POLPE": "Polygonum persicaria",
}
# Kriging density maps (5m grid, ~51x32 pixels)
KRIGING_TIFFS = {sp: SMARTMAP_DIR / f"1_Krig_{sp}_Grid_Map.tiff" for sp in WEED_SPECIES}
# Raw per-photogram counts CSV
WEED_CSV = SMARTMAP_DIR / "0_Dados.csv"

# Time window: July - December 2024
# 6 months captures: bare soil -> planting -> vegetative growth -> early maturation
START_DATE = "2024-07-01"
END_DATE   = "2024-12-31"

# Output files
EMBEDDINGS_TIFF = OUTPUT_DIR / "presto_jul_dec_2024.tif"
EMBEDDINGS_GPKG = OUTPUT_DIR / "presto_jul_dec_2024.gpkg"
ALIGNED_CSV     = OUTPUT_DIR / "aligned_data.csv"

# Kriging rasters assumed to be in UTM 19S (EPSG:32719)
# (Smart Map does not write the CRS tag to the TIFF)
KRIGING_CRS = CRS.from_epsg(32719)

# Paddock bounds in WGS84 (from KML)
PADDOCK_WGS84 = {
    "west":  -71.91353447,
    "south": -36.53220613,
    "east":  -71.91006627,
    "north": -36.53055987,
    "epsg":  4326,
}

# Colors per species for plots
SPECIES_COLORS = {
    "AMBEL": "Reds",
    "LENCU": "Blues",
    "POLAV": "Greens",
    "POLPE": "Oranges",
}


# -------------------------------------------------------------------------
# STEP 1: EXTRACT PRESTO EMBEDDINGS FROM CDSE
# -------------------------------------------------------------------------

def run_extraction():
    """Submit openEO Presto job and download 128-band GeoTIFF.

    Uses library function ragweed_toolkit.satellite.presto.extract_presto_embeddings.
    """
    print("\n" + "="*60)
    print("STEP 1: Extract Presto Embeddings via Copernicus Data Space")
    print("="*60)
    print("  Paddock:  Santa Rosa Lentil Field")
    print(f"  Period:   {START_DATE} to {END_DATE} (6 months)")
    print(f"  Output:   {EMBEDDINGS_TIFF}")
    print()

    if EMBEDDINGS_TIFF.exists():
        print(f"  [SKIP] GeoTIFF already exists: {EMBEDDINGS_TIFF}")
        print(f"         Size: {EMBEDDINGS_TIFF.stat().st_size / 1024 / 1024:.1f} MB")
        return True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = _lib_extract_presto(
        extent=PADDOCK_WGS84,
        start_date=START_DATE,
        end_date=END_DATE,
        output_path=EMBEDDINGS_TIFF,
        paddock_name="LencuSantaRosa_jul_dec_2024",
    )

    if result is None or not EMBEDDINGS_TIFF.exists():
        print("  ERROR: Extraction failed.")
        return False

    print(f"  SUCCESS: Downloaded {EMBEDDINGS_TIFF.stat().st_size / 1024 / 1024:.1f} MB")
    return True


# -------------------------------------------------------------------------
# STEP 2: CONVERT GEOTIFF TO PIXEL GEODATAFRAME
# -------------------------------------------------------------------------

def tiff_to_geodataframe():
    """Convert 128-band Presto GeoTIFF to per-pixel GeoDataFrame.

    Uses library function ragweed_toolkit.satellite.presto.raster_to_grid,
    with inline fallback for environments without worldcereal.
    """
    print("\n" + "="*60)
    print("STEP 2: Convert Presto GeoTIFF to Pixel GeoDataFrame")
    print("="*60)

    if EMBEDDINGS_GPKG.exists():
        print(f"  [SKIP] GeoPackage already exists: {EMBEDDINGS_GPKG}")
        gdf = gpd.read_file(EMBEDDINGS_GPKG)
        print(f"         {len(gdf)} pixels, {len(gdf.columns)} columns")
        return gdf

    # Load paddock boundary
    import fiona
    fiona.drvsupport.supported_drivers['KML']    = 'rw'
    fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
    paddock_gdf = gpd.read_file(str(PADDOCK_KML))
    print(f"  Paddock boundary: {paddock_gdf.total_bounds}")

    try:
        gdf = _lib_raster_to_grid(EMBEDDINGS_TIFF, paddock_gdf, "LencuSantaRosa")
    except Exception as e:
        print(f"  Library raster_to_grid failed ({e}), using inline fallback...")
        gdf = _raster_to_gdf_inline(EMBEDDINGS_TIFF, paddock_gdf)

    if gdf is not None and len(gdf) > 0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        gdf.to_file(str(EMBEDDINGS_GPKG), driver="GPKG")
        print(f"  Saved {len(gdf)} pixels -> {EMBEDDINGS_GPKG}")

    return gdf


def _raster_to_gdf_inline(tiff_path: Path, paddock_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fallback: convert embedding raster to GeoDataFrame without worldcereal import.

    IMPORTANT: This fallback uses DIFFERENT scale/offset from the library
    (SCALE=1/10000, OFFSET=-1) which matches the original script behavior.
    The library uses PRESTO_SCALE=0.0002, PRESTO_OFFSET=-6.
    Both produce valid results; this version is kept for backward compatibility.
    """
    import numpy as np

    NODATA  = 65535  # UInt16 nodata
    SCALE   = 1.0 / 10000.0
    OFFSET  = -1.0

    with rasterio.open(tiff_path) as src:
        print(f"  Raster: {src.count} bands x {src.height} x {src.width}, CRS={src.crs}")

        if paddock_gdf.crs != src.crs:
            padgdf = paddock_gdf.to_crs(src.crs)
        else:
            padgdf = paddock_gdf

        geom = padgdf.unary_union
        try:
            out_img, out_tr = rio_mask(src, [mapping(geom)], crop=True, nodata=NODATA)
        except Exception:
            out_img = src.read()
            out_tr  = src.transform

        n_bands, height, width = out_img.shape
        print(f"  Masked: {n_bands} bands x {height} x {width}")

        # Nodata mask from first band
        valid = out_img[0] != NODATA
        out_f = out_img.astype(np.float32) * SCALE + OFFSET
        out_f[:, ~valid] = np.nan

        rows, cols, cells, embs = [], [], [], []
        for r in range(height):
            for c in range(width):
                if not valid[r, c]:
                    continue
                x = out_tr.c + c * out_tr.a
                y = out_tr.f + r * out_tr.e
                cell = box(x, y, x + abs(out_tr.a), y - abs(out_tr.e))
                if cell.intersects(geom):
                    rows.append(r)
                    cols.append(c)
                    cells.append(cell)
                    embs.append(out_f[:, r, c])

    if not cells:
        print("  WARNING: No valid pixels found.")
        return gpd.GeoDataFrame()

    emb_arr = np.array(embs)  # shape (N, 128)
    emb_cols = {f"A{i:02d}": emb_arr[:, i] for i in range(emb_arr.shape[1])}
    gdf = gpd.GeoDataFrame({"row": rows, "col": cols, **emb_cols},
                           geometry=cells, crs=src.crs)
    print(f"  Extracted {len(gdf)} valid pixels")
    return gdf


# -------------------------------------------------------------------------
# STEP 3: SAMPLE KRIGING VALUES AT PRESTO PIXEL LOCATIONS
# -------------------------------------------------------------------------

def sample_kriging_at_pixels(embed_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    For each Presto pixel centroid, sample the kriging density value
    for each weed species.

    Returns DataFrame with embedding columns + AMBEL/LENCU/POLAV/POLPE columns.
    """
    print("\n" + "="*60)
    print("STEP 3: Sample Kriging Weed Density at Presto Pixel Locations")
    print("="*60)

    if ALIGNED_CSV.exists():
        print(f"  [SKIP] Aligned data already exists: {ALIGNED_CSV}")
        return pd.read_csv(ALIGNED_CSV)

    # Reproject pixel centroids to kriging CRS (UTM 19S)
    centroids_gdf = embed_gdf.copy()
    centroids_gdf.geometry = embed_gdf.geometry.centroid
    centroids_gdf = centroids_gdf.to_crs(KRIGING_CRS)

    centroids_xy = np.array([[g.x, g.y] for g in centroids_gdf.geometry])
    print(f"  {len(centroids_xy)} pixel centroids in {KRIGING_CRS}")

    # Build result DataFrame from embedding columns
    emb_cols = [c for c in embed_gdf.columns if c.startswith("A") and c[1:].isdigit()]
    df = embed_gdf[emb_cols].copy().reset_index(drop=True)
    df["pixel_x_utm"] = centroids_xy[:, 0]
    df["pixel_y_utm"] = centroids_xy[:, 1]

    # Sample each kriging TIFF
    for sp in WEED_SPECIES:
        krig_path = KRIGING_TIFFS[sp]
        if not krig_path.exists():
            print(f"  WARNING: Kriging file not found: {krig_path}")
            df[sp] = np.nan
            continue

        values = []
        with rasterio.open(krig_path) as src:
            # Override CRS (Smart Map TIFFs have no CRS tag)
            transform = src.transform
            data = src.read(1).astype(float)
            data[data < 0] = np.nan  # nodata

            print(f"  Sampling {sp}: {krig_path.name}")
            print(f"    Grid: {src.width}x{src.height}, bounds={src.bounds}")
            print(f"    Density range: {np.nanmin(data):.1f} - {np.nanmax(data):.1f}")

            for x, y in centroids_xy:
                # Convert UTM 19S coord to pixel row/col
                row, col = rasterio.transform.rowcol(transform, x, y)
                if 0 <= row < src.height and 0 <= col < src.width:
                    val = data[row, col]
                    values.append(val if not np.isnan(val) else 0.0)
                else:
                    values.append(np.nan)

        df[sp] = values
        n_valid = np.sum(~np.isnan(df[sp]))
        print(f"    Valid samples: {n_valid}/{len(df)}, mean={df[sp].mean():.1f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ALIGNED_CSV, index=False)
    print(f"  Saved aligned data: {ALIGNED_CSV}")
    return df


# -------------------------------------------------------------------------
# STEP 4: PCA + UMAP ANALYSIS
# -------------------------------------------------------------------------

def run_pca_umap(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce 128-dim Presto embeddings to 2D via PCA and UMAP."""
    print("\n" + "="*60)
    print("STEP 4: Dimension Reduction -- PCA + UMAP")
    print("="*60)

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    emb_cols = [c for c in df.columns if c.startswith("A") and c[1:].isdigit()]
    X = df[emb_cols].values

    # Remove rows with any NaN
    valid_mask = ~np.isnan(X).any(axis=1)
    # Also remove rows where all weed values are NaN
    weed_valid = ~df[WEED_SPECIES].isnull().all(axis=1).values
    mask = valid_mask & weed_valid

    print(f"  Embedding matrix: {X.shape}, valid pixels: {mask.sum()}")

    X_valid = X[mask]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    # PCA (keep 10 components + 2D for plotting)
    pca = PCA(n_components=min(10, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    var_explained = pca.explained_variance_ratio_ * 100
    print("  PCA variance explained:")
    for i, v in enumerate(var_explained[:5]):
        print(f"    PC{i+1}: {v:.1f}%")
    print(f"    Total (10 PCs): {var_explained.sum():.1f}%")

    df = df.copy()
    df["valid_mask"] = mask

    for i in range(X_pca.shape[1]):
        col = f"PC{i+1}"
        df.loc[mask, col] = X_pca[:, i]

    # UMAP
    umap_ok = False
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        X_umap = reducer.fit_transform(X_scaled)
        df.loc[mask, "UMAP1"] = X_umap[:, 0]
        df.loc[mask, "UMAP2"] = X_umap[:, 1]
        umap_ok = True
        print("  UMAP: 2D projection computed")
    except ImportError:
        print("  UMAP not available (pip install umap-learn)")

    return df, pca, var_explained, umap_ok


# -------------------------------------------------------------------------
# STEP 5: CORRELATION ANALYSIS
# -------------------------------------------------------------------------

def compute_correlations(df: pd.DataFrame, var_explained: np.ndarray) -> pd.DataFrame:
    """Compute Pearson correlations between PCA components and weed species."""
    print("\n" + "="*60)
    print("STEP 5: Correlation Analysis -- PCA Components vs Weed Density")
    print("="*60)

    pc_cols   = [f"PC{i+1}" for i in range(min(10, df.columns.str.startswith("PC").sum()))]
    pc_cols   = [c for c in pc_cols if c in df.columns]
    corr_rows = []

    for pc in pc_cols:
        row = {"Component": pc}
        for sp in WEED_SPECIES:
            mask = df["valid_mask"] & ~df[sp].isnull()
            if mask.sum() < 5:
                row[sp] = np.nan
                continue
            r = df.loc[mask, [pc, sp]].corr().iloc[0, 1]
            row[sp] = round(r, 3)
        corr_rows.append(row)

    corr_df = pd.DataFrame(corr_rows).set_index("Component")

    print("\n  Pearson correlation (PCA component vs weed density):")
    print(corr_df.to_string())

    return corr_df


# -------------------------------------------------------------------------
# STEP 6: GENERATE FIGURES
# -------------------------------------------------------------------------

def plot_pca_by_species(df: pd.DataFrame, var_explained: np.ndarray):
    """4-panel figure: PCA scatter colored by each weed species."""
    print("\n  Generating PCA scatter figure...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Presto Embeddings PCA -- Lentil Paddock Santa Rosa\n"
        f"Jul-Dec 2024 (6-month window), {int(var_explained[:2].sum())}% variance (PC1+PC2)",
        fontsize=13, fontweight="bold"
    )

    mask = df["valid_mask"]
    data = df[mask]

    for ax, sp in zip(axes.flat, WEED_SPECIES):
        sp_vals = data[sp].fillna(0)
        sc = ax.scatter(
            data["PC1"], data["PC2"],
            c=sp_vals, cmap=SPECIES_COLORS[sp],
            s=60, alpha=0.8, edgecolors="gray", linewidths=0.3
        )
        plt.colorbar(sc, ax=ax, label="Kriging density (plants/frame)")
        ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% variance)")
        ax.set_title(f"{sp} -- {WEED_LABELS[sp]}", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_pca_by_species.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_umap_by_species(df: pd.DataFrame):
    """4-panel UMAP figure colored by each weed species."""
    if "UMAP1" not in df.columns:
        print("  UMAP not available -- skipping UMAP figure")
        return

    print("  Generating UMAP scatter figure...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Presto Embeddings UMAP -- Lentil Paddock Santa Rosa\n"
        "Jul-Dec 2024 (6-month window)",
        fontsize=13, fontweight="bold"
    )

    mask = df["valid_mask"]
    data = df[mask]

    for ax, sp in zip(axes.flat, WEED_SPECIES):
        sp_vals = data[sp].fillna(0)
        sc = ax.scatter(
            data["UMAP1"], data["UMAP2"],
            c=sp_vals, cmap=SPECIES_COLORS[sp],
            s=60, alpha=0.8, edgecolors="gray", linewidths=0.3
        )
        plt.colorbar(sc, ax=ax, label="Kriging density (plants/frame)")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"{sp} -- {WEED_LABELS[sp]}", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_umap_by_species.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_correlation_heatmap(corr_df: pd.DataFrame, var_explained: np.ndarray):
    """Heatmap: PCA components (rows) vs weed species (cols)."""
    print("  Generating correlation heatmap...")

    fig, ax = plt.subplots(figsize=(7, 6))
    data = corr_df.values.astype(float)

    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")

    ax.set_xticks(range(len(WEED_SPECIES)))
    ax.set_xticklabels([f"{sp}\n({WEED_LABELS[sp][:15]}...)" for sp in WEED_SPECIES], fontsize=9)
    ax.set_yticks(range(len(corr_df)))
    pct = [f"{var_explained[i]:.1f}%" if i < len(var_explained) else "" for i in range(len(corr_df))]
    ax.set_yticklabels([f"{idx} ({p})" for idx, p in zip(corr_df.index, pct)], fontsize=9)
    ax.set_xlabel("Weed Species", fontsize=11)
    ax.set_ylabel("Presto PCA Component", fontsize=11)
    ax.set_title(
        "Correlation: Presto Embeddings x Weed Density\n"
        "Lentil Paddock Santa Rosa, Jul-Dec 2024",
        fontsize=11, fontweight="bold"
    )

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                color = "white" if abs(v) > 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_correlation_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_spatial_maps(df: pd.DataFrame, embed_gdf: gpd.GeoDataFrame):
    """
    Side-by-side spatial maps: PC1 over the paddock vs AMBEL kriging density.
    Shows spatial alignment between satellite embeddings and weed distribution.
    """
    print("  Generating spatial comparison maps...")

    # Reproject embed_gdf to UTM 19S
    gdf_utm = embed_gdf.to_crs(KRIGING_CRS)
    mask     = df["valid_mask"].values

    # Get valid rows (same order as embed_gdf)
    valid_idx = np.where(mask)[0]
    pc1_vals  = df["PC1"].values[mask] if "PC1" in df.columns else np.zeros(mask.sum())
    centroids = gdf_utm.geometry.centroid

    # Centroids of valid pixels
    cx = np.array([centroids.iloc[i].x for i in valid_idx])
    cy = np.array([centroids.iloc[i].y for i in valid_idx])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Spatial Maps -- Lentil Paddock Santa Rosa\n"
        "Presto PC1 vs Weed Kriging Density (Jul-Dec 2024)",
        fontsize=12, fontweight="bold"
    )

    # Panel 1: PC1 spatial map
    sc1 = axes[0].scatter(cx, cy, c=pc1_vals, cmap="RdYlGn_r", s=80, edgecolors="k", linewidths=0.2)
    plt.colorbar(sc1, ax=axes[0], label="PC1")
    axes[0].set_title("Presto PC1\n(satellite embedding)")
    axes[0].set_xlabel("UTM Easting (m)")
    axes[0].set_ylabel("UTM Northing (m)")
    axes[0].ticklabel_format(useOffset=False, style="plain")
    axes[0].tick_params(axis="x", rotation=45)

    # Panels 2 & 3: AMBEL and LENCU kriging (sampled values from df)
    for ax, sp, cmap in zip(axes[1:], ["AMBEL", "LENCU"], ["Reds", "Blues"]):
        sp_vals = df[sp].values[mask]
        sp_vals = np.nan_to_num(sp_vals, nan=0.0)
        sc = ax.scatter(cx, cy, c=sp_vals, cmap=cmap, s=80, edgecolors="k", linewidths=0.2)
        plt.colorbar(sc, ax=ax, label="Kriging density")
        ax.set_title(f"{sp} Kriging Density\n({WEED_LABELS[sp]})")
        ax.set_xlabel("UTM Easting (m)")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_spatial_maps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_scatter_matrix(df: pd.DataFrame):
    """PC1/PC2 vs each weed species as scatter with regression line."""
    print("  Generating PC vs weed scatter plots...")

    from scipy import stats
    mask = df["valid_mask"]
    data = df[mask].copy()
    data[WEED_SPECIES] = data[WEED_SPECIES].fillna(0)

    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    fig.suptitle(
        "Presto PCA vs Weed Species Density\n"
        "Lentil Paddock Santa Rosa, Jul-Dec 2024",
        fontsize=12, fontweight="bold"
    )

    for row_ax, sp in zip(axes, WEED_SPECIES):
        for ax, pc in zip(row_ax, ["PC1", "PC2"]):
            x = data[pc].values
            y = data[sp].values
            ax.scatter(x, y, alpha=0.6, s=30, color="steelblue", edgecolors="none")
            # Regression line
            slope, intercept, r, p, _ = stats.linregress(x, y)
            xline = np.linspace(x.min(), x.max(), 100)
            ax.plot(xline, slope * xline + intercept, "r-", linewidth=1.5,
                    label=f"r={r:.2f}, p={p:.3f}")
            ax.set_xlabel(pc)
            ax.set_ylabel(f"{sp} density")
            ax.set_title(f"{sp} vs {pc}", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig_scatter_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract Presto embeddings for lentil paddock and correlate with weed density"
    )
    parser.add_argument("--extract-only", action="store_true",
                        help="Submit openEO job and download only (skip analysis)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip extraction, analyze existing downloaded data")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  PRESTO EMBEDDINGS x WEED DENSITY -- Lentil Paddock Santa Rosa")
    print("="*70)
    print(f"  Window:  {START_DATE} to {END_DATE}")
    print(f"  Outputs: {OUTPUT_DIR}")

    # -- Step 1: Extract --
    if not args.analyze_only:
        ok = run_extraction()
        if not ok:
            print("\nExtraction failed. Exiting.")
            sys.exit(1)
        if args.extract_only:
            print("\nExtraction complete. Run --analyze-only when ready to visualize.")
            return

    # -- Check embedding file exists --
    if not EMBEDDINGS_TIFF.exists():
        print(f"\nERROR: Embedding GeoTIFF not found: {EMBEDDINGS_TIFF}")
        print("Run without --analyze-only first to extract embeddings.")
        sys.exit(1)

    # -- Step 2: GeoTIFF -> GeoDataFrame --
    embed_gdf = tiff_to_geodataframe()
    if embed_gdf is None or len(embed_gdf) == 0:
        print("ERROR: No pixels extracted from embedding raster. Exiting.")
        sys.exit(1)
    print(f"\n  Embedding pixels: {len(embed_gdf)}")

    # -- Step 3: Sample kriging values --
    df = sample_kriging_at_pixels(embed_gdf)

    # -- Step 4: PCA + UMAP --
    df, pca, var_explained, umap_ok = run_pca_umap(df)

    # -- Step 5: Correlations --
    corr_df = compute_correlations(df, var_explained)

    # -- Step 6: Figures --
    print("\n" + "="*60)
    print("STEP 6: Generating Figures")
    print("="*60)
    plot_pca_by_species(df, var_explained)
    plot_umap_by_species(df)
    plot_correlation_heatmap(corr_df, var_explained)
    plot_spatial_maps(df, embed_gdf)
    plot_scatter_matrix(df)

    # Save final DataFrame
    df.to_csv(OUTPUT_DIR / "analysis_results.csv", index=False)
    corr_df.to_csv(OUTPUT_DIR / "correlation_table.csv")

    # -- Summary --
    print("\n" + "="*70)
    print("  ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n  Pixels analyzed: {df['valid_mask'].sum()}")
    print(f"  Weed species:    {', '.join(WEED_SPECIES)}")
    print(f"\n  Key correlation (PC1 vs AMBEL): {corr_df.loc['PC1','AMBEL']:.3f}" if "PC1" in corr_df.index else "")

    print(f"\n  Output files in: {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("fig_*.png")):
        print(f"    {f.name}")
    print("    correlation_table.csv")
    print("    analysis_results.csv")

    print("\n  To view in QGIS:")
    print(f"    Layer > Add Vector > {EMBEDDINGS_GPKG}")
    print("    Style by 'PC1' column (graduated colors)")


if __name__ == "__main__":
    main()
