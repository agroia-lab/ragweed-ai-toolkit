"""
PRESTO Foundation Model Embeddings
====================================

Extract 128-dimensional embeddings from Sentinel-1/2 time series via
the WorldCereal/Presto model and openEO backend.

IMPORTANT: Presto requires exactly 12 monthly timesteps (a full year).
Shorter temporal windows will fail.

Functions
---------
extract_presto_embeddings
    Submit an openEO job to extract Presto embeddings for a bounding box.
raster_to_grid
    Convert a Presto raster (128-band GeoTIFF) to a 10m grid GeoDataFrame.
apply_clustering
    Apply KMeans clustering to embedding columns.
apply_pca
    Apply PCA dimensionality reduction for visualization.
"""

from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import box, mapping

# Presto UInt16 scaling
PRESTO_DIMENSIONS = 128
PRESTO_SCALE = 0.0002
PRESTO_OFFSET = -6
PRESTO_NODATA = 65535

# 12-month season windows (Presto requirement)
SEASON_WINDOWS_12M = {
    "20-21": ("2020-04-01", "2021-03-31"),
    "21-22": ("2021-04-01", "2022-03-31"),
    "22-23": ("2022-04-01", "2023-03-31"),
    "24-25": ("2024-04-01", "2025-03-31"),
    "25-26": ("2025-04-01", "2026-03-31"),
}


def extract_presto_embeddings(
    extent: Dict[str, float],
    start_date: str,
    end_date: str,
    output_path: str,
    *,
    paddock_name: str = "paddock",
    openeo_url: str = "https://openeo.dataspace.copernicus.eu",
) -> Optional[Path]:
    """Submit an openEO job to extract Presto embeddings.

    Requires the ``worldcereal`` and ``openeo`` packages in a dedicated
    conda environment.

    Args:
        extent: Bounding box with ``west``, ``south``, ``east``, ``north``,
            ``epsg`` keys (UTM coordinates).
        start_date: Start date (must cover 12 months).
        end_date: End date.
        output_path: Path for the output 128-band GeoTIFF.
        paddock_name: Name for the openEO job title.
        openeo_url: Copernicus Data Space endpoint.

    Returns:
        Path to downloaded GeoTIFF, or ``None`` on failure.
    """
    try:
        from openeo_gfmap.spatial import BoundingBoxExtent
        from openeo_gfmap.temporal import TemporalContext
        from worldcereal.job import INFERENCE_JOB_OPTIONS, create_embeddings_process_graph
        from worldcereal.parameters import EmbeddingsParameters
    except ImportError as e:
        raise ImportError(
            f"WorldCereal/openEO packages required: {e}\n"
            "Install: pip install worldcereal openeo openeo-gfmap"
        )

    bbox_extent = BoundingBoxExtent(
        west=extent["west"],
        south=extent["south"],
        east=extent["east"],
        north=extent["north"],
        epsg=extent["epsg"],
    )
    temporal_extent = TemporalContext(start_date=start_date, end_date=end_date)

    inference_result = create_embeddings_process_graph(
        spatial_extent=bbox_extent,
        temporal_extent=temporal_extent,
        embeddings_parameters=EmbeddingsParameters(),
        scale_uint16=True,
    )

    job = inference_result.create_job(
        title=f"Presto {paddock_name} {start_date} to {end_date}",
        job_options=INFERENCE_JOB_OPTIONS,
    )
    job.start_and_wait()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    for asset in job.get_results().get_assets():
        if asset.metadata.get("type", "").startswith("image/tiff"):
            asset.download(str(out))
            break
    else:
        return None

    return out if out.exists() else None


def raster_to_grid(
    raster_path: str,
    paddock_gdf: gpd.GeoDataFrame,
    paddock_name: str,
) -> gpd.GeoDataFrame:
    """Convert a Presto 128-band raster to a 10m grid GeoDataFrame.

    Reads the raster, clips to the paddock boundary, converts UInt16
    values to float embeddings, and returns a GeoDataFrame with columns
    ``A00`` through ``A127``.

    Args:
        raster_path: Path to the 128-band GeoTIFF.
        paddock_gdf: Paddock boundary GeoDataFrame.
        paddock_name: Name of the paddock (added as column).

    Returns:
        GeoDataFrame with grid cells and embedding columns.
    """
    import rasterio
    from rasterio.mask import mask as rio_mask

    with rasterio.open(raster_path) as src:
        if paddock_gdf.crs != src.crs:
            paddock_gdf = paddock_gdf.to_crs(src.crs)

        paddock_geom = paddock_gdf.unary_union

        try:
            out_image, out_transform = rio_mask(
                src, [mapping(paddock_geom)], crop=True, nodata=PRESTO_NODATA
            )
        except Exception:
            out_image = src.read()
            out_transform = src.transform

        n_bands, height, width = out_image.shape
        nodata_mask = out_image[0] == PRESTO_NODATA
        out_image = out_image.astype(np.float32) * PRESTO_SCALE + PRESTO_OFFSET
        out_image[:, nodata_mask] = np.nan

        cells = []
        emb_list = []
        cell_id = 0

        for row in range(height):
            for col in range(width):
                x = out_transform.c + col * out_transform.a
                y = out_transform.f + row * out_transform.e
                cell_geom = box(x, y, x + abs(out_transform.a), y - abs(out_transform.e))

                pixel_emb = out_image[:, row, col]
                if np.isnan(pixel_emb[0]):
                    continue
                if cell_geom.intersects(paddock_geom):
                    clipped = cell_geom.intersection(paddock_geom)
                    if not clipped.is_empty:
                        cells.append({"cell_id": cell_id, "geometry": clipped})
                        emb_list.append(pixel_emb)
                        cell_id += 1

    if not cells:
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame(cells, crs=src.crs)
    if gdf.crs.to_epsg() != 32719:
        gdf = gdf.to_crs(epsg=32719)

    emb_array = np.array(emb_list)
    for i in range(min(PRESTO_DIMENSIONS, emb_array.shape[1])):
        gdf[f"A{i:02d}"] = emb_array[:, i]

    gdf["paddock"] = paddock_name
    return gdf


def apply_clustering(
    gdf: gpd.GeoDataFrame,
    n_clusters: int = 20,
    *,
    scope: str = "global",
) -> gpd.GeoDataFrame:
    """Apply KMeans clustering to Presto embeddings.

    Args:
        gdf: GeoDataFrame with ``A00``-``A127`` columns.
        n_clusters: Number of clusters.
        scope: ``"global"`` (all paddocks together) or ``"local"``
            (per-paddock).

    Returns:
        GeoDataFrame with added ``cluster_{scope}_{n_clusters}`` column.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    emb_cols = [f"A{i:02d}" for i in range(PRESTO_DIMENSIONS) if f"A{i:02d}" in gdf.columns]
    if not emb_cols:
        return gdf

    col_name = f"cluster_{scope}_{n_clusters}"
    gdf[col_name] = np.nan

    if scope == "global":
        valid = gdf[emb_cols[0]].notna()
        if valid.sum() < n_clusters:
            return gdf
        X = StandardScaler().fit_transform(gdf.loc[valid, emb_cols].values)
        gdf.loc[valid, col_name] = KMeans(
            n_clusters=n_clusters, random_state=42, n_init=10
        ).fit_predict(X).astype(int)
    else:
        for paddock in gdf["paddock"].unique():
            mask = (gdf["paddock"] == paddock) & gdf[emb_cols[0]].notna()
            X_raw = gdf.loc[mask, emb_cols].values
            k = min(n_clusters, len(X_raw) - 1)
            if k < 2:
                continue
            X = StandardScaler().fit_transform(X_raw)
            gdf.loc[mask, col_name] = KMeans(
                n_clusters=k, random_state=42, n_init=10
            ).fit_predict(X).astype(int)

    return gdf


def apply_pca(
    gdf: gpd.GeoDataFrame,
    n_components: int = 3,
    *,
    scope: str = "global",
) -> gpd.GeoDataFrame:
    """Apply PCA to Presto embeddings for visualization.

    First 3 components are also normalized to 0-255 for RGB mapping.

    Args:
        gdf: GeoDataFrame with ``A00``-``A127`` columns.
        n_components: Number of PCA components.
        scope: ``"global"`` or ``"local"`` (per-paddock).

    Returns:
        GeoDataFrame with added ``pca_{scope}_{i}`` columns.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    emb_cols = [f"A{i:02d}" for i in range(PRESTO_DIMENSIONS) if f"A{i:02d}" in gdf.columns]
    if not emb_cols:
        return gdf

    def _do_pca(mask):
        X = StandardScaler().fit_transform(gdf.loc[mask, emb_cols].values)
        pca = PCA(n_components=n_components)
        result = pca.fit_transform(X)
        for i in range(n_components):
            col = f"pca_{scope}_{i + 1}"
            if col not in gdf.columns:
                gdf[col] = np.nan
            gdf.loc[mask, col] = result[:, i]
            # RGB normalization
            col_rgb = f"{col}_rgb"
            if col_rgb not in gdf.columns:
                gdf[col_rgb] = np.nan
            vals = result[:, i]
            vmin, vmax = vals.min(), vals.max()
            if vmax > vmin:
                gdf.loc[mask, col_rgb] = ((vals - vmin) / (vmax - vmin) * 255).astype(int)

    if scope == "global":
        valid = gdf[emb_cols[0]].notna()
        if valid.sum() > n_components:
            _do_pca(valid)
    else:
        for paddock in gdf["paddock"].unique():
            mask = (gdf["paddock"] == paddock) & gdf[emb_cols[0]].notna()
            if mask.sum() > n_components:
                _do_pca(mask)

    return gdf
