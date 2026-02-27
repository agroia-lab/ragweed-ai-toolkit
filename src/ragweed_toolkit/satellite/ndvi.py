"""
NDVI Time Series Statistics
============================

Compute per-pixel NDVI statistics from Sentinel-2 over a crop season
(Oct-Mar). Supports cloud masking, sub-season means, linear trend,
and date-of-maximum computation via Earth Engine.

Functions
---------
compute_ndvi_statistics
    Compute multi-band NDVI statistics image via Earth Engine.
compute_ndvi_slope
    Compute linear trend (slope) of NDVI over time.
compute_date_of_max
    Compute the day-of-year when maximum NDVI occurred.
export_ndvi_stats
    Export NDVI statistics to Google Drive.
extract_raster_to_grid
    Sample a downloaded NDVI stats raster onto a 10m grid.
create_risk_zones
    Classify pixels into orobanche risk zones.
"""


import ee
import geopandas as gpd
import numpy as np
from shapely.geometry import box

# Crop season windows (Oct 1 - Mar 31)
CROP_SEASON_WINDOWS = {
    "25-26": ("2025-10-01", "2026-03-31"),
    "24-25": ("2024-10-01", "2025-03-31"),
    "23-24": ("2023-10-01", "2024-03-31"),
    "22-23": ("2022-10-01", "2023-03-31"),
    "21-22": ("2021-10-01", "2022-03-31"),
    "20-21": ("2020-10-01", "2021-03-31"),
}

# Default grid size matching Sentinel-2 resolution
GRID_SIZE = 10


def _mask_clouds(image: ee.Image) -> ee.Image:
    """Mask clouds using SCL band."""
    scl = image.select("SCL")
    clear = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11))
    return image.updateMask(clear)


def _add_ndvi(image: ee.Image) -> ee.Image:
    """Add NDVI band."""
    return image.addBands(
        image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    )


def _add_month(image: ee.Image) -> ee.Image:
    """Add month band for sub-season filtering."""
    month = ee.Date(image.get("system:time_start")).get("month")
    return image.addBands(ee.Image.constant(month).rename("month").toInt())


def _add_date(image: ee.Image) -> ee.Image:
    """Add date band (days since epoch)."""
    days = ee.Date(image.get("system:time_start")).difference(ee.Date("1970-01-01"), "day")
    return image.addBands(ee.Image.constant(days).rename("date").toFloat())


def _add_doy(image: ee.Image) -> ee.Image:
    """Add day-of-year band."""
    doy = ee.Date(image.get("system:time_start")).getRelative("day", "year")
    return image.addBands(ee.Image.constant(doy).rename("doy").toFloat())


def _get_collection(geometry, start_date, end_date, cloud_pct=30):
    """Get cloud-masked, NDVI-enriched collection."""
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(_mask_clouds)
        .map(_add_ndvi)
        .map(_add_month)
        .select(["NDVI", "month"])
    )


def compute_ndvi_statistics(
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    *,
    cloud_pct: int = 30,
) -> ee.Image:
    """Compute NDVI time series statistics for a region.

    Returns a multi-band image with: ``ndvi_min``, ``ndvi_max``,
    ``ndvi_mean``, ``ndvi_median``, ``ndvi_std``, ``ndvi_count``,
    ``ndvi_range``, ``ndvi_cv``, ``ndvi_p10``, ``ndvi_p90``,
    ``ndvi_early``, ``ndvi_peak``, ``ndvi_late``, ``ndvi_decline``.

    Args:
        geometry: Earth Engine geometry.
        start_date: Start date.
        end_date: End date.
        cloud_pct: Maximum cloud percentage filter.

    Returns:
        Multi-band ``ee.Image``.
    """
    processed = _get_collection(geometry, start_date, end_date, cloud_pct)

    ndvi_min = processed.select("NDVI").reduce(ee.Reducer.min()).toFloat().rename("ndvi_min")
    ndvi_max = processed.select("NDVI").reduce(ee.Reducer.max()).toFloat().rename("ndvi_max")
    ndvi_mean = processed.select("NDVI").reduce(ee.Reducer.mean()).toFloat().rename("ndvi_mean")
    ndvi_median = (
        processed.select("NDVI").reduce(ee.Reducer.median())
        .toFloat().rename("ndvi_median")
    )
    ndvi_std = processed.select("NDVI").reduce(ee.Reducer.stdDev()).toFloat().rename("ndvi_std")
    ndvi_count = processed.select("NDVI").reduce(ee.Reducer.count()).toFloat().rename("ndvi_count")
    ndvi_p10 = (
        processed.select("NDVI").reduce(ee.Reducer.percentile([10]))
        .toFloat().rename("ndvi_p10")
    )
    ndvi_p90 = (
        processed.select("NDVI").reduce(ee.Reducer.percentile([90]))
        .toFloat().rename("ndvi_p90")
    )

    ndvi_range = ndvi_max.subtract(ndvi_min).toFloat().rename("ndvi_range")
    ndvi_cv = ndvi_std.divide(ndvi_mean).toFloat().rename("ndvi_cv")

    # Sub-season means
    early = processed.filter(ee.Filter.Or(ee.Filter.eq("month", 10), ee.Filter.eq("month", 11)))
    ndvi_early = early.select("NDVI").reduce(ee.Reducer.mean()).toFloat().rename("ndvi_early")

    peak = processed.filter(ee.Filter.Or(ee.Filter.eq("month", 12), ee.Filter.eq("month", 1)))
    ndvi_peak = peak.select("NDVI").reduce(ee.Reducer.mean()).toFloat().rename("ndvi_peak")

    late = processed.filter(ee.Filter.Or(ee.Filter.eq("month", 2), ee.Filter.eq("month", 3)))
    ndvi_late = late.select("NDVI").reduce(ee.Reducer.mean()).toFloat().rename("ndvi_late")

    ndvi_decline = ndvi_peak.subtract(ndvi_late).divide(ndvi_peak).toFloat().rename("ndvi_decline")

    return (
        ndvi_min.addBands(ndvi_max).addBands(ndvi_mean).addBands(ndvi_median)
        .addBands(ndvi_std).addBands(ndvi_count).addBands(ndvi_range)
        .addBands(ndvi_cv).addBands(ndvi_p10).addBands(ndvi_p90)
        .addBands(ndvi_early).addBands(ndvi_peak).addBands(ndvi_late)
        .addBands(ndvi_decline)
    )


def compute_ndvi_slope(
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    *,
    cloud_pct: int = 30,
) -> ee.Image:
    """Compute linear trend of NDVI over time.

    Positive slope = improving, negative = declining. Scaled to per-month.

    Args:
        geometry: Earth Engine geometry.
        start_date: Start date.
        end_date: End date.
        cloud_pct: Maximum cloud percentage.

    Returns:
        Two-band image: ``ndvi_slope`` and ``ndvi_intercept``.
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(_mask_clouds)
        .map(_add_ndvi)
        .map(_add_date)
        .select(["NDVI", "date"])
    )

    linear_fit = collection.reduce(ee.Reducer.linearFit())
    slope = linear_fit.select("scale").multiply(30).toFloat().rename("ndvi_slope")
    intercept = linear_fit.select("offset").toFloat().rename("ndvi_intercept")
    return slope.addBands(intercept)


def compute_date_of_max(
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    *,
    cloud_pct: int = 30,
) -> ee.Image:
    """Compute day-of-year when maximum NDVI occurred.

    Args:
        geometry: Earth Engine geometry.
        start_date: Start date.
        end_date: End date.
        cloud_pct: Maximum cloud percentage.

    Returns:
        Single-band image: ``ndvi_date_max``.
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(_mask_clouds)
        .map(_add_ndvi)
        .map(_add_doy)
    )
    max_image = collection.qualityMosaic("NDVI")
    return max_image.select("doy").toFloat().rename("ndvi_date_max")


def export_ndvi_stats(
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    paddock_name: str,
    *,
    drive_folder: str = "ndvi_stats_exports",
    season_label: str = "",
) -> ee.batch.Task:
    """Export all NDVI statistics + slope + date_max to Google Drive.

    Args:
        geometry: Earth Engine geometry.
        start_date: Start date.
        end_date: End date.
        paddock_name: Paddock name for the filename.
        drive_folder: Google Drive folder.
        season_label: Season label for the filename.

    Returns:
        Earth Engine export task.
    """
    stats = compute_ndvi_statistics(geometry, start_date, end_date)
    slope = compute_ndvi_slope(geometry, start_date, end_date)
    date_max = compute_date_of_max(geometry, start_date, end_date)
    result = stats.addBands(slope).addBands(date_max)

    safe_name = paddock_name.replace(" ", "_").replace(",", "")
    if season_label:
        filename = f"ndvi_stats_{season_label}_{safe_name}"
    else:
        filename = f"ndvi_stats_{safe_name}"

    task = ee.batch.Export.image.toDrive(
        image=result,
        description=filename,
        folder=drive_folder,
        fileNamePrefix=filename,
        region=geometry,
        scale=10,
        crs="EPSG:32719",
        maxPixels=1e9,
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


def _generate_grid(geometry, cell_size: int = 10) -> gpd.GeoDataFrame:
    """Generate a grid of cells over a geometry."""
    minx, miny, maxx, maxy = geometry.bounds
    cells, ids = [], []
    cell_id = 0
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cell = box(x, y, x + cell_size, y + cell_size)
            if cell.intersects(geometry):
                clipped = cell.intersection(geometry)
                if not clipped.is_empty:
                    cells.append(clipped)
                    ids.append(cell_id)
                    cell_id += 1
            y += cell_size
        x += cell_size
    return gpd.GeoDataFrame({"cell_id": ids, "geometry": cells}, crs="EPSG:32719")


def extract_raster_to_grid(
    raster_path: str,
    paddock_gdf: gpd.GeoDataFrame,
    paddock_name: str,
    *,
    grid_size: int = 10,
) -> gpd.GeoDataFrame:
    """Sample a downloaded NDVI stats raster onto a 10m grid.

    Args:
        raster_path: Path to multi-band GeoTIFF.
        paddock_gdf: Paddock boundary GeoDataFrame.
        paddock_name: Name of the paddock.
        grid_size: Grid cell size in meters.

    Returns:
        GeoDataFrame with grid cells and NDVI statistic columns.
    """
    import rasterio

    with rasterio.open(raster_path) as src:
        band_names = list(src.descriptions) if src.descriptions[0] else [
            "ndvi_min", "ndvi_max", "ndvi_mean", "ndvi_median", "ndvi_std",
            "ndvi_count", "ndvi_range", "ndvi_cv", "ndvi_p10", "ndvi_p90",
            "ndvi_early", "ndvi_peak", "ndvi_late", "ndvi_decline",
            "ndvi_slope", "ndvi_intercept", "ndvi_date_max",
        ]

        if paddock_gdf.crs != src.crs:
            paddock_gdf = paddock_gdf.to_crs(src.crs)

        grid = _generate_grid(paddock_gdf.unary_union, grid_size)
        centroids = grid.geometry.centroid
        coords = [(p.x, p.y) for p in centroids]
        values = list(src.sample(coords))

        result = grid.copy()
        for i, bname in enumerate(band_names[:src.count]):
            result[bname] = [
                float(v[i]) if not np.isnan(v[i]) and v[i] != src.nodata else np.nan
                for v in values
            ]
        result["paddock"] = paddock_name

    return result


def create_risk_zones(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Classify pixels into orobanche risk zones based on NDVI patterns.

    Risk score combines (inverted) ndvi_max, ndvi_std, (inverted) ndvi_slope,
    and ndvi_decline. Zones are quantile-based: Low / Medium / High.

    Args:
        gdf: GeoDataFrame with NDVI statistic columns.

    Returns:
        GeoDataFrame with added ``risk_score`` and ``risk_zone`` columns.
    """
    if "ndvi_max" not in gdf.columns:
        return gdf

    gdf["risk_score"] = 0.0
    valid = gdf["ndvi_max"].notna()
    if valid.sum() == 0:
        return gdf

    for col, weight, invert in [
        ("ndvi_max", 0.3, True),
        ("ndvi_std", 0.2, False),
        ("ndvi_slope", 0.25, True),
        ("ndvi_decline", 0.25, False),
    ]:
        if col not in gdf.columns:
            continue
        vals = gdf.loc[valid, col].values
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if vmax > vmin:
            norm = (vals - vmin) / (vmax - vmin)
            if invert:
                norm = 1 - norm
            gdf.loc[valid, "risk_score"] += norm * weight

    gdf["risk_zone"] = "Unknown"
    risk_vals = gdf.loc[valid, "risk_score"]
    q33 = risk_vals.quantile(0.33)
    q66 = risk_vals.quantile(0.66)
    gdf.loc[valid & (gdf["risk_score"] <= q33), "risk_zone"] = "Low"
    gdf.loc[valid & (gdf["risk_score"] > q33) & (gdf["risk_score"] <= q66), "risk_zone"] = "Medium"
    gdf.loc[valid & (gdf["risk_score"] > q66), "risk_zone"] = "High"

    return gdf
