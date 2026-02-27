"""
Satellite Composites
=====================

Earth Engine export functions for Sentinel-2 NDVI and RGB composites
with cloud masking (SCL bands 3, 8, 9, 10).

Functions
---------
get_sentinel2_collection
    Get a cloud-filtered Sentinel-2 image collection.
compute_ndvi
    Add an NDVI band to a Sentinel-2 image.
export_ndvi_max
    Export max NDVI composite as a GeoTIFF to Google Drive.
export_rgb_median
    Export median RGB composite as a GeoTIFF to Google Drive.
export_composites
    Batch export NDVI and/or RGB for multiple paddocks.
"""

from typing import List, Optional

import ee
import geopandas as gpd

# Cloud mask: SCL clear values (vegetation, bare soil, water, unclassified, snow)
SCL_CLEAR_VALUES = [4, 5, 6, 7, 11]

# Season date windows (peak crop season: Dec 15 - Jan 15)
SEASON_WINDOWS = {
    "25-26": ("2024-12-15", "2025-01-15"),
    "24-25": ("2023-12-15", "2024-01-15"),
    "22-23": ("2022-12-15", "2023-01-15"),
    "21-22": ("2021-12-15", "2022-01-15"),
    "20-21": ("2020-12-15", "2021-01-15"),
}


def get_sentinel2_collection(
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    *,
    max_cloud_pct: int = 30,
) -> ee.ImageCollection:
    """Get a cloud-filtered Sentinel-2 SR Harmonized collection.

    Args:
        geometry: Earth Engine geometry to filter by.
        start_date: Start date (``YYYY-MM-DD``).
        end_date: End date (``YYYY-MM-DD``).
        max_cloud_pct: Maximum cloudy pixel percentage per scene.

    Returns:
        Filtered ``ee.ImageCollection``.
    """
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
    )


def compute_ndvi(image: ee.Image) -> ee.Image:
    """Add an NDVI band to a Sentinel-2 image.

    NDVI = (B8 - B4) / (B8 + B4)

    Args:
        image: Sentinel-2 image with B8 (NIR) and B4 (Red).

    Returns:
        Image with an added ``NDVI`` band.
    """
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi)


def mask_clouds_scl(image: ee.Image) -> ee.Image:
    """Mask clouds using the Scene Classification Layer (SCL).

    Keeps pixels classified as vegetation (4), bare soil (5), water (6),
    unclassified (7), or snow (11).

    Args:
        image: Sentinel-2 image with SCL band.

    Returns:
        Cloud-masked image.
    """
    scl = image.select("SCL")
    mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11))
    return image.updateMask(mask)


def export_ndvi_max(
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    paddock_name: str,
    *,
    drive_folder: str = "satellite_exports",
    scale: int = 10,
    crs: str = "EPSG:32719",
) -> Optional[ee.batch.Task]:
    """Export max NDVI composite to Google Drive.

    Args:
        geometry: Earth Engine geometry for the paddock.
        start_date: Start date.
        end_date: End date.
        paddock_name: Name used in the output filename.
        drive_folder: Google Drive folder name.
        scale: Export resolution in meters.
        crs: Coordinate reference system.

    Returns:
        Earth Engine export task, or ``None`` if no images found.
    """
    collection = get_sentinel2_collection(geometry, start_date, end_date)
    count = collection.size().getInfo()
    if count == 0:
        return None

    ndvi_max = collection.map(compute_ndvi).select("NDVI").max()

    task = ee.batch.Export.image.toDrive(
        image=ndvi_max,
        description=f"ndvi_max_{paddock_name}",
        folder=drive_folder,
        fileNamePrefix=f"ndvi_max_{paddock_name}",
        region=geometry,
        scale=scale,
        crs=crs,
        maxPixels=1e9,
    )
    task.start()
    return task


def export_rgb_median(
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    paddock_name: str,
    *,
    drive_folder: str = "satellite_exports",
    scale: int = 10,
    crs: str = "EPSG:32719",
) -> Optional[ee.batch.Task]:
    """Export median RGB composite to Google Drive.

    Args:
        geometry: Earth Engine geometry for the paddock.
        start_date: Start date.
        end_date: End date.
        paddock_name: Name used in the output filename.
        drive_folder: Google Drive folder name.
        scale: Export resolution in meters.
        crs: Coordinate reference system.

    Returns:
        Earth Engine export task, or ``None`` if no images found.
    """
    collection = get_sentinel2_collection(geometry, start_date, end_date)
    count = collection.size().getInfo()
    if count == 0:
        return None

    rgb_median = collection.select(["B4", "B3", "B2"]).median()
    rgb_vis = rgb_median.divide(3000).multiply(255).clamp(0, 255).toUint8()

    task = ee.batch.Export.image.toDrive(
        image=rgb_vis,
        description=f"rgb_median_{paddock_name}",
        folder=drive_folder,
        fileNamePrefix=f"rgb_median_{paddock_name}",
        region=geometry,
        scale=scale,
        crs=crs,
        maxPixels=1e9,
    )
    task.start()
    return task


def export_composites(
    boundaries_path: str,
    season: str,
    *,
    paddock: str = "all",
    composite_type: str = "both",
    ee_project: Optional[str] = None,
) -> List[ee.batch.Task]:
    """Batch export composites for paddocks.

    Initializes Earth Engine, loads paddock boundaries from a GeoPackage,
    and submits export tasks to Google Drive.

    Args:
        boundaries_path: Path to paddock boundaries GeoPackage.
        season: Season key (e.g. ``"25-26"``).
        paddock: Paddock name or ``"all"``.
        composite_type: ``"ndvi"``, ``"rgb"``, or ``"both"``.
        ee_project: Earth Engine project ID (optional).

    Returns:
        List of submitted export tasks.
    """
    if ee_project:
        ee.Initialize(project=ee_project)

    start_date, end_date = SEASON_WINDOWS[season]

    gdf = gpd.read_file(boundaries_path)
    col = "paddock" if "paddock" in gdf.columns else gdf.columns[0]
    if paddock.lower() != "all":
        gdf = gdf[gdf[col] == paddock]

    tasks: List[ee.batch.Task] = []

    for name in gdf[col].unique():
        subset = gdf[gdf[col] == name]
        geom_wgs84 = subset.to_crs("EPSG:4326").unary_union

        if geom_wgs84.geom_type == "MultiPolygon":
            geom_wgs84 = max(geom_wgs84.geoms, key=lambda p: p.area)

        coords = [[c[0], c[1]] for c in geom_wgs84.exterior.coords]
        ee_geom = ee.Geometry.Polygon([coords])

        if composite_type in ("ndvi", "both"):
            t = export_ndvi_max(ee_geom, start_date, end_date, name)
            if t:
                tasks.append(t)
        if composite_type in ("rgb", "both"):
            t = export_rgb_median(ee_geom, start_date, end_date, name)
            if t:
                tasks.append(t)

    return tasks
