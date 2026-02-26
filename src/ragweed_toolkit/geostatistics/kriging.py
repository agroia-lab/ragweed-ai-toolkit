"""
Ordinary Kriging Interpolator
==============================

Produce a continuous density surface from point observations over a polygon
boundary using ordinary kriging with an exponential variogram model.

Default parameters from the Springer chapter (Table 6):
    nugget      = 43,066.5
    sill        = 86,133.0
    range       = 81.2 m

The output is a regular grid (GeoDataFrame of square cells) at a configurable
resolution (default 5 m), clipped to the input polygon boundary.

Chapter reference: Section 5.4.2 (Kriging interpolation).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import geopandas as gpd
from shapely.geometry import box

from .variograms import VariogramModel


def _exponential_covariance(h: np.ndarray, nugget: float, partial_sill: float, range_param: float) -> np.ndarray:
    """Covariance function C(h) = sill - gamma(h) for nugget model."""
    sill = nugget + partial_sill
    gamma = nugget + partial_sill * (1 - np.exp(-h / range_param))
    # At h=0, C(0) = sill (total variance)
    cov = np.where(h == 0, sill, sill - gamma)
    return cov


def _build_kriging_system(
    obs_coords: np.ndarray,
    nugget: float,
    partial_sill: float,
    range_param: float,
) -> np.ndarray:
    """
    Build the left-hand-side kriging matrix (n+1 x n+1).

    Includes the Lagrange multiplier row/column for the unbiasedness constraint.
    """
    n = len(obs_coords)
    dist = np.sqrt(
        ((obs_coords[:, np.newaxis, :] - obs_coords[np.newaxis, :, :]) ** 2).sum(axis=2)
    )
    C = _exponential_covariance(dist, nugget, partial_sill, range_param)

    # Augment with Lagrange multiplier
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = C
    K[n, :n] = 1.0
    K[:n, n] = 1.0
    # K[n, n] = 0  (already zero)

    return K


def ordinary_kriging(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    boundary: gpd.GeoDataFrame,
    resolution: float = 5.0,
    nugget: float = 43066.5,
    partial_sill: float = 43066.5,
    range_param: float = 81.2,
    crs: Optional[str] = None,
    variogram_model: Optional[VariogramModel] = None,
) -> gpd.GeoDataFrame:
    """
    Interpolate point observations onto a regular grid using ordinary kriging.

    Parameters
    ----------
    gdf : GeoDataFrame
        Point observations with a numeric ``value_col`` column.
        Must be in a projected CRS (metres).
    value_col : str
        Name of the column to interpolate.
    boundary : GeoDataFrame
        Polygon(s) defining the study area. Grid cells outside the boundary
        are discarded.
    resolution : float
        Grid cell size in metres (default 5 m).
    nugget : float
        Nugget (C0) of the exponential variogram.
    partial_sill : float
        Partial sill (C) of the exponential variogram.
    range_param : float
        Practical range (a) of the exponential variogram in metres.
    crs : str, optional
        Override CRS string (e.g. ``"EPSG:32719"``).  By default the CRS of
        *gdf* is used.
    variogram_model : VariogramModel, optional
        If provided, variogram parameters are taken from the model and the
        individual nugget/partial_sill/range_param arguments are ignored.

    Returns
    -------
    GeoDataFrame
        Regular grid with columns ``predicted``, ``variance``, and polygon
        geometries for each cell.  CRS matches the input data.
    """
    # Extract variogram parameters
    if variogram_model is not None:
        nugget = variogram_model.nugget
        partial_sill = variogram_model.partial_sill
        range_param = variogram_model.range_param

    if crs is None:
        crs = gdf.crs

    # Ensure same CRS
    gdf_proj = gdf.to_crs(crs) if gdf.crs != crs else gdf
    boundary_proj = boundary.to_crs(crs) if boundary.crs != crs else boundary

    # Observation coordinates and values
    obs_x = np.array([g.x for g in gdf_proj.geometry])
    obs_y = np.array([g.y for g in gdf_proj.geometry])
    obs_coords = np.column_stack([obs_x, obs_y])
    obs_vals = gdf_proj[value_col].values.astype(float)

    # Drop NaN observations
    valid = ~np.isnan(obs_vals)
    obs_coords = obs_coords[valid]
    obs_vals = obs_vals[valid]
    n = len(obs_vals)

    if n < 3:
        raise ValueError(f"Need at least 3 valid observations, got {n}.")

    # Build the kriging system (LHS)
    K = _build_kriging_system(obs_coords, nugget, partial_sill, range_param)

    # Generate regular grid within bounding box
    union_geom = boundary_proj.union_all()
    minx, miny, maxx, maxy = union_geom.bounds
    xs = np.arange(minx + resolution / 2, maxx, resolution)
    ys = np.arange(miny + resolution / 2, maxy, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Solve kriging for each prediction point
    predicted = np.full(len(grid_coords), np.nan)
    variance = np.full(len(grid_coords), np.nan)
    sill = nugget + partial_sill

    # Pre-factorize K for efficiency
    try:
        K_inv = np.linalg.solve(K, np.eye(n + 1))
    except np.linalg.LinAlgError:
        # Add small diagonal regularization
        K += np.eye(n + 1) * 1e-6
        K_inv = np.linalg.solve(K, np.eye(n + 1))

    for i, (px, py) in enumerate(grid_coords):
        dist_to_obs = np.sqrt((obs_coords[:, 0] - px) ** 2 + (obs_coords[:, 1] - py) ** 2)
        c0 = _exponential_covariance(dist_to_obs, nugget, partial_sill, range_param)

        # Right-hand side (augmented)
        rhs = np.zeros(n + 1)
        rhs[:n] = c0
        rhs[n] = 1.0

        weights = K_inv @ rhs

        predicted[i] = weights[:n] @ obs_vals
        variance[i] = sill - weights[:n] @ c0 - weights[n]

    # Build grid cell polygons and clip to boundary
    half = resolution / 2
    cells = []
    preds = []
    variances = []

    for i, (px, py) in enumerate(grid_coords):
        if np.isnan(predicted[i]):
            continue
        cell = box(px - half, py - half, px + half, py + half)
        if union_geom.intersects(cell):
            cells.append(cell)
            preds.append(predicted[i])
            variances.append(max(variance[i], 0.0))

    result = gpd.GeoDataFrame(
        {"predicted": preds, "variance": variances},
        geometry=cells,
        crs=crs,
    )

    return result
