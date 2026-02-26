"""
Bivariate Moran's I
=====================

Global bivariate spatial autocorrelation between two variables observed at
the same locations.  Measures whether high values of variable *x* at a
location tend to be surrounded by high (or low) values of variable *y*.

Formula (Anselin, 2019; Chapter 5, Section 5.4.1):

    I_xy = (n / sum(w_ij)) * (sum_ij w_ij * z_xi * z_yj) / sum(z_xi^2)

where z_xi and z_yj are standardised deviates and w_ij are spatial weights.

Dependencies: esda, libpysal, geopandas, numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import geopandas as gpd
from libpysal.weights import DistanceBand
from esda.moran import Moran_BV


@dataclass
class BivariateMoranResult:
    """Result container for bivariate global Moran's I."""

    I: float
    p_value: float
    z_score: float
    permutations: int
    var_x: str
    var_y: str
    n: int
    threshold: float

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    @property
    def significance_stars(self) -> str:
        if self.p_value <= 0.001:
            return "***"
        if self.p_value < 0.01:
            return "**"
        if self.p_value < 0.05:
            return "*"
        return "ns"


def build_distance_weights(
    gdf: gpd.GeoDataFrame,
    threshold: float = 15.0,
    *,
    x_col: str = "geometry",
    max_retries: int = 10,
) -> DistanceBand:
    """
    Build a row-standardised DistanceBand weight matrix.

    If isolated observations (islands) exist at the requested threshold,
    the threshold is increased by 5 m and retried up to *max_retries* times.

    Parameters
    ----------
    gdf : GeoDataFrame
        Point geometries in a projected CRS (metres).
    threshold : float
        Distance cutoff in metres.
    x_col : str
        Ignored (coords extracted from geometry).
    max_retries : int
        Maximum number of automatic threshold increases.

    Returns
    -------
    DistanceBand
        Row-standardised spatial weight matrix.
    """
    coords = [(geom.x, geom.y) for geom in gdf.geometry]

    for attempt in range(max_retries):
        w = DistanceBand(coords, threshold=threshold, binary=True, silence_warnings=True)
        w.transform = "r"
        if len(w.islands) == 0:
            return w
        threshold += 5.0

    raise ValueError(
        f"Could not eliminate islands after {max_retries} retries "
        f"(last threshold={threshold:.0f} m)."
    )


def compute_bivariate_morans(
    gdf: gpd.GeoDataFrame,
    var_x: str,
    var_y: str,
    threshold: float = 15.0,
    permutations: int = 999,
    w: Optional[DistanceBand] = None,
) -> BivariateMoranResult:
    """
    Compute the bivariate global Moran's I between two variables.

    Parameters
    ----------
    gdf : GeoDataFrame
        Point observations in a projected CRS with columns *var_x* and *var_y*.
    var_x : str
        Column name for the first variable (spatially lagged).
    var_y : str
        Column name for the second variable (focal).
    threshold : float
        Distance threshold for the DistanceBand weight matrix (metres).
    permutations : int
        Number of random permutations for pseudo p-value.
    w : DistanceBand, optional
        Pre-computed weight matrix.  Built from *gdf* when *None*.

    Returns
    -------
    BivariateMoranResult
    """
    # Drop NaN rows
    mask = gdf[var_x].notna() & gdf[var_y].notna()
    gdf_valid = gdf[mask].reset_index(drop=True)

    if len(gdf_valid) < 10:
        raise ValueError(f"Too few valid observations ({len(gdf_valid)}) for Moran's I.")

    if w is None:
        w = build_distance_weights(gdf_valid, threshold=threshold)

    x = gdf_valid[var_x].values
    y = gdf_valid[var_y].values

    bv = Moran_BV(x, y, w, permutations=permutations)

    return BivariateMoranResult(
        I=float(bv.I),
        p_value=float(bv.p_sim),
        z_score=float(bv.z_sim),
        permutations=permutations,
        var_x=var_x,
        var_y=var_y,
        n=len(gdf_valid),
        threshold=threshold,
    )
