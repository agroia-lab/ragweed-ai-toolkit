"""
Geographically Weighted Regression (GWR)
==========================================

Wrapper around ``mgwr.gwr.GWR`` with adaptive bisquare kernel for modelling
spatially varying relationships between weed density and environmental
predictors (e.g. Presto PCA components, soil indices).

Model form (Chapter 5, Section 5.4.3):

    d_i = beta_0(u_i, v_i) + beta_1(u_i, v_i) * PC1 + ... + epsilon_i

where (u_i, v_i) are the spatial coordinates and the betas vary continuously
across the study area.

The module also provides an OLS baseline (``spreg.OLS``) so that R^2 and AICc
can be compared side-by-side.

Dependencies: mgwr, spreg, numpy, geopandas, sklearn
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import geopandas as gpd
from sklearn.preprocessing import StandardScaler

from mgwr.gwr import GWR as _GWR
from mgwr.sel_bw import Sel_BW
from spreg import OLS


@dataclass
class OLSResult:
    """Summary of an OLS regression fit."""

    r2: float
    adj_r2: float
    aic: float
    n: int
    k: int
    coefficients: dict[str, float]


@dataclass
class GWRResult:
    """Summary of a GWR fit with local coefficient arrays."""

    r2: float
    adj_r2: float
    aicc: float
    bandwidth: float
    n: int
    k: int
    local_r2: np.ndarray
    local_betas: dict[str, np.ndarray]
    residuals: np.ndarray


@dataclass
class ModelComparison:
    """Side-by-side OLS vs GWR metrics."""

    ols: OLSResult
    gwr: GWRResult
    r2_improvement: float
    aicc_delta: float


def fit_ols(
    gdf: gpd.GeoDataFrame,
    y_col: str,
    x_cols: Sequence[str],
    *,
    standardise: bool = True,
) -> OLSResult:
    """
    Fit an OLS regression baseline.

    Parameters
    ----------
    gdf : GeoDataFrame
        Observations with columns *y_col* and each of *x_cols*.
    y_col : str
        Dependent variable column.
    x_cols : sequence of str
        Predictor column names.
    standardise : bool
        Whether to z-score standardise predictors before fitting.

    Returns
    -------
    OLSResult
    """
    df = gdf.dropna(subset=[y_col] + list(x_cols)).copy()

    y = df[y_col].values.reshape(-1, 1)
    X = df[x_cols].values

    if standardise:
        X = StandardScaler().fit_transform(X)

    ols = OLS(y, X, name_y=y_col, name_x=list(x_cols))

    coefficients = {"intercept": float(ols.betas[0, 0])}
    for i, name in enumerate(x_cols):
        coefficients[name] = float(ols.betas[i + 1, 0])

    return OLSResult(
        r2=float(ols.r2),
        adj_r2=float(ols.ar2),
        aic=float(ols.aic),
        n=int(ols.n),
        k=int(ols.k),
        coefficients=coefficients,
    )


def fit_gwr(
    gdf: gpd.GeoDataFrame,
    y_col: str,
    x_cols: Sequence[str],
    *,
    standardise: bool = True,
    kernel: str = "bisquare",
    fixed: bool = False,
    criterion: str = "AICc",
) -> GWRResult:
    """
    Fit a GWR model with adaptive bisquare kernel and AICc bandwidth selection.

    Parameters
    ----------
    gdf : GeoDataFrame
        Point observations in a projected CRS.
    y_col : str
        Dependent variable column.
    x_cols : sequence of str
        Predictor column names.
    standardise : bool
        Whether to z-score standardise predictors.
    kernel : str
        Kernel type for ``mgwr`` (default ``"bisquare"``).
    fixed : bool
        ``True`` for fixed bandwidth (metres), ``False`` for adaptive
        (nearest neighbours).
    criterion : str
        Bandwidth selection criterion (``"AICc"`` or ``"AIC"``).

    Returns
    -------
    GWRResult
    """
    df = gdf.dropna(subset=[y_col] + list(x_cols)).copy()

    coords = np.column_stack([
        [geom.x for geom in df.geometry],
        [geom.y for geom in df.geometry],
    ])
    y = df[y_col].values.reshape(-1, 1)
    X = df[x_cols].values

    if standardise:
        X = StandardScaler().fit_transform(X)

    # Bandwidth selection
    selector = Sel_BW(coords, y, X, kernel=kernel, fixed=fixed)
    bw = selector.search(criterion=criterion, search_method="golden_section")

    # Fit
    model = _GWR(coords, y, X, bw=bw, kernel=kernel, fixed=fixed)
    results = model.fit()

    # Local betas
    local_betas = {"intercept": results.params[:, 0]}
    for i, name in enumerate(x_cols):
        local_betas[name] = results.params[:, i + 1]

    return GWRResult(
        r2=float(results.R2),
        adj_r2=float(results.adj_R2),
        aicc=float(results.aicc),
        bandwidth=float(bw),
        n=len(y),
        k=X.shape[1],
        local_r2=results.localR2.flatten(),
        local_betas=local_betas,
        residuals=results.resid_response.flatten(),
    )


def compare_ols_gwr(
    gdf: gpd.GeoDataFrame,
    y_col: str,
    x_cols: Sequence[str],
    **gwr_kwargs,
) -> ModelComparison:
    """
    Fit both OLS and GWR and return a side-by-side comparison.

    Parameters
    ----------
    gdf : GeoDataFrame
        Point observations in a projected CRS.
    y_col : str
        Dependent variable column.
    x_cols : sequence of str
        Predictor column names.
    **gwr_kwargs
        Additional keyword arguments forwarded to :func:`fit_gwr`.

    Returns
    -------
    ModelComparison
        Contains ``ols``, ``gwr``, ``r2_improvement``, and ``aicc_delta``.
    """
    ols_result = fit_ols(gdf, y_col, x_cols)
    gwr_result = fit_gwr(gdf, y_col, x_cols, **gwr_kwargs)

    return ModelComparison(
        ols=ols_result,
        gwr=gwr_result,
        r2_improvement=gwr_result.r2 - ols_result.r2,
        aicc_delta=gwr_result.aicc - ols_result.aic,
    )
