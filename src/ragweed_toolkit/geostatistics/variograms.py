"""
Semivariogram Computation and Model Fitting
=============================================

Compute experimental semivariograms from point observations and fit
parametric models (exponential) for use in kriging interpolation.

Exponential model (Chapter 5, Section 5.4.2):

    gamma(h) = C0 + C * [1 - exp(-h / a)]

where:
    C0 = nugget (micro-scale variance + measurement error)
    C  = partial sill (spatially structured variance)
    a  = practical range (distance at which ~63% of sill is reached)

Reference parameters from chapter (Table 6):
    nugget = 43,066.5
    sill   = 86,133.0  (nugget + partial_sill)
    range  = 81.2 m
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform


@dataclass
class VariogramModel:
    """Fitted exponential variogram parameters."""

    nugget: float
    partial_sill: float
    range_param: float
    sill: float
    lag_centers: np.ndarray
    experimental_gamma: np.ndarray
    pair_counts: np.ndarray

    def predict(self, h: np.ndarray | float) -> np.ndarray:
        """Evaluate the fitted model at distance(s) *h*."""
        h = np.asarray(h, dtype=float)
        return self.nugget + self.partial_sill * (1 - np.exp(-h / self.range_param))


# --------------------------------------------------------------------------- #
# Core functions
# --------------------------------------------------------------------------- #

def compute_experimental_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    n_lags: int = 15,
    max_lag: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the experimental (isotropic) semivariogram.

    Parameters
    ----------
    coords : ndarray, shape (n, 2)
        Projected coordinates (UTM easting, northing).
    values : ndarray, shape (n,)
        Observed values at each point.
    n_lags : int
        Number of distance bins.
    max_lag : float, optional
        Maximum lag distance.  Defaults to the median pairwise distance.

    Returns
    -------
    lag_centers : ndarray, shape (n_lags,)
    gamma : ndarray, shape (n_lags,)
        Semi-variance for each bin (NaN where no pairs exist).
    counts : ndarray, shape (n_lags,)
        Number of point pairs in each bin.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)

    dist_vec = pdist(coords)
    if max_lag is None:
        max_lag = float(np.median(dist_vec))

    dist_matrix = squareform(dist_vec)
    lag_edges = np.linspace(0, max_lag, n_lags + 1)
    lag_centers = (lag_edges[:-1] + lag_edges[1:]) / 2
    gamma = np.full(n_lags, np.nan)
    counts = np.zeros(n_lags, dtype=int)

    len(values)
    for k in range(n_lags):
        lo, hi = lag_edges[k], lag_edges[k + 1]
        mask = np.triu((dist_matrix > lo) & (dist_matrix <= hi), k=1)
        idx = np.where(mask)
        npairs = len(idx[0])
        counts[k] = npairs
        if npairs > 0:
            diffs = values[idx[0]] - values[idx[1]]
            gamma[k] = np.sum(diffs ** 2) / (2.0 * npairs)

    return lag_centers, gamma, counts


def _exponential_model(h: np.ndarray, C0: float, C: float, a: float) -> np.ndarray:
    """Exponential variogram model for curve_fit."""
    return C0 + C * (1 - np.exp(-h / a))


def fit_exponential_model(
    lag_centers: np.ndarray,
    gamma: np.ndarray,
    initial_nugget: Optional[float] = None,
    initial_sill: Optional[float] = None,
    initial_range: Optional[float] = None,
) -> VariogramModel:
    """
    Fit an exponential variogram model to experimental semi-variance.

    Parameters
    ----------
    lag_centers : ndarray
        Midpoints of each lag bin.
    gamma : ndarray
        Experimental semi-variance (NaN entries are dropped).
    initial_nugget, initial_sill, initial_range : float, optional
        Starting guesses for the optimiser.  Sensible defaults are derived
        from the data when omitted.

    Returns
    -------
    VariogramModel
        Fitted model with nugget, partial_sill, range_param, sill.

    Raises
    ------
    RuntimeError
        If curve_fit fails to converge.
    """
    valid = ~np.isnan(gamma)
    h_fit = lag_centers[valid]
    g_fit = gamma[valid]

    if len(h_fit) < 3:
        raise ValueError("Need at least 3 valid lag bins to fit a variogram model.")

    # Sensible defaults
    if initial_nugget is None:
        initial_nugget = float(g_fit[0]) if len(g_fit) > 0 else 0.0
    if initial_sill is None:
        initial_sill = float(np.max(g_fit) - initial_nugget)
    if initial_range is None:
        initial_range = float(h_fit[len(h_fit) // 2])

    p0 = [max(initial_nugget, 0), max(initial_sill, 1e-6), max(initial_range, 1e-6)]
    bounds = ([0, 0, 1e-6], [np.inf, np.inf, np.inf])

    try:
        popt, _ = curve_fit(_exponential_model, h_fit, g_fit, p0=p0, bounds=bounds, maxfev=10000)
    except RuntimeError as exc:
        raise RuntimeError(f"Variogram fitting failed: {exc}") from exc

    C0, C, a = popt
    # Reconstruct full arrays (NaN-padded)
    pair_counts = np.zeros_like(lag_centers, dtype=int)  # placeholder

    return VariogramModel(
        nugget=float(C0),
        partial_sill=float(C),
        range_param=float(a),
        sill=float(C0 + C),
        lag_centers=lag_centers,
        experimental_gamma=gamma,
        pair_counts=pair_counts,
    )


def plot_variogram(
    model: VariogramModel,
    *,
    title: str = "Semivariogram",
    ax=None,
    show: bool = False,
):
    """
    Plot experimental points and fitted exponential curve.

    Parameters
    ----------
    model : VariogramModel
        Fitted variogram returned by :func:`fit_exponential_model`.
    title : str
        Plot title.
    ax : matplotlib Axes, optional
        Existing axes to draw on.  A new figure is created when *None*.
    show : bool
        Call ``plt.show()`` after drawing.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    valid = ~np.isnan(model.experimental_gamma)
    ax.scatter(
        model.lag_centers[valid],
        model.experimental_gamma[valid],
        c="#3498db",
        s=50,
        edgecolors="k",
        linewidths=0.5,
        zorder=3,
        label="Experimental",
    )

    h_smooth = np.linspace(0, model.lag_centers[-1], 200)
    ax.plot(
        h_smooth,
        model.predict(h_smooth),
        color="#e74c3c",
        linewidth=2,
        label=(
            f"Exponential  (C0={model.nugget:.0f}, "
            f"C={model.partial_sill:.0f}, a={model.range_param:.1f}m)"
        ),
    )

    ax.axhline(
        model.sill, color="gray", linestyle="--", linewidth=0.8,
        label=f"Sill = {model.sill:.0f}",
    )
    ax.set_xlabel("Lag distance (m)", fontsize=11)
    ax.set_ylabel("Semivariance", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()

    return ax
