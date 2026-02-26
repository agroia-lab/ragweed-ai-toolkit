"""
Cross-Variogram Computation
=============================

Compute the cross-semivariance between two spatial variables to quantify
how their spatial co-variation changes with distance.

Cross-semivariogram definition:

    gamma_xy(h) = 1 / (2 * N(h)) * sum[ (z1(i) - z1(j)) * (z2(i) - z2(j)) ]

for all point pairs (i, j) whose separation falls within lag bin h.

Uses ``scipy.spatial.cKDTree`` for efficient pair finding on large datasets.

Chapter reference: Section 5.4.2 (Cross-variograms).
Dependencies: numpy, scipy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree


@dataclass
class CrossVariogramResult:
    """Container for cross-variogram output."""

    lag_centers: np.ndarray
    gamma: np.ndarray
    pair_counts: np.ndarray
    var1_name: str
    var2_name: str
    max_lag: float
    n_lags: int
    approx_range: float


def compute_cross_variogram(
    coords: np.ndarray,
    z1: np.ndarray,
    z2: np.ndarray,
    n_lags: int = 15,
    max_lag: Optional[float] = None,
) -> CrossVariogramResult:
    """
    Compute the experimental cross-semivariogram between *z1* and *z2*.

    Parameters
    ----------
    coords : ndarray, shape (n, 2)
        Projected coordinates (UTM).
    z1 : ndarray, shape (n,)
        First variable values.
    z2 : ndarray, shape (n,)
        Second variable values.
    n_lags : int
        Number of distance bins.
    max_lag : float, optional
        Maximum lag distance. Defaults to the median pairwise distance
        (estimated from a random sample for large datasets).

    Returns
    -------
    CrossVariogramResult
    """
    coords = np.asarray(coords, dtype=float)
    z1 = np.asarray(z1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    n = len(z1)

    # Estimate max_lag from a sample if not provided
    if max_lag is None:
        if n > 2000:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=2000, replace=False)
            sample_coords = coords[idx]
        else:
            sample_coords = coords
        from scipy.spatial.distance import pdist
        max_lag = float(np.median(pdist(sample_coords)))

    lag_edges = np.linspace(0, max_lag, n_lags + 1)
    lag_centers = (lag_edges[:-1] + lag_edges[1:]) / 2
    gamma = np.full(n_lags, np.nan)
    counts = np.zeros(n_lags, dtype=int)

    # Use cKDTree for efficient pair finding
    tree = cKDTree(coords)

    for k in range(n_lags):
        lo, hi = lag_edges[k], lag_edges[k + 1]
        mid = (lo + hi) / 2
        radius = (hi - lo) / 2

        # Find all pairs within hi
        pairs_hi = tree.query_pairs(r=hi, output_type="ndarray")
        if len(pairs_hi) == 0:
            continue

        # Compute actual distances for these pairs
        dists = np.sqrt(
            (coords[pairs_hi[:, 0], 0] - coords[pairs_hi[:, 1], 0]) ** 2
            + (coords[pairs_hi[:, 0], 1] - coords[pairs_hi[:, 1], 1]) ** 2
        )

        # Filter to this lag bin
        in_bin = (dists > lo) & (dists <= hi)
        bin_pairs = pairs_hi[in_bin]
        npairs = len(bin_pairs)
        counts[k] = npairs

        if npairs > 0:
            i_idx = bin_pairs[:, 0]
            j_idx = bin_pairs[:, 1]
            diffs = (z1[i_idx] - z1[j_idx]) * (z2[i_idx] - z2[j_idx])
            gamma[k] = np.sum(diffs) / (2.0 * npairs)

    # Approximate range (distance at which 90% of sill is reached)
    approx_range = _estimate_range(lag_centers, gamma)

    return CrossVariogramResult(
        lag_centers=lag_centers,
        gamma=gamma,
        pair_counts=counts,
        var1_name="z1",
        var2_name="z2",
        max_lag=max_lag,
        n_lags=n_lags,
        approx_range=approx_range,
    )


def _estimate_range(lags: np.ndarray, gammas: np.ndarray) -> float:
    """Estimate the practical range as the lag where 90% of the sill is reached."""
    valid = ~np.isnan(gammas)
    if not valid.any():
        return float("nan")

    abs_gammas = np.abs(gammas[valid])
    sill = np.max(abs_gammas) if len(abs_gammas) > 0 else 0
    if sill == 0:
        return float("nan")

    valid_lags = lags[valid]
    valid_gammas = gammas[valid]
    for i, g in enumerate(valid_gammas):
        if abs(g) >= 0.9 * sill:
            return float(valid_lags[i])

    return float(valid_lags[-1])


def fit_cross_variogram_model(
    result: CrossVariogramResult,
) -> dict:
    """
    Fit an exponential model to the cross-variogram.

    Returns
    -------
    dict
        Keys: ``nugget``, ``partial_sill``, ``range_param``, ``sill``.
        Returns None if fitting fails.
    """
    valid = ~np.isnan(result.gamma)
    h = result.lag_centers[valid]
    g = result.gamma[valid]

    if len(h) < 3:
        return None

    def _exp_model(h, C0, C, a):
        return C0 + C * (1 - np.exp(-h / a))

    p0 = [float(g[0]), float(np.max(np.abs(g)) - g[0]), float(h[len(h) // 2])]
    try:
        popt, _ = curve_fit(_exp_model, h, g, p0=p0, maxfev=10000)
        return {
            "nugget": float(popt[0]),
            "partial_sill": float(popt[1]),
            "range_param": float(popt[2]),
            "sill": float(popt[0] + popt[1]),
        }
    except RuntimeError:
        return None
