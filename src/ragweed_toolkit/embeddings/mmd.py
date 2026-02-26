"""
Maximum Mean Discrepancy (MMD) with Deployment Gate
=====================================================

Computes the MMD between two sets of embeddings using a Gaussian RBF kernel,
with bandwidth selected via the median heuristic. Includes a permutation test
for statistical significance and a deployment gate that maps MMD values to
actionable risk tiers.

This implements the domain shift framework from Chapter 6:

    MMD^2(X, Y) = (1/m^2) sum_ij k(xi, xj)
                + (1/n^2) sum_ij k(yi, yj)
                - (2/mn)  sum_ij k(xi, yj)

where k(a, b) = exp(-||a-b||^2 / (2 sigma^2)) is the RBF kernel and
sigma = median pairwise Euclidean distance (Eq. 4).

Deployment gate thresholds (Table 3):

    MMD < 0.15   -> green  : Deploy directly, minimal domain shift
    0.15 - 0.30  -> yellow : Deploy with augmentation recommended
    0.30 - 0.45  -> orange : Active learning required
    > 0.45       -> red    : Collect substantial new training data

Usage::

    from ragweed_toolkit.embeddings import compute_mmd, permutation_test, deployment_gate

    mmd_val = compute_mmd(X, Y)
    mmd_val, p_value = permutation_test(X, Y)
    tier, recommendation = deployment_gate(mmd_val)
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import numpy as np


class DeploymentResult(NamedTuple):
    """Result from the deployment gate.

    Attributes
    ----------
    tier : str
        Risk tier: ``"green"``, ``"yellow"``, ``"orange"``, or ``"red"``.
    recommendation : str
        Human-readable recommendation.
    mmd : float
        The MMD value used for the decision.
    """

    tier: str
    recommendation: str
    mmd: float


# Deployment gate thresholds and recommendations (Chapter 6, Table 3)
_THRESHOLDS = [
    (0.15, "green", "Deploy directly - minimal domain shift"),
    (0.30, "yellow", "Deploy with augmentation recommended"),
    (0.45, "orange", "Active learning required"),
    (float("inf"), "red", "Collect substantial new training data"),
]


def _compute_gamma(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute RBF kernel bandwidth using the median heuristic.

    sigma = median of pairwise Euclidean distances on the combined set,
    gamma = 1 / (2 * sigma^2).

    Parameters
    ----------
    X, Y : np.ndarray
        Embedding matrices.

    Returns
    -------
    float
        The gamma parameter for the RBF kernel.
    """
    from sklearn.metrics.pairwise import euclidean_distances

    combined = np.vstack([X, Y])
    dists = euclidean_distances(combined, combined)
    median_dist = np.median(dists[dists > 0])
    return 1.0 / (2.0 * median_dist ** 2)


def compute_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: Optional[float] = None,
) -> float:
    """Compute Maximum Mean Discrepancy between two embedding sets.

    Uses the unbiased estimator with an RBF kernel. If *gamma* is not
    provided, bandwidth is selected via the median heuristic.

    Parameters
    ----------
    X : np.ndarray
        First set of embeddings, shape ``(m, d)``.
    Y : np.ndarray
        Second set of embeddings, shape ``(n, d)``.
    gamma : float, optional
        RBF kernel bandwidth. If ``None``, uses median heuristic.

    Returns
    -------
    float
        MMD value (non-negative). Higher means more different distributions.
    """
    from sklearn.metrics.pairwise import rbf_kernel

    if gamma is None:
        gamma = _compute_gamma(X, Y)

    m = len(X)
    n = len(Y)

    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)

    # Unbiased estimator: exclude diagonal
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    mmd_squared = (
        Kxx.sum() / (m * (m - 1))
        + Kyy.sum() / (n * (n - 1))
        - 2.0 * Kxy.sum() / (m * n)
    )

    return float(max(0.0, mmd_squared) ** 0.5)


def permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    gamma: Optional[float] = None,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Compute MMD with a permutation test for statistical significance.

    Under H0 (identical distributions), the labels (X vs Y) are randomly
    shuffled. The p-value is the fraction of permuted MMDs >= observed MMD.

    Parameters
    ----------
    X : np.ndarray
        First set of embeddings, shape ``(m, d)``.
    Y : np.ndarray
        Second set of embeddings, shape ``(n, d)``.
    gamma : float, optional
        RBF kernel bandwidth. If ``None``, computed once from the combined set.
    n_permutations : int
        Number of permutations (default 1000).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, float]
        ``(mmd_value, p_value)`` where p_value is in [0, 1].
    """
    if gamma is None:
        gamma = _compute_gamma(X, Y)

    observed_mmd = compute_mmd(X, Y, gamma=gamma)

    combined = np.vstack([X, Y])
    m = len(X)
    total = len(combined)
    rng = np.random.RandomState(random_state)

    count_ge = 0
    for _ in range(n_permutations):
        perm = rng.permutation(total)
        X_perm = combined[perm[:m]]
        Y_perm = combined[perm[m:]]
        perm_mmd = compute_mmd(X_perm, Y_perm, gamma=gamma)
        if perm_mmd >= observed_mmd:
            count_ge += 1

    # Continuity correction
    p_value = (count_ge + 1) / (n_permutations + 1)

    return observed_mmd, p_value


def deployment_gate(mmd_value: float) -> DeploymentResult:
    """Map an MMD value to a deployment risk tier.

    Parameters
    ----------
    mmd_value : float
        Observed MMD between a target domain and the training distribution.

    Returns
    -------
    DeploymentResult
        Named tuple with ``tier``, ``recommendation``, and ``mmd`` fields.

    Examples
    --------
    >>> result = deployment_gate(0.10)
    >>> result.tier
    'green'
    >>> result = deployment_gate(0.25)
    >>> result.tier
    'yellow'
    >>> result = deployment_gate(0.50)
    >>> result.tier
    'red'
    """
    for threshold, tier, recommendation in _THRESHOLDS:
        if mmd_value < threshold:
            return DeploymentResult(
                tier=tier,
                recommendation=recommendation,
                mmd=mmd_value,
            )
    # Should not reach here, but fallback to red
    return DeploymentResult(
        tier="red",
        recommendation=_THRESHOLDS[-1][2],
        mmd=mmd_value,
    )
