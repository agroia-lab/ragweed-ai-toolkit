"""
Tests for ragweed_toolkit.geostatistics module.

Covers exponential variogram fitting with known params and
experimental variogram computation.
"""

import numpy as np
import pytest

from ragweed_toolkit.geostatistics.variograms import (
    _exponential_model,
    compute_experimental_variogram,
    fit_exponential_model,
)

# ------------------------------------------------------------------ #
# Helper: generate synthetic variogram data
# ------------------------------------------------------------------ #

def _generate_synthetic_variogram(nugget, partial_sill, range_param, n_lags=20, noise_frac=0.02):
    """Generate synthetic experimental variogram data from known exponential params.

    Returns lag_centers and gamma arrays with small noise added.
    """
    max_lag = range_param * 3
    lag_centers = np.linspace(max_lag / n_lags / 2, max_lag, n_lags)
    gamma_clean = nugget + partial_sill * (1 - np.exp(-lag_centers / range_param))
    rng = np.random.RandomState(42)
    noise = rng.normal(0, noise_frac * partial_sill, size=n_lags)
    gamma = gamma_clean + noise
    gamma = np.maximum(gamma, 0)  # ensure non-negative
    return lag_centers, gamma


# ------------------------------------------------------------------ #
# Exponential model function
# ------------------------------------------------------------------ #

class TestExponentialModel:
    """Test the raw exponential model function."""

    def test_at_zero(self):
        # gamma(0) = C0 + C * (1 - exp(0)) = C0
        val = _exponential_model(np.array([0.0]), 100, 400, 50)
        assert val[0] == pytest.approx(100.0, abs=0.01)

    def test_at_large_distance(self):
        # gamma(inf) -> C0 + C = sill
        val = _exponential_model(np.array([1e6]), 100, 400, 50)
        assert val[0] == pytest.approx(500.0, abs=0.1)

    def test_at_range(self):
        # gamma(a) = C0 + C * (1 - exp(-1)) ~ C0 + 0.632 * C
        val = _exponential_model(np.array([50.0]), 100, 400, 50)
        expected = 100 + 400 * (1 - np.exp(-1))
        assert val[0] == pytest.approx(expected, abs=0.01)


# ------------------------------------------------------------------ #
# Variogram fitting — recover known parameters
# ------------------------------------------------------------------ #

class TestFitExponentialModel:
    """Fit exponential model and verify recovered params within 20%."""

    @pytest.fixture
    def known_params(self):
        return {"nugget": 100, "partial_sill": 400, "range_param": 50}

    def test_recover_nugget(self, known_params):
        lag_centers, gamma = _generate_synthetic_variogram(
            known_params["nugget"],
            known_params["partial_sill"],
            known_params["range_param"],
        )
        model = fit_exponential_model(lag_centers, gamma)
        assert model.nugget == pytest.approx(known_params["nugget"], rel=0.20)

    def test_recover_partial_sill(self, known_params):
        lag_centers, gamma = _generate_synthetic_variogram(
            known_params["nugget"],
            known_params["partial_sill"],
            known_params["range_param"],
        )
        model = fit_exponential_model(lag_centers, gamma)
        assert model.partial_sill == pytest.approx(known_params["partial_sill"], rel=0.20)

    def test_recover_range(self, known_params):
        lag_centers, gamma = _generate_synthetic_variogram(
            known_params["nugget"],
            known_params["partial_sill"],
            known_params["range_param"],
        )
        model = fit_exponential_model(lag_centers, gamma)
        assert model.range_param == pytest.approx(known_params["range_param"], rel=0.20)

    def test_sill_equals_nugget_plus_partial(self, known_params):
        lag_centers, gamma = _generate_synthetic_variogram(
            known_params["nugget"],
            known_params["partial_sill"],
            known_params["range_param"],
        )
        model = fit_exponential_model(lag_centers, gamma)
        assert model.sill == pytest.approx(model.nugget + model.partial_sill, abs=0.01)

    def test_model_predict(self, known_params):
        lag_centers, gamma = _generate_synthetic_variogram(
            known_params["nugget"],
            known_params["partial_sill"],
            known_params["range_param"],
        )
        model = fit_exponential_model(lag_centers, gamma)
        predicted = model.predict(np.array([0.0, 50.0, 1000.0]))
        assert len(predicted) == 3
        # At h=0, prediction should be near nugget
        assert predicted[0] == pytest.approx(model.nugget, abs=1)

    def test_too_few_lags_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            fit_exponential_model(
                np.array([1.0, 2.0]),
                np.array([100.0, 200.0]),
            )


# ------------------------------------------------------------------ #
# Experimental variogram computation
# ------------------------------------------------------------------ #

class TestComputeExperimentalVariogram:
    """Test compute_experimental_variogram with synthetic spatial data."""

    @pytest.fixture
    def spatial_data(self):
        """Generate random 2D points with spatially correlated values."""
        rng = np.random.RandomState(42)
        n = 100
        coords = rng.uniform(0, 200, size=(n, 2))
        # Values with spatial structure: correlated with x-coord
        values = coords[:, 0] * 2 + rng.normal(0, 10, n)
        return coords, values

    def test_returns_correct_number_of_lags(self, spatial_data):
        coords, values = spatial_data
        n_lags = 10
        lag_centers, gamma, counts = compute_experimental_variogram(
            coords, values, n_lags=n_lags
        )
        assert len(lag_centers) == n_lags
        assert len(gamma) == n_lags
        assert len(counts) == n_lags

    def test_lag_centers_increasing(self, spatial_data):
        coords, values = spatial_data
        lag_centers, _, _ = compute_experimental_variogram(coords, values, n_lags=15)
        # Lag centers should be monotonically increasing
        assert np.all(np.diff(lag_centers) > 0)

    def test_pair_counts_nonnegative(self, spatial_data):
        coords, values = spatial_data
        _, _, counts = compute_experimental_variogram(coords, values, n_lags=10)
        assert np.all(counts >= 0)

    def test_gamma_nonnegative_where_valid(self, spatial_data):
        coords, values = spatial_data
        _, gamma, _ = compute_experimental_variogram(coords, values, n_lags=10)
        valid = ~np.isnan(gamma)
        assert np.all(gamma[valid] >= 0)

    def test_custom_max_lag(self, spatial_data):
        coords, values = spatial_data
        lag_centers, _, _ = compute_experimental_variogram(
            coords, values, n_lags=10, max_lag=50.0
        )
        assert lag_centers[-1] < 55  # should be within the max_lag range
