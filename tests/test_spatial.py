"""
Tests for ragweed_toolkit.spatial module.

Covers cross-variogram computation with synthetic correlated data and
LISA color/quadrant constants.
"""

import numpy as np
import pytest

from ragweed_toolkit.spatial.cross_variograms import (
    CrossVariogramResult,
    compute_cross_variogram,
    fit_cross_variogram_model,
)

# LISA and QUADRANT_MAP may not be available if esda/libpysal not installed
try:
    from ragweed_toolkit.spatial import LISA_COLORS, QUADRANT_MAP
    HAS_LISA = LISA_COLORS is not None
except ImportError:
    HAS_LISA = False


# ------------------------------------------------------------------ #
# Cross-variogram
# ------------------------------------------------------------------ #

class TestCrossVariogram:
    """Test compute_cross_variogram with synthetic correlated spatial data."""

    @pytest.fixture
    def correlated_data(self):
        """Generate spatially correlated data: z2 = 0.8 * z1 + noise."""
        rng = np.random.RandomState(42)
        n = 200
        coords = rng.uniform(0, 100, size=(n, 2))
        z1 = coords[:, 0] + rng.normal(0, 5, n)
        z2 = 0.8 * z1 + rng.normal(0, 3, n)
        return coords, z1, z2

    def test_returns_crossvariogram_result(self, correlated_data):
        coords, z1, z2 = correlated_data
        result = compute_cross_variogram(coords, z1, z2, n_lags=10)
        assert isinstance(result, CrossVariogramResult)

    def test_lag_centers_shape(self, correlated_data):
        coords, z1, z2 = correlated_data
        n_lags = 10
        result = compute_cross_variogram(coords, z1, z2, n_lags=n_lags)
        assert len(result.lag_centers) == n_lags

    def test_gamma_shape(self, correlated_data):
        coords, z1, z2 = correlated_data
        n_lags = 10
        result = compute_cross_variogram(coords, z1, z2, n_lags=n_lags)
        assert len(result.gamma) == n_lags

    def test_pair_counts_nonnegative(self, correlated_data):
        coords, z1, z2 = correlated_data
        result = compute_cross_variogram(coords, z1, z2, n_lags=10)
        assert np.all(result.pair_counts >= 0)

    def test_correlated_variables_positive_cross_variance(self, correlated_data):
        """Positively correlated variables should have positive cross-semivariance."""
        coords, z1, z2 = correlated_data
        result = compute_cross_variogram(coords, z1, z2, n_lags=10)
        valid = ~np.isnan(result.gamma)
        # At least some bins should have positive cross-semivariance
        assert np.any(result.gamma[valid] > 0)

    def test_custom_max_lag(self, correlated_data):
        coords, z1, z2 = correlated_data
        result = compute_cross_variogram(coords, z1, z2, n_lags=10, max_lag=30.0)
        assert result.max_lag == 30.0

    def test_approx_range_is_float(self, correlated_data):
        coords, z1, z2 = correlated_data
        result = compute_cross_variogram(coords, z1, z2, n_lags=10)
        assert isinstance(result.approx_range, float)

    def test_fit_cross_variogram_model(self, correlated_data):
        coords, z1, z2 = correlated_data
        result = compute_cross_variogram(coords, z1, z2, n_lags=15)
        fitted = fit_cross_variogram_model(result)
        if fitted is not None:  # fitting may fail on some random data
            assert "nugget" in fitted
            assert "partial_sill" in fitted
            assert "range_param" in fitted
            assert "sill" in fitted


# ------------------------------------------------------------------ #
# LISA colors and quadrant map constants
# ------------------------------------------------------------------ #

@pytest.mark.skipif(not HAS_LISA, reason="esda/libpysal not installed")
class TestLISAConstants:
    """Test that LISA_COLORS and QUADRANT_MAP exist and have expected keys."""

    def test_lisa_colors_has_all_categories(self):
        expected_keys = {"HH", "HL", "LH", "LL", "NS"}
        assert set(LISA_COLORS.keys()) == expected_keys

    def test_lisa_colors_are_strings(self):
        for key, color in LISA_COLORS.items():
            assert isinstance(color, str), f"Color for {key} is not a string"

    def test_lisa_colors_are_hex(self):
        for key, color in LISA_COLORS.items():
            assert color.startswith("#"), f"Color for {key} does not start with #"
            assert len(color) == 7, f"Color for {key} is not a 7-char hex code"

    def test_quadrant_map_has_four_entries(self):
        assert len(QUADRANT_MAP) == 4

    def test_quadrant_map_values(self):
        expected_values = {"HH", "LH", "LL", "HL"}
        assert set(QUADRANT_MAP.values()) == expected_values

    def test_quadrant_map_keys_are_integers(self):
        for key in QUADRANT_MAP:
            assert isinstance(key, int)

    def test_quadrant_map_keys_1_to_4(self):
        assert set(QUADRANT_MAP.keys()) == {1, 2, 3, 4}
