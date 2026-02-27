"""
Tests for ragweed_toolkit.satellite module.

Covers compute_spectral_indices with known band values.
"""

import numpy as np
import pytest

from ragweed_toolkit.satellite.indices import (
    BAND_INDEX,
    BAND_NAMES,
    INDEX_DESCRIPTIONS,
    compute_spectral_indices,
)

# ------------------------------------------------------------------ #
# Helper: build a 10-band Sentinel-2 pixel array
# ------------------------------------------------------------------ #

def _make_spectral_array(
    B2=0.1, B3=0.1, B4=0.1, B5=0.1, B6=0.1, B7=0.1,
    B8=0.1, B8A=0.1, B11=0.1, B12=0.1,
    h=1, w=1,
):
    """Create a (10, H, W) array with specified reflectance values."""
    arr = np.zeros((10, h, w), dtype=float)
    bands = [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]
    for i, val in enumerate(bands):
        arr[i, :, :] = val
    return arr


# ------------------------------------------------------------------ #
# NDVI — known vegetation values
# ------------------------------------------------------------------ #

class TestNDVI:
    """Test NDVI computation with known band values."""

    def test_pure_vegetation(self):
        """Pure vegetation: NIR=0.8, RED=0.1 -> NDVI ~ 0.778."""
        spectral = _make_spectral_array(B4=0.1, B8=0.8)
        result = compute_spectral_indices(spectral, indices=["NDVI"])
        ndvi = result["NDVI"][0, 0]
        expected = (0.8 - 0.1) / (0.8 + 0.1)
        assert ndvi == pytest.approx(expected, abs=0.001)
        assert abs(ndvi - 0.778) < 0.01

    def test_bare_soil(self):
        """Bare soil: NIR ~ RED -> NDVI ~ 0."""
        spectral = _make_spectral_array(B4=0.2, B8=0.2)
        result = compute_spectral_indices(spectral, indices=["NDVI"])
        ndvi = result["NDVI"][0, 0]
        assert abs(ndvi) < 0.01

    def test_water(self):
        """Water: RED > NIR -> NDVI < 0."""
        spectral = _make_spectral_array(B4=0.3, B8=0.05)
        result = compute_spectral_indices(spectral, indices=["NDVI"])
        ndvi = result["NDVI"][0, 0]
        assert ndvi < 0


# ------------------------------------------------------------------ #
# All 9 indices return valid floats
# ------------------------------------------------------------------ #

class TestAllIndices:
    """Test that all 9 spectral indices return valid float values."""

    @pytest.fixture
    def vegetation_spectral(self):
        """Typical vegetation reflectance values."""
        return _make_spectral_array(
            B2=0.04, B3=0.06, B4=0.05, B5=0.12,
            B6=0.25, B7=0.30, B8=0.35, B8A=0.33,
            B11=0.15, B12=0.08,
        )

    def test_all_indices_computed(self, vegetation_spectral):
        result = compute_spectral_indices(vegetation_spectral)
        assert len(result) == 9

    @pytest.mark.parametrize("index_name", list(INDEX_DESCRIPTIONS.keys()))
    def test_index_returns_float(self, vegetation_spectral, index_name):
        result = compute_spectral_indices(vegetation_spectral, indices=[index_name])
        val = result[index_name][0, 0]
        assert isinstance(float(val), float)
        assert np.isfinite(val)

    def test_ndvi_in_expected_range(self, vegetation_spectral):
        result = compute_spectral_indices(vegetation_spectral, indices=["NDVI"])
        ndvi = result["NDVI"][0, 0]
        assert -1.0 <= ndvi <= 1.0

    def test_evi_returns_float(self, vegetation_spectral):
        result = compute_spectral_indices(vegetation_spectral, indices=["EVI"])
        evi = result["EVI"][0, 0]
        assert np.isfinite(evi)

    def test_gndvi_returns_float(self, vegetation_spectral):
        result = compute_spectral_indices(vegetation_spectral, indices=["GNDVI"])
        gndvi = result["GNDVI"][0, 0]
        assert np.isfinite(gndvi)

    def test_clay_ratio(self, vegetation_spectral):
        """Clay = B11/B12, should be > 1 for typical vegetation."""
        result = compute_spectral_indices(vegetation_spectral, indices=["Clay"])
        clay = result["Clay"][0, 0]
        assert clay > 0

    def test_swird_difference(self, vegetation_spectral):
        """SWIRd = B11 - B12."""
        result = compute_spectral_indices(vegetation_spectral, indices=["SWIRd"])
        swird = result["SWIRd"][0, 0]
        expected = 0.15 - 0.08  # B11 - B12
        assert swird == pytest.approx(expected, abs=0.001)


# ------------------------------------------------------------------ #
# Edge cases
# ------------------------------------------------------------------ #

class TestSpectralEdgeCases:
    """Test edge cases for spectral index computation."""

    def test_wrong_band_count_raises(self):
        bad_arr = np.zeros((5, 10, 10))
        with pytest.raises(ValueError, match="Expected 10-band"):
            compute_spectral_indices(bad_arr)

    def test_unknown_index_raises(self):
        spectral = _make_spectral_array()
        with pytest.raises(ValueError, match="Unknown index"):
            compute_spectral_indices(spectral, indices=["FAKE_INDEX"])

    def test_valid_mask_applies_nan(self):
        """Pixels marked invalid should be NaN in output."""
        spectral = _make_spectral_array(B4=0.1, B8=0.8, h=3, w=3)
        valid = np.ones((3, 3), dtype=bool)
        valid[1, 1] = False
        result = compute_spectral_indices(spectral, valid=valid, indices=["NDVI"])
        assert np.isnan(result["NDVI"][1, 1])
        assert np.isfinite(result["NDVI"][0, 0])

    def test_multi_pixel_array(self):
        """Test with a multi-pixel array."""
        spectral = _make_spectral_array(B4=0.1, B8=0.8, h=10, w=10)
        result = compute_spectral_indices(spectral, indices=["NDVI"])
        assert result["NDVI"].shape == (10, 10)
        # All pixels should have same NDVI since all bands are uniform
        assert np.allclose(result["NDVI"], result["NDVI"][0, 0])

    def test_band_names_and_index(self):
        """Verify BAND_NAMES and BAND_INDEX are consistent."""
        assert len(BAND_NAMES) == 10
        for name in BAND_NAMES:
            assert name in BAND_INDEX
        for name, idx in BAND_INDEX.items():
            assert BAND_NAMES[idx] == name

    def test_index_descriptions_count(self):
        """All 9 indices should have descriptions."""
        assert len(INDEX_DESCRIPTIONS) == 9
