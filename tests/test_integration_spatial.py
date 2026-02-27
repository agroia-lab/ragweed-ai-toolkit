"""
Integration tests for spatial and satellite pipelines with real data.

These tests verify end-to-end functionality using actual data files
from the LenCu Santa Rosa field study. Tests that require external
data are skipped when the data is not available.

Run with:
    pytest tests/test_integration_spatial.py -v
    pytest tests/test_integration_spatial.py -v -k "not slow"
"""

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Data availability markers
# ---------------------------------------------------------------------------

KRIGING_DIR = Path("/media/malezainia2/E/lencu_santarosa_full/kriging")
PRESTO_SHP = Path("/home/malezainia2/Downloads/presto_PC1.shp")

has_kriging_data = pytest.mark.skipif(
    not KRIGING_DIR.exists(),
    reason=f"Kriging data not found at {KRIGING_DIR}",
)
has_presto_data = pytest.mark.skipif(
    not PRESTO_SHP.exists(),
    reason=f"PRESTO shapefile not found at {PRESTO_SHP}",
)


# ---------------------------------------------------------------------------
# Kriging raster tests
# ---------------------------------------------------------------------------

VARIOGRAM_MODELS = ["exponential", "gaussian", "linear", "spherical"]


@has_kriging_data
class TestKrigingRasters:
    """Verify kriging raster outputs from the Santa Rosa density analysis."""

    def _raster_path(self, model: str) -> Path:
        return KRIGING_DIR / f"kriging_total_det_{model}_5m.tif"

    def test_kriging_all_models_exist(self):
        """All four variogram model rasters should be present."""
        for model in VARIOGRAM_MODELS:
            path = self._raster_path(model)
            assert path.exists(), f"Missing kriging raster: {path.name}"

    def test_kriging_raster_loading(self):
        """Load exponential kriging raster and verify shape, CRS, and stats."""
        rasterio = pytest.importorskip("rasterio")

        path = self._raster_path("exponential")
        with rasterio.open(path) as src:
            data = src.read(1)
            # Shape should be non-trivial
            assert data.shape[0] > 10, f"Too few rows: {data.shape[0]}"
            assert data.shape[1] > 10, f"Too few cols: {data.shape[1]}"
            # CRS must be UTM zone 19S (Chile)
            assert src.crs is not None
            assert src.crs.to_epsg() == 32719, f"Expected EPSG:32719, got {src.crs}"
            # Max density should be positive (detections per unit area)
            valid = data[data > -9999]
            assert len(valid) > 0, "No valid (non-nodata) pixels"
            assert valid.max() > 0, "Max density should be positive"

    @pytest.mark.parametrize("model", VARIOGRAM_MODELS)
    def test_kriging_raster_crs_consistent(self, model):
        """All model rasters should share the same CRS (EPSG:32719)."""
        rasterio = pytest.importorskip("rasterio")

        path = self._raster_path(model)
        if not path.exists():
            pytest.skip(f"{model} raster not found")
        with rasterio.open(path) as src:
            assert src.crs.to_epsg() == 32719


# ---------------------------------------------------------------------------
# PRESTO + LISA tests
# ---------------------------------------------------------------------------


@has_presto_data
class TestPrestoShapefile:
    """Verify PRESTO PC1 shapefile structure and bivariate LISA analysis."""

    @pytest.fixture(scope="class")
    def presto_gdf(self):
        """Load PRESTO shapefile once for all tests in this class."""
        gpd = pytest.importorskip("geopandas")
        return gpd.read_file(str(PRESTO_SHP))

    def test_presto_shapefile_loading(self, presto_gdf):
        """Shapefile should load with expected columns and row count."""
        # Must have rows
        assert len(presto_gdf) > 100, f"Expected >100 rows, got {len(presto_gdf)}"
        # Key columns
        assert "PC1" in presto_gdf.columns, "Missing PC1 column"
        assert "AMBEL" in presto_gdf.columns, "Missing AMBEL column"
        assert "geometry" in presto_gdf.columns, "Missing geometry column"
        # CRS should be UTM zone 19S
        assert presto_gdf.crs is not None
        assert presto_gdf.crs.to_epsg() == 32719

    def test_presto_numeric_columns(self, presto_gdf):
        """PC1 through PC5 and species columns should be numeric."""
        expected_numeric = ["PC1", "PC2", "PC3", "PC4", "PC5", "AMBEL", "LENCU"]
        for col in expected_numeric:
            if col in presto_gdf.columns:
                assert np.issubdtype(
                    presto_gdf[col].dtype, np.number
                ), f"{col} is not numeric: {presto_gdf[col].dtype}"

    @pytest.mark.slow
    def test_bivariate_lisa_real_data(self, presto_gdf):
        """Run bivariate LISA on PC1 x AMBEL and verify cluster distribution."""
        from ragweed_toolkit.spatial import compute_bivariate_lisa, lisa_summary

        # Project to UTM if geographic
        if presto_gdf.crs and presto_gdf.crs.is_geographic:
            gdf_proj = presto_gdf.to_crs("EPSG:32719")
        else:
            gdf_proj = presto_gdf

        gdf_lisa = compute_bivariate_lisa(
            gdf_proj,
            var_x="PC1",
            var_y="AMBEL",
            threshold=15.0,
            permutations=999,
        )

        # Must have lisa_cluster column
        assert "lisa_cluster" in gdf_lisa.columns

        summary = lisa_summary(gdf_lisa)

        # All five categories present
        for cat in ["HH", "HL", "LH", "LL", "NS"]:
            assert cat in summary, f"Missing category: {cat}"
            assert "count" in summary[cat]
            assert "pct" in summary[cat]

        # Significant fraction should be substantial (expected ~68%)
        sig_pct = 100.0 - summary["NS"]["pct"]
        assert sig_pct > 40, f"Significant fraction too low: {sig_pct:.1f}%"

        # HH and LL should be the dominant significant clusters
        hh_pct = summary["HH"]["pct"]
        ll_pct = summary["LL"]["pct"]
        assert hh_pct > 10, f"HH cluster too small: {hh_pct}%"
        assert ll_pct > 10, f"LL cluster too small: {ll_pct}%"


# ---------------------------------------------------------------------------
# Spectral indices tests (always run -- synthetic data)
# ---------------------------------------------------------------------------


class TestSpectralIndices:
    """Verify spectral index computation with synthetic Sentinel-2 bands."""

    EXPECTED_INDICES = {
        "NDVI", "EVI", "GNDVI", "RENDVI", "S2WI", "NBR2", "BSI", "Clay", "SWIRd"
    }

    @pytest.fixture
    def vegetation_bands(self):
        """Create synthetic Sentinel-2 bands with vegetation-like signal.

        Band order: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12.
        """
        rng = np.random.RandomState(42)
        bands = rng.uniform(0.01, 0.5, size=(10, 100, 100)).astype(np.float32)
        # Vegetation: high NIR (B8), low Red (B4)
        bands[6] = 0.35  # B8 (NIR)
        bands[2] = 0.05  # B4 (Red)
        return bands

    def test_spectral_indices_synthetic(self, vegetation_bands):
        """All 9 spectral indices should be computed from synthetic bands."""
        from ragweed_toolkit.satellite.indices import compute_spectral_indices

        indices = compute_spectral_indices(vegetation_bands)

        assert set(indices.keys()) == self.EXPECTED_INDICES, (
            f"Mismatch: missing={self.EXPECTED_INDICES - set(indices.keys())}, "
            f"extra={set(indices.keys()) - self.EXPECTED_INDICES}"
        )

        for name, arr in indices.items():
            assert arr.shape == (100, 100), f"{name} shape mismatch: {arr.shape}"
            assert np.isfinite(arr).any(), f"{name} has no finite values"

    def test_spectral_indices_vegetation_signal(self, vegetation_bands):
        """NDVI should be positive for vegetation-like spectral profile."""
        from ragweed_toolkit.satellite.indices import compute_spectral_indices

        indices = compute_spectral_indices(vegetation_bands)
        ndvi = indices["NDVI"]

        assert ndvi.mean() > 0, (
            f"NDVI mean should be positive for vegetation, got {ndvi.mean():.4f}"
        )
        # For our synthetic data (NIR=0.35, Red=0.05): NDVI = (0.35-0.05)/(0.35+0.05) = 0.75
        assert ndvi.mean() > 0.5, (
            f"NDVI mean should be >0.5 for strong vegetation signal, got {ndvi.mean():.4f}"
        )

    def test_spectral_indices_bare_soil(self):
        """NDVI should be near zero for bare-soil-like bands."""
        from ragweed_toolkit.satellite.indices import compute_spectral_indices

        rng = np.random.RandomState(99)
        bands = rng.uniform(0.1, 0.3, size=(10, 50, 50)).astype(np.float32)
        # Bare soil: similar Red and NIR reflectance
        bands[2] = 0.20  # B4 (Red)
        bands[6] = 0.22  # B8 (NIR)

        indices = compute_spectral_indices(bands)
        ndvi = indices["NDVI"]
        # NDVI = (0.22-0.20)/(0.22+0.20) ~ 0.048
        assert abs(ndvi.mean()) < 0.3, (
            f"NDVI should be near zero for bare soil, got {ndvi.mean():.4f}"
        )
