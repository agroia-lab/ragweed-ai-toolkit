"""
Tests for ragweed_toolkit.orthomosaic module.

Covers tiling math (expected tile count from image dimensions) and
score_images returning a sorted list.
"""

import math

import numpy as np
import pandas as pd
import pytest

try:
    from ragweed_toolkit.orthomosaic.active_learning import score_images
    HAS_ACTIVE_LEARNING = True
except ImportError:
    HAS_ACTIVE_LEARNING = False


# ------------------------------------------------------------------ #
# Tiling math (pure arithmetic, no rasterio needed)
# ------------------------------------------------------------------ #

def _compute_tile_grid(width, height, tile_size, overlap=0):
    """Replicate the tiling math from tiling.py without rasterio."""
    step = tile_size - overlap
    n_cols = math.ceil(width / step)
    n_rows = math.ceil(height / step)
    total = n_rows * n_cols
    return n_rows, n_cols, total, step


class TestTilingMath:
    """Test tile grid calculations match expected values."""

    def test_exact_fit_no_overlap(self):
        """Image exactly divisible by tile_size -> no extra tiles."""
        n_rows, n_cols, total, _ = _compute_tile_grid(
            width=2048, height=2048, tile_size=1024, overlap=0
        )
        assert n_rows == 2
        assert n_cols == 2
        assert total == 4

    def test_partial_tile_adds_one(self):
        """Image not divisible by tile_size -> ceiling adds an extra tile."""
        n_rows, n_cols, total, _ = _compute_tile_grid(
            width=2050, height=2050, tile_size=1024, overlap=0
        )
        assert n_rows == 3  # ceil(2050/1024) = 3
        assert n_cols == 3
        assert total == 9

    def test_with_overlap(self):
        """Overlap reduces step, so more tiles are needed."""
        n_rows, n_cols, total, step = _compute_tile_grid(
            width=2048, height=2048, tile_size=1024, overlap=200
        )
        assert step == 824
        # ceil(2048/824) = 3
        assert n_cols == 3
        assert n_rows == 3
        assert total == 9

    def test_small_image(self):
        """Image smaller than tile_size -> single tile."""
        n_rows, n_cols, total, _ = _compute_tile_grid(
            width=500, height=500, tile_size=1024, overlap=0
        )
        assert n_rows == 1
        assert n_cols == 1
        assert total == 1

    def test_single_pixel_wide(self):
        """Very narrow image."""
        n_rows, n_cols, total, _ = _compute_tile_grid(
            width=1, height=5000, tile_size=1024, overlap=0
        )
        assert n_cols == 1
        assert n_rows == 5  # ceil(5000/1024) = 5
        assert total == 5

    def test_large_orthomosaic(self):
        """Typical drone orthomosaic: ~20k x 20k pixels."""
        n_rows, n_cols, total, _ = _compute_tile_grid(
            width=20000, height=20000, tile_size=1024, overlap=0
        )
        expected_per_axis = math.ceil(20000 / 1024)  # 20
        assert n_rows == expected_per_axis
        assert n_cols == expected_per_axis
        assert total == expected_per_axis ** 2

    def test_step_equals_tile_minus_overlap(self):
        _, _, _, step = _compute_tile_grid(
            width=4096, height=4096, tile_size=1024, overlap=128
        )
        assert step == 1024 - 128


# ------------------------------------------------------------------ #
# score_images — active learning
# ------------------------------------------------------------------ #

@pytest.mark.skipif(not HAS_ACTIVE_LEARNING, reason="active_learning deps not available")
class TestScoreImages:
    """Test score_images returns sorted DataFrame."""

    @pytest.fixture
    def mock_images_data(self):
        """Create synthetic images_data dict with varying uncertainty."""
        return {
            "img_confident.jpg": {
                "detections": [
                    {"confidence": 0.95, "class_name": "ragweed", "bbox": [10, 10, 50, 50]},
                    {"confidence": 0.92, "class_name": "ragweed", "bbox": [60, 60, 100, 100]},
                ],
                "width": 640,
                "height": 640,
            },
            "img_uncertain.jpg": {
                "detections": [
                    {"confidence": 0.30, "class_name": "ragweed", "bbox": [10, 10, 50, 50]},
                    {"confidence": 0.28, "class_name": "ragweed", "bbox": [60, 60, 100, 100]},
                    {"confidence": 0.35, "class_name": "ragweed", "bbox": [110, 110, 150, 150]},
                ],
                "width": 640,
                "height": 640,
            },
            "img_empty.jpg": {
                "detections": [],
                "width": 640,
                "height": 640,
            },
        }

    def test_returns_dataframe(self, mock_images_data):
        df = score_images(mock_images_data)
        assert isinstance(df, pd.DataFrame)

    def test_all_images_present(self, mock_images_data):
        df = score_images(mock_images_data)
        assert len(df) == 3

    def test_sorted_by_composite_score_descending(self, mock_images_data):
        df = score_images(mock_images_data)
        scores = df["composite_score"].values
        assert np.all(scores[:-1] >= scores[1:])

    def test_has_rank_column(self, mock_images_data):
        df = score_images(mock_images_data)
        assert "rank" in df.columns
        assert df["rank"].iloc[0] == 1

    def test_has_required_columns(self, mock_images_data):
        df = score_images(mock_images_data)
        required = {"image", "n_detections", "composite_score", "rank"}
        assert required.issubset(set(df.columns))

    def test_uncertain_image_ranked_higher(self, mock_images_data):
        """Image with low-confidence detections should rank higher."""
        df = score_images(mock_images_data)
        top_image = df.iloc[0]["image"]
        # The uncertain image should be ranked first
        assert top_image == "img_uncertain.jpg"

    def test_composite_score_nonnegative(self, mock_images_data):
        df = score_images(mock_images_data)
        assert (df["composite_score"] >= 0).all()

    def test_n_detections_correct(self, mock_images_data):
        df = score_images(mock_images_data)
        for _, row in df.iterrows():
            expected = len(mock_images_data[row["image"]]["detections"])
            assert row["n_detections"] == expected
