"""
Tests for ragweed_toolkit.embeddings module.

Covers FeatureExtractor output shape, MMD computation, permutation test,
deployment gate thresholds, and dimensionality reduction output shape.
"""

import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from ragweed_toolkit.embeddings.extractor import FeatureExtractor
    HAS_EXTRACTOR = True
except ImportError:
    HAS_EXTRACTOR = False

try:
    from ragweed_toolkit.embeddings.mmd import (
        compute_mmd,
        deployment_gate,
        permutation_test,
    )
    HAS_MMD = True
except ImportError:
    HAS_MMD = False

try:
    from ragweed_toolkit.embeddings.reduction import reduce_embeddings
    HAS_REDUCTION = True
except ImportError:
    HAS_REDUCTION = False


# ------------------------------------------------------------------ #
# FeatureExtractor
# ------------------------------------------------------------------ #

@pytest.mark.skipif(not HAS_TORCH or not HAS_EXTRACTOR, reason="torch/torchvision not installed")
class TestFeatureExtractor:
    """Test FeatureExtractor output shape with dummy tensor."""

    def test_resnet50_output_shape(self):
        extractor = FeatureExtractor("resnet50")
        extractor.eval()
        dummy = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = extractor(dummy)
        assert out.shape == (4, 2048)

    def test_resnet50_embedding_dim(self):
        extractor = FeatureExtractor("resnet50")
        assert extractor.embedding_dim == 2048

    def test_single_image(self):
        extractor = FeatureExtractor("resnet50")
        extractor.eval()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = extractor(dummy)
        assert out.shape == (1, 2048)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            FeatureExtractor("nonexistent_model")


# ------------------------------------------------------------------ #
# MMD
# ------------------------------------------------------------------ #

@pytest.mark.skipif(not HAS_MMD, reason="sklearn not installed")
class TestMMD:
    """Test compute_mmd returns positive float."""

    def test_identical_distributions_low_mmd(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 10)
        Y = rng.randn(50, 10)
        mmd = compute_mmd(X, Y)
        assert isinstance(mmd, float)
        assert mmd >= 0.0

    def test_different_distributions_higher_mmd(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 10)
        Y = rng.randn(50, 10) + 5.0  # shifted distribution
        mmd = compute_mmd(X, Y)
        assert mmd > 0.0

    def test_same_data_zero_or_near_zero(self):
        rng = np.random.RandomState(42)
        X = rng.randn(30, 10)
        mmd = compute_mmd(X, X.copy())
        # Unbiased estimator with same data should be near zero
        assert mmd < 0.1

    def test_returns_float(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 5)
        Y = rng.randn(20, 5) + 1.0
        mmd = compute_mmd(X, Y)
        assert isinstance(mmd, float)


# ------------------------------------------------------------------ #
# Permutation test
# ------------------------------------------------------------------ #

@pytest.mark.skipif(not HAS_MMD, reason="sklearn not installed")
class TestPermutationTest:
    """Test permutation_test returns p-value in [0, 1]."""

    def test_pvalue_range(self):
        rng = np.random.RandomState(42)
        X = rng.randn(30, 5)
        Y = rng.randn(30, 5) + 2.0
        mmd_val, p_value = permutation_test(X, Y, n_permutations=50, random_state=42)
        assert 0.0 <= p_value <= 1.0
        assert isinstance(mmd_val, float)
        assert mmd_val >= 0.0

    def test_identical_data_high_pvalue(self):
        rng = np.random.RandomState(42)
        X = rng.randn(30, 5)
        _, p_value = permutation_test(X, X.copy(), n_permutations=50, random_state=42)
        # Same data -> MMD ~0 -> most permutations should be >= observed
        assert p_value > 0.1

    def test_shifted_data_low_pvalue(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        Y = rng.randn(50, 5) + 10.0
        _, p_value = permutation_test(X, Y, n_permutations=99, random_state=42)
        # Very different distributions -> low p-value
        assert p_value < 0.1


# ------------------------------------------------------------------ #
# Deployment gate
# ------------------------------------------------------------------ #

@pytest.mark.skipif(not HAS_MMD, reason="mmd module not available")
class TestDeploymentGate:
    """Test deployment_gate threshold mapping."""

    def test_green(self):
        result = deployment_gate(0.10)
        assert result.tier == "green"

    def test_yellow(self):
        result = deployment_gate(0.20)
        assert result.tier == "yellow"

    def test_orange(self):
        result = deployment_gate(0.35)
        assert result.tier == "orange"

    def test_red(self):
        result = deployment_gate(0.50)
        assert result.tier == "red"

    def test_boundary_green_yellow(self):
        # At exactly 0.15 -> yellow (>= 0.15)
        result = deployment_gate(0.15)
        assert result.tier == "yellow"

    def test_boundary_yellow_orange(self):
        result = deployment_gate(0.30)
        assert result.tier == "orange"

    def test_boundary_orange_red(self):
        result = deployment_gate(0.45)
        assert result.tier == "red"

    def test_zero_mmd(self):
        result = deployment_gate(0.0)
        assert result.tier == "green"

    def test_very_high_mmd(self):
        result = deployment_gate(100.0)
        assert result.tier == "red"

    def test_result_has_mmd_field(self):
        result = deployment_gate(0.25)
        assert result.mmd == 0.25

    def test_result_has_recommendation(self):
        result = deployment_gate(0.10)
        assert isinstance(result.recommendation, str)
        assert len(result.recommendation) > 0


# ------------------------------------------------------------------ #
# Dimensionality reduction
# ------------------------------------------------------------------ #

@pytest.mark.skipif(not HAS_REDUCTION, reason="sklearn not installed")
class TestReduceEmbeddings:
    """Test reduce_embeddings output shape."""

    def test_pca_output_shape(self):
        rng = np.random.RandomState(42)
        emb = rng.randn(50, 100)
        df = reduce_embeddings(emb, method="pca", n_components=2)
        assert len(df) == 50
        assert "pca_x" in df.columns
        assert "pca_y" in df.columns

    def test_pca_3d(self):
        rng = np.random.RandomState(42)
        emb = rng.randn(50, 100)
        df = reduce_embeddings(emb, method="pca", n_components=3)
        assert "pca_z" in df.columns

    def test_metadata_included(self):
        rng = np.random.RandomState(42)
        emb = rng.randn(20, 50)
        labels = ["A"] * 10 + ["B"] * 10
        df = reduce_embeddings(emb, method="pca", metadata={"group": labels})
        assert "group" in df.columns
        assert list(df["group"].unique()) == ["A", "B"] or set(df["group"]) == {"A", "B"}

    def test_invalid_method_raises(self):
        rng = np.random.RandomState(42)
        emb = rng.randn(20, 50)
        with pytest.raises(ValueError, match="Unknown method"):
            reduce_embeddings(emb, method="invalid_method")
