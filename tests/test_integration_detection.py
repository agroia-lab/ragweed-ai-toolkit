"""
Integration tests for detection and embedding pipelines with real data.

These tests require:
- External drive mounted at /media/malezainia2/E/
- YOLO model weights at training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/
- CUDA GPU available

Skip conditions:
- @pytest.mark.skipif when external drive is not mounted
- @pytest.mark.skipif when CUDA is not available
"""

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------
EXTERNAL_DRIVE = Path("/media/malezainia2/E/ProcessingData")
EMBEDDINGS_DIR = Path(
    "/media/malezainia2/E/rageweed_international_databases/ragweed_embeddings"
)
MODEL_PATH = Path(
    "/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/"
    "training_results/malezainia2/ambel_seba/"
    "run3_lentejas640_b64_2gpu/weights/best.pt"
)
CL_IMAGE_DIR = EXTERNAL_DRIVE / "ambrosia-lentejas-seba-v1" / "test" / "images"
INTL_IMAGE_DIR = EXTERNAL_DRIVE / "ambel-international-v1" / "images"

_drive_available = EXTERNAL_DRIVE.exists()
_model_available = MODEL_PATH.exists()

try:
    import torch

    _cuda_available = torch.cuda.is_available()
    _cuda_device = "cuda:0"
except ImportError:
    _cuda_available = False
    _cuda_device = "cpu"

skip_no_drive = pytest.mark.skipif(
    not _drive_available,
    reason="External drive not mounted at /media/malezainia2/E/ProcessingData",
)
skip_no_model = pytest.mark.skipif(
    not _model_available,
    reason=f"YOLO model not found at {MODEL_PATH}",
)
skip_no_cuda = pytest.mark.skipif(
    not _cuda_available,
    reason="CUDA not available",
)


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------
class TestSahiDetection:
    """Integration tests for SAHI sliced inference pipeline."""

    @skip_no_drive
    @skip_no_model
    @skip_no_cuda
    def test_sahi_batch_inference(self):
        """Run SAHI inference on 5 CL_Seba test images and verify output."""
        from ragweed_toolkit.detection import SahiConfig, run_sahi_batch

        image_paths = sorted(CL_IMAGE_DIR.glob("*.jpg"))[:5]
        assert len(image_paths) == 5, f"Expected 5 images, found {len(image_paths)}"

        config = SahiConfig(
            model_path=str(MODEL_PATH),
            slice_size=640,
            overlap_ratio=0.2,
            confidence_threshold=0.25,
            device=_cuda_device,
        )
        results = run_sahi_batch(image_paths, config, verbose=False)

        assert len(results) == 5
        total_dets = sum(len(dets) for dets in results.values())
        assert total_dets > 0, "Expected at least 1 detection across 5 images"

    @skip_no_drive
    @skip_no_model
    @skip_no_cuda
    def test_detection_structure(self):
        """Verify individual detection dictionaries have correct keys/types."""
        from ragweed_toolkit.detection import SahiConfig, run_sahi

        image_paths = sorted(CL_IMAGE_DIR.glob("*.jpg"))[:1]
        config = SahiConfig(
            model_path=str(MODEL_PATH),
            slice_size=640,
            confidence_threshold=0.25,
            device=_cuda_device,
        )
        dets = run_sahi(image_paths[0], config)

        assert isinstance(dets, list)
        if dets:
            d = dets[0]
            assert set(d.keys()) == {"class_id", "class_name", "bbox", "confidence"}
            assert isinstance(d["class_id"], int)
            assert isinstance(d["class_name"], str)
            assert len(d["bbox"]) == 4
            assert all(isinstance(v, float) for v in d["bbox"])
            assert 0.0 <= d["confidence"] <= 1.0


class TestGpsExtraction:
    """Integration tests for GPS EXIF extraction."""

    @skip_no_drive
    def test_extract_gps_geodataframe(self):
        """Extract GPS from test images and verify GeoDataFrame structure."""
        from ragweed_toolkit.detection import extract_gps_geodataframe

        image_paths = sorted(CL_IMAGE_DIR.glob("*.jpg"))[:5]
        gdf = extract_gps_geodataframe([str(p) for p in image_paths])

        assert len(gdf) == 5
        expected_cols = {"filename", "filepath", "latitude", "longitude",
                         "altitude", "has_gps", "geometry"}
        assert expected_cols.issubset(set(gdf.columns))
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326

    @skip_no_drive
    def test_gps_export_geopackage(self, tmp_path):
        """Verify GeoDataFrame can be exported to GeoPackage."""
        from ragweed_toolkit.detection import extract_gps_geodataframe

        image_paths = sorted(CL_IMAGE_DIR.glob("*.jpg"))[:5]
        gdf = extract_gps_geodataframe([str(p) for p in image_paths])

        out_path = tmp_path / "test_gps.gpkg"
        gdf.to_file(str(out_path), driver="GPKG")
        assert out_path.exists()
        assert out_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------
class TestEmbeddingExtraction:
    """Integration tests for ResNet-50 feature extraction."""

    @skip_no_drive
    @skip_no_cuda
    def test_extract_embeddings_shape(self):
        """Extract embeddings from 20 images and verify shape."""
        from ragweed_toolkit.embeddings.extractor import (
            FeatureExtractor,
            extract_embeddings,
        )

        extractor = FeatureExtractor("resnet50")
        assert extractor.embedding_dim == 2048

        image_paths = sorted(CL_IMAGE_DIR.glob("*.jpg"))[:20]
        embeddings = extract_embeddings(
            image_paths, extractor, device=_cuda_device, num_workers=2,
        )
        assert embeddings.shape == (20, 2048)
        assert embeddings.dtype == np.float32

    @skip_no_drive
    @skip_no_cuda
    def test_embeddings_deterministic(self):
        """Same images produce same embeddings (no stochastic augmentation)."""
        from ragweed_toolkit.embeddings.extractor import (
            FeatureExtractor,
            extract_embeddings,
        )

        extractor = FeatureExtractor("resnet50")
        image_paths = sorted(CL_IMAGE_DIR.glob("*.jpg"))[:5]

        emb1 = extract_embeddings(
            image_paths, extractor, device=_cuda_device, num_workers=0,
        )
        emb2 = extract_embeddings(
            image_paths, extractor, device=_cuda_device, num_workers=0,
        )
        np.testing.assert_allclose(emb1, emb2, atol=1e-5)


class TestMmdComputation:
    """Integration tests for MMD domain shift analysis."""

    @skip_no_drive
    @skip_no_cuda
    def test_mmd_cross_domain(self):
        """Compute MMD between CL and INTL embeddings."""
        from ragweed_toolkit.embeddings.extractor import (
            FeatureExtractor,
            extract_embeddings,
        )
        from ragweed_toolkit.embeddings.mmd import compute_mmd

        extractor = FeatureExtractor("resnet50")
        cl_images = sorted(CL_IMAGE_DIR.glob("*.jpg"))[:20]
        intl_images = sorted(INTL_IMAGE_DIR.glob("*.jpg"))[:20]

        emb_cl = extract_embeddings(
            cl_images, extractor, device=_cuda_device, num_workers=2,
        )
        emb_intl = extract_embeddings(
            intl_images, extractor, device=_cuda_device, num_workers=2,
        )

        mmd_val = compute_mmd(emb_cl, emb_intl)
        assert mmd_val >= 0.0
        assert isinstance(mmd_val, float)

        # Self-similarity should be lower than cross-domain
        mmd_self = compute_mmd(emb_cl, emb_cl)
        assert mmd_self < mmd_val

    def test_deployment_gate_tiers(self):
        """Verify deployment gate maps values to correct tiers."""
        from ragweed_toolkit.embeddings.mmd import deployment_gate

        result_green = deployment_gate(0.10)
        assert result_green.tier == "green"
        assert result_green.mmd == 0.10

        result_yellow = deployment_gate(0.25)
        assert result_yellow.tier == "yellow"

        result_orange = deployment_gate(0.35)
        assert result_orange.tier == "orange"

        result_red = deployment_gate(0.50)
        assert result_red.tier == "red"

    def test_deployment_gate_output_structure(self):
        """Verify DeploymentResult has expected fields."""
        from ragweed_toolkit.embeddings.mmd import DeploymentResult, deployment_gate

        result = deployment_gate(0.20)
        assert isinstance(result, DeploymentResult)
        assert hasattr(result, "tier")
        assert hasattr(result, "recommendation")
        assert hasattr(result, "mmd")
        assert isinstance(result.recommendation, str)
        assert len(result.recommendation) > 0


class TestPrecomputedEmbeddings:
    """Integration tests for pre-computed 9-database embeddings."""

    @pytest.mark.skipif(
        not EMBEDDINGS_DIR.exists(),
        reason=f"Pre-computed embeddings not found at {EMBEDDINGS_DIR}",
    )
    def test_npz_structure(self):
        """Verify embeddings.npz has expected keys and shapes."""
        npz_path = EMBEDDINGS_DIR / "embeddings.npz"
        assert npz_path.exists()

        npz = np.load(str(npz_path), allow_pickle=True)
        assert "embeddings" in npz
        assert "metadata" in npz

        embeddings = npz["embeddings"]
        metadata = npz["metadata"]
        assert embeddings.ndim == 2
        assert embeddings.shape[1] == 2048
        assert len(metadata) == len(embeddings)

    @pytest.mark.skipif(
        not EMBEDDINGS_DIR.exists(),
        reason=f"Pre-computed embeddings not found at {EMBEDDINGS_DIR}",
    )
    def test_mmd_matrix(self):
        """Verify MMD matrix is symmetric with zero diagonal."""
        import pandas as pd

        mmd_path = EMBEDDINGS_DIR / "mmd_matrix.csv"
        if not mmd_path.exists():
            pytest.skip("mmd_matrix.csv not found")

        mmd_df = pd.read_csv(str(mmd_path), index_col=0)
        matrix = mmd_df.values

        # Symmetric
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-6)

        # Zero diagonal
        np.testing.assert_allclose(np.diag(matrix), 0.0, atol=0.01)

        # All values non-negative
        assert (matrix >= -0.01).all(), "MMD values should be non-negative"

        # At least 9 databases
        assert mmd_df.shape[0] >= 9, f"Expected >= 9 databases, got {mmd_df.shape[0]}"
