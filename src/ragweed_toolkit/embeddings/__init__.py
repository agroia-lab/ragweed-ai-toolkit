"""
Embeddings Module
=================

ResNet-50 feature extraction for cross-domain analysis. Includes Maximum Mean
Discrepancy (MMD) computation with deployment gate framework, and PCA/UMAP/t-SNE
dimensionality reduction pipelines.

Key components:
    - extractor: ResNet-50 backbone feature extraction (2048-D)
    - mmd: Maximum Mean Discrepancy with RBF kernel and permutation testing
    - reduction: PCA -> UMAP/t-SNE dimensionality reduction pipeline
    - viz: Embedding scatter plots and MMD heatmaps
"""

from ragweed_toolkit.embeddings.extractor import (
    FeatureExtractor,
    default_transform,
    extract_embeddings,
    scan_image_directory,
)
from ragweed_toolkit.embeddings.mmd import (
    DeploymentResult,
    compute_mmd,
    deployment_gate,
    permutation_test,
)
from ragweed_toolkit.embeddings.reduction import reduce_embeddings
from ragweed_toolkit.embeddings.viz import mmd_bar_chart, mmd_heatmap, umap_scatter

__all__ = [
    # extractor
    "FeatureExtractor",
    "default_transform",
    "extract_embeddings",
    "scan_image_directory",
    # mmd
    "compute_mmd",
    "permutation_test",
    "deployment_gate",
    "DeploymentResult",
    # reduction
    "reduce_embeddings",
    # viz
    "umap_scatter",
    "mmd_heatmap",
    "mmd_bar_chart",
]
