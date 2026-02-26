"""
Dimensionality Reduction Pipeline
===================================

Provides a unified interface for reducing high-dimensional embeddings
(e.g. 2048-D ResNet-50 features) to 2D or 3D coordinates for visualization.

The pipeline follows the two-stage approach from Chapter 6:

    1. PCA pre-reduction: 2048 -> 50 components (preserves ~90% variance)
    2. Non-linear projection: UMAP or t-SNE from 50 -> 2D/3D

Default parameters match the chapter's experimental settings:

    - PCA: 50 components (Eq. 5)
    - UMAP: n_neighbors=15, min_dist=0.1, metric='cosine' (Eq. 6)
    - t-SNE: perplexity=30 (Eq. 7)

Usage::

    from ragweed_toolkit.embeddings import reduce_embeddings

    # Full pipeline: PCA -> UMAP
    df = reduce_embeddings(embeddings, method="umap", metadata={"database": labels})

    # Direct PCA only
    df = reduce_embeddings(embeddings, method="pca")
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Default parameters matching Chapter 6
PCA_COMPONENTS = 50
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
TSNE_PERPLEXITY = 30


def _pca_prereduction(
    embeddings: np.ndarray,
    n_components: int = PCA_COMPONENTS,
    random_state: int = 42,
) -> np.ndarray:
    """Apply PCA as a pre-reduction step.

    Parameters
    ----------
    embeddings : np.ndarray
        Input embeddings, shape ``(N, D)`` where D is typically 2048.
    n_components : int
        Number of PCA components to keep (default 50).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Reduced embeddings, shape ``(N, n_components)``.
    """
    from sklearn.decomposition import PCA

    n_components = min(n_components, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {embeddings.shape[1]} -> {n_components} dims "
          f"({explained:.1%} variance explained)")
    return reduced


def reduce_embeddings(
    embeddings: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    pca_components: int = PCA_COMPONENTS,
    umap_n_neighbors: int = UMAP_N_NEIGHBORS,
    umap_min_dist: float = UMAP_MIN_DIST,
    umap_metric: str = UMAP_METRIC,
    tsne_perplexity: int = TSNE_PERPLEXITY,
    random_state: int = 42,
) -> pd.DataFrame:
    """Reduce embeddings and return a DataFrame with coordinates + metadata.

    For ``"umap"`` and ``"tsne"``, PCA pre-reduction is applied automatically
    when the input dimensionality exceeds *pca_components*.

    Parameters
    ----------
    embeddings : np.ndarray
        Input embeddings, shape ``(N, D)``.
    method : str
        Reduction method: ``"umap"``, ``"tsne"``, or ``"pca"``.
    n_components : int
        Output dimensionality (2 or 3).
    metadata : dict, optional
        Metadata columns to include in the output DataFrame.
        Keys are column names, values are arrays of length N.
    pca_components : int
        Number of PCA components for pre-reduction.
    umap_n_neighbors : int
        UMAP ``n_neighbors`` parameter.
    umap_min_dist : float
        UMAP ``min_dist`` parameter.
    umap_metric : str
        UMAP distance metric.
    tsne_perplexity : int
        t-SNE perplexity.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for coordinates (e.g. ``umap_x``, ``umap_y``)
        plus any metadata columns.

    Raises
    ------
    ValueError
        If *method* is not one of ``"umap"``, ``"tsne"``, ``"pca"``.

    Examples
    --------
    >>> import numpy as np
    >>> emb = np.random.randn(100, 2048)
    >>> labels = ["db_A"] * 50 + ["db_B"] * 50
    >>> df = reduce_embeddings(emb, "pca", metadata={"database": labels})
    >>> list(df.columns)
    ['pca_x', 'pca_y', 'database']
    """
    method = method.lower()
    if method not in ("umap", "tsne", "pca"):
        raise ValueError(
            f"Unknown method: {method!r}. Choose from: umap, tsne, pca"
        )

    working = embeddings.copy()

    # Pre-reduce with PCA for UMAP/t-SNE when input is high-dimensional
    if method in ("umap", "tsne") and working.shape[1] > pca_components:
        working = _pca_prereduction(
            working, n_components=pca_components, random_state=random_state
        )

    if method == "umap":
        import umap as umap_lib

        print(f"  UMAP: {working.shape[1]} -> {n_components}D "
              f"(n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist})")
        reducer = umap_lib.UMAP(
            n_components=n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=random_state,
            verbose=True,
        )
        coords = reducer.fit_transform(working)

    elif method == "tsne":
        from sklearn.manifold import TSNE

        print(f"  t-SNE: {working.shape[1]} -> {n_components}D "
              f"(perplexity={tsne_perplexity})")
        reducer = TSNE(
            n_components=n_components,
            perplexity=tsne_perplexity,
            random_state=random_state,
            verbose=1,
            max_iter=1000,
        )
        coords = reducer.fit_transform(working)

    elif method == "pca":
        from sklearn.decomposition import PCA

        n_comp = min(n_components, working.shape[0] - 1, working.shape[1])
        reducer = PCA(n_components=n_comp, random_state=random_state)
        coords = reducer.fit_transform(working)
        explained = reducer.explained_variance_ratio_.sum()
        print(f"  PCA: {working.shape[1]} -> {n_comp}D "
              f"({explained:.1%} variance explained)")

    # Build DataFrame
    prefix = method
    col_names = [f"{prefix}_x", f"{prefix}_y"]
    if n_components >= 3:
        col_names.append(f"{prefix}_z")

    df = pd.DataFrame(coords[:, :len(col_names)], columns=col_names)

    # Attach metadata
    if metadata:
        for col_name, values in metadata.items():
            df[col_name] = values

    return df
