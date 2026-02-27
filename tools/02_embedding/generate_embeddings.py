#!/usr/bin/env python3
"""
Image Embedding Generation and Visualization Pipeline
======================================================

Generates embeddings from SUGAL and INIA images using pre-trained models,
then applies dimensionality reduction (UMAP, t-SNE, PCA) for visualization.

Usage:
    python generate_embeddings.py

Output:
    - embeddings.npz: Raw embeddings
    - embeddings_reduced.csv: Reduced coordinates with metadata
    - visualization_umap_2d.html: Interactive UMAP plot
    - visualization_tsne_2d.html: Interactive t-SNE plot
    - visualization_pca_2d.html: Interactive PCA plot
    - visualization_umap_3d.html: Interactive 3D UMAP plot
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ragweed_toolkit library imports
from ragweed_toolkit.embeddings import (
    FeatureExtractor,
    default_transform,
    extract_embeddings,
    reduce_embeddings,
    umap_scatter,
)

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data paths
    "data_dir": "/media/malezainia1/LORENZO/data_sugal_cp_25-26",
    "output_dir": "/media/malezainia1/LORENZO/outputs_sugal_25-26/image_embeddings",

    # Model configuration
    "model_name": "resnet50",  # Options: resnet50, efficientnet_b0, efficientnet_b2
    "embedding_dim": 2048,  # ResNet50 output dimension

    # Processing configuration
    "batch_size": 32,
    "num_workers": 4,
    "image_size": 224,
    "max_images": None,  # Set to int to limit (for testing)

    # Dimensionality reduction
    "umap_n_neighbors": 15,
    "umap_min_dist": 0.1,
    "tsne_perplexity": 30,
    "pca_components": 50,  # Intermediate PCA before UMAP/t-SNE

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# FeatureExtractor is now imported from ragweed_toolkit.embeddings


# ============================================================================
# IMAGE DATASET
# ============================================================================

class ImageDataset(Dataset):
    """Dataset for loading images from directory structure."""

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        max_images: Optional[int] = None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Find all images
        extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        self.image_paths = []

        print(f"Scanning for images in {root_dir}...")
        for ext in extensions:
            self.image_paths.extend(list(self.root_dir.rglob(f"*{ext}")))

        # Sort for reproducibility
        self.image_paths.sort()

        # Limit if specified
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

        print(f"Found {len(self.image_paths)} images")

        # Extract metadata from paths
        self.metadata = self._extract_metadata()

    def _extract_metadata(self) -> List[Dict]:
        """Extract location/source info from path structure."""
        metadata = []
        for path in self.image_paths:
            rel_path = path.relative_to(self.root_dir)
            parts = rel_path.parts

            # Extract location (first directory level)
            location = parts[0] if len(parts) > 0 else "unknown"

            # Extract source type (Celular, Dron, etc.)
            source_type = "unknown"
            for part in parts:
                if "Celular" in part:
                    source_type = "Celular"
                    break
                elif "Dron" in part or "dron" in part:
                    source_type = "Dron"
                    break
                elif "MEDIA" in part:
                    source_type = "Camera"
                    break

            metadata.append({
                "filename": path.name,
                "full_path": str(path),
                "relative_path": str(rel_path),
                "location": location,
                "source_type": source_type,
            })

        return metadata

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, idx

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image on error
            if self.transform:
                return torch.zeros(3, 224, 224), idx
            return Image.new('RGB', (224, 224)), idx


# extract_embeddings is now imported from ragweed_toolkit.embeddings


# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================

def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    **kwargs
) -> np.ndarray:
    """
    Apply dimensionality reduction to embeddings.

    Delegates to ragweed_toolkit.embeddings.reduce_embeddings() for the core
    computation. Returns raw numpy coordinates (not the full DataFrame).

    Args:
        embeddings: Input embeddings (n_samples, n_features)
        method: "umap", "tsne", or "pca"
        n_components: Output dimensions (2 or 3)
        **kwargs: Additional parameters (n_neighbors, min_dist, perplexity)

    Returns:
        reduced: Reduced coordinates (n_samples, n_components)
    """
    df = reduce_embeddings(
        embeddings,
        method=method,
        n_components=n_components,
        umap_n_neighbors=kwargs.get("n_neighbors", 15),
        umap_min_dist=kwargs.get("min_dist", 0.1),
        tsne_perplexity=kwargs.get("perplexity", 30),
    )
    # Extract raw coordinates from the DataFrame
    prefix = method.lower()
    cols = [f"{prefix}_x", f"{prefix}_y"]
    if n_components >= 3:
        cols.append(f"{prefix}_z")
    return df[cols].values


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualization(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: Optional[str] = None,
    color_col: str = "location",
    title: str = "Embedding Visualization",
    output_path: str = "visualization.html"
):
    """
    Create interactive scatter plot visualization.

    Delegates to ragweed_toolkit.embeddings.umap_scatter() for the core plot.

    Args:
        df: DataFrame with coordinates and metadata
        x_col, y_col, z_col: Column names for coordinates
        color_col: Column to use for coloring points
        title: Plot title
        output_path: Output HTML file path
    """
    return umap_scatter(
        df,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        color_col=color_col,
        title=title,
        output_path=output_path,
    )


def create_combined_visualization(
    df: pd.DataFrame,
    output_path: str = "visualization_combined.html"
):
    """
    Create a combined visualization with all reduction methods.
    """
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("UMAP 2D", "t-SNE 2D", "PCA 2D", "UMAP 3D"),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter3d"}]
        ]
    )

    # Get unique locations for color mapping
    locations = df["location"].unique()
    colors = px.colors.qualitative.Set3[:len(locations)]
    color_map = dict(zip(locations, colors))

    # UMAP 2D
    for loc in locations:
        mask = df["location"] == loc
        fig.add_trace(
            go.Scatter(
                x=df.loc[mask, "umap_x"],
                y=df.loc[mask, "umap_y"],
                mode="markers",
                name=loc,
                marker=dict(color=color_map[loc], size=4, opacity=0.7),
                text=df.loc[mask, "filename"],
                hovertemplate="%{text}<br>Location: " + loc,
                showlegend=True
            ),
            row=1, col=1
        )

    # t-SNE 2D
    for loc in locations:
        mask = df["location"] == loc
        fig.add_trace(
            go.Scatter(
                x=df.loc[mask, "tsne_x"],
                y=df.loc[mask, "tsne_y"],
                mode="markers",
                name=loc,
                marker=dict(color=color_map[loc], size=4, opacity=0.7),
                text=df.loc[mask, "filename"],
                hovertemplate="%{text}<br>Location: " + loc,
                showlegend=False
            ),
            row=1, col=2
        )

    # PCA 2D
    for loc in locations:
        mask = df["location"] == loc
        fig.add_trace(
            go.Scatter(
                x=df.loc[mask, "pca_x"],
                y=df.loc[mask, "pca_y"],
                mode="markers",
                name=loc,
                marker=dict(color=color_map[loc], size=4, opacity=0.7),
                text=df.loc[mask, "filename"],
                hovertemplate="%{text}<br>Location: " + loc,
                showlegend=False
            ),
            row=2, col=1
        )

    # UMAP 3D
    for loc in locations:
        mask = df["location"] == loc
        fig.add_trace(
            go.Scatter3d(
                x=df.loc[mask, "umap_3d_x"],
                y=df.loc[mask, "umap_3d_y"],
                z=df.loc[mask, "umap_3d_z"],
                mode="markers",
                name=loc,
                marker=dict(color=color_map[loc], size=2, opacity=0.7),
                text=df.loc[mask, "filename"],
                hovertemplate="%{text}<br>Location: " + loc,
                showlegend=False
            ),
            row=2, col=2
        )

    fig.update_layout(
        height=1000,
        width=1400,
        title_text="Image Embeddings - Multiple Dimensionality Reduction Methods",
        template="plotly_white"
    )

    fig.write_html(output_path)
    print(f"Saved combined visualization: {output_path}")

    return fig


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("IMAGE EMBEDDING GENERATION AND VISUALIZATION PIPELINE")
    print("=" * 70)
    print()

    # Create output directory
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    print()

    # -------------------------------------------------------------------------
    # STEP 1: Load images and create dataset
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("STEP 1: Loading images")
    print("-" * 70)

    transform = default_transform(CONFIG["image_size"])

    dataset = ImageDataset(
        root_dir=CONFIG["data_dir"],
        transform=transform,
        max_images=CONFIG["max_images"]
    )

    if len(dataset) == 0:
        print("ERROR: No images found!")
        sys.exit(1)

    print()

    # -------------------------------------------------------------------------
    # STEP 2: Extract embeddings
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("STEP 2: Extracting embeddings")
    print("-" * 70)

    # Check for cached embeddings
    embeddings_path = output_dir / "embeddings.npz"

    if embeddings_path.exists():
        print(f"Loading cached embeddings from {embeddings_path}")
        data = np.load(embeddings_path, allow_pickle=True)
        embeddings = data["embeddings"]
        metadata = data["metadata"].tolist()
        print(f"Loaded {len(embeddings)} embeddings")
    else:
        # Initialize model (from ragweed_toolkit.embeddings)
        print(f"Initializing {CONFIG['model_name']} model...")
        model = FeatureExtractor(CONFIG["model_name"])
        print(f"Model embedding dimension: {model.embedding_dim}")

        # Extract embeddings (from ragweed_toolkit.embeddings)
        embeddings = extract_embeddings(
            images=dataset,
            model=model,
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
            device=CONFIG["device"],
        )

        metadata = dataset.metadata

        # Save embeddings
        np.savez_compressed(
            embeddings_path,
            embeddings=embeddings,
            metadata=np.array(metadata)
        )
        print(f"Saved embeddings to: {embeddings_path}")

    print(f"Embeddings shape: {embeddings.shape}")
    print()

    # -------------------------------------------------------------------------
    # STEP 3: Dimensionality reduction
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("STEP 3: Applying dimensionality reduction")
    print("-" * 70)

    # UMAP 2D
    print("\n[1/4] UMAP 2D...")
    umap_2d = reduce_dimensions(
        embeddings,
        method="umap",
        n_components=2,
        n_neighbors=CONFIG["umap_n_neighbors"],
        min_dist=CONFIG["umap_min_dist"]
    )

    # UMAP 3D
    print("\n[2/4] UMAP 3D...")
    umap_3d = reduce_dimensions(
        embeddings,
        method="umap",
        n_components=3,
        n_neighbors=CONFIG["umap_n_neighbors"],
        min_dist=CONFIG["umap_min_dist"]
    )

    # t-SNE 2D
    print("\n[3/4] t-SNE 2D...")
    tsne_2d = reduce_dimensions(
        embeddings,
        method="tsne",
        n_components=2,
        perplexity=CONFIG["tsne_perplexity"]
    )

    # PCA 2D
    print("\n[4/4] PCA 2D...")
    pca_2d = reduce_dimensions(embeddings, method="pca", n_components=2)

    print()

    # -------------------------------------------------------------------------
    # STEP 4: Create DataFrame with all data
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("STEP 4: Creating combined dataset")
    print("-" * 70)

    df = pd.DataFrame(metadata)

    # Add reduced coordinates
    df["umap_x"] = umap_2d[:, 0]
    df["umap_y"] = umap_2d[:, 1]
    df["umap_3d_x"] = umap_3d[:, 0]
    df["umap_3d_y"] = umap_3d[:, 1]
    df["umap_3d_z"] = umap_3d[:, 2]
    df["tsne_x"] = tsne_2d[:, 0]
    df["tsne_y"] = tsne_2d[:, 1]
    df["pca_x"] = pca_2d[:, 0]
    df["pca_y"] = pca_2d[:, 1]

    # Save CSV
    csv_path = output_dir / "embeddings_reduced.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved reduced embeddings to: {csv_path}")

    # Print statistics
    print("\nDataset statistics:")
    print(f"  Total images: {len(df)}")
    print("\n  By location:")
    for loc, count in df["location"].value_counts().items():
        print(f"    {loc}: {count}")
    print("\n  By source type:")
    for src, count in df["source_type"].value_counts().items():
        print(f"    {src}: {count}")

    print()

    # -------------------------------------------------------------------------
    # STEP 5: Create visualizations
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("STEP 5: Creating visualizations")
    print("-" * 70)

    # Individual visualizations
    create_visualization(
        df,
        x_col="umap_x",
        y_col="umap_y",
        color_col="location",
        title="UMAP 2D - Image Embeddings by Location",
        output_path=str(output_dir / "visualization_umap_2d.html")
    )

    create_visualization(
        df,
        x_col="tsne_x",
        y_col="tsne_y",
        color_col="location",
        title="t-SNE 2D - Image Embeddings by Location",
        output_path=str(output_dir / "visualization_tsne_2d.html")
    )

    create_visualization(
        df,
        x_col="pca_x",
        y_col="pca_y",
        color_col="location",
        title="PCA 2D - Image Embeddings by Location",
        output_path=str(output_dir / "visualization_pca_2d.html")
    )

    create_visualization(
        df,
        x_col="umap_3d_x",
        y_col="umap_3d_y",
        z_col="umap_3d_z",
        color_col="location",
        title="UMAP 3D - Image Embeddings by Location",
        output_path=str(output_dir / "visualization_umap_3d.html")
    )

    # Source type visualizations
    create_visualization(
        df,
        x_col="umap_x",
        y_col="umap_y",
        color_col="source_type",
        title="UMAP 2D - Image Embeddings by Source Type",
        output_path=str(output_dir / "visualization_umap_2d_by_source.html")
    )

    # Combined visualization
    create_combined_visualization(
        df,
        output_path=str(output_dir / "visualization_combined.html")
    )

    print()
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("Generated files:")
    for f in sorted(output_dir.glob("*")):
        size = f.stat().st_size / 1024
        unit = "KB"
        if size > 1024:
            size /= 1024
            unit = "MB"
        print(f"  {f.name}: {size:.1f} {unit}")

    return df


if __name__ == "__main__":
    main()
