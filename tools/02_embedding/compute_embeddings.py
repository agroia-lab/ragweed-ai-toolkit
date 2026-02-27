#!/usr/bin/env python3
"""
Compute CLIP Embeddings and Create UMAP Visualization

Computes CLIP embeddings for FiftyOne datasets and creates interactive
UMAP visualizations without requiring FiftyOne Enterprise.

Usage:
    # Compute embeddings for campo dataset
    python scripts/utils/compute_embeddings.py --dataset campos_completo --compute

    # Create UMAP visualization
    python scripts/utils/compute_embeddings.py --dataset campos_completo --visualize

    # Both compute and visualize
    python scripts/utils/compute_embeddings.py --dataset campos_completo --compute --visualize

    # Quick test with subset
    python scripts/utils/compute_embeddings.py --dataset campos_completo --compute --max-samples 500
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ragweed_toolkit library imports
from ragweed_toolkit.embeddings import reduce_embeddings

# TODO: Move CLIP embedding computation to ragweed_toolkit.embeddings when
# the library supports CLIP backends alongside ResNet50. Currently only
# ResNet50/EfficientNet are in FeatureExtractor.

# TODO: Move FiftyOne integration helpers to ragweed_toolkit when API
# supports dataset management (compute_clip_embeddings,
# get_embeddings_from_dataset, find_similar_images).

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Lazy imports
fo = None
F = None


def _ensure_fiftyone():
    """Lazy import FiftyOne."""
    global fo, F
    if fo is None:
        import fiftyone as _fo
        from fiftyone import ViewField as _F
        fo = _fo
        F = _F
    return fo, F


# =============================================================================
# CLIP Embedding Computation
# =============================================================================

def compute_clip_embeddings(
    dataset,
    model_name: str = "ViT-B/32",
    batch_size: int = 32,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    embedding_field: str = "clip_embedding"
):
    """Compute CLIP embeddings for all samples in dataset using OpenAI CLIP.

    Args:
        dataset: FiftyOne dataset
        model_name: CLIP model name (ViT-B/32, ViT-B/16, ViT-L/14, etc.)
        batch_size: Batch size for inference
        device: Device (cuda/cpu)
        max_samples: Limit samples for testing
        embedding_field: Field name to store embeddings

    Returns:
        numpy array of embeddings (N x embedding_dim)
    """
    import clip
    import pillow_heif
    import torch
    from PIL import Image

    # Register HEIC support
    pillow_heif.register_heif_opener()

    print(f"\nLoading CLIP model: {model_name}")

    # Check device
    if device == "cuda" and torch.cuda.is_available():
        device = "cuda"
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("  Using CPU")

    # Load model
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    # Get samples
    samples = list(dataset)
    if max_samples:
        samples = samples[:max_samples]

    print(f"\nComputing embeddings for {len(samples)} samples...")
    print(f"  Batch size: {batch_size}")

    all_embeddings = []
    sample_ids = []

    # Process in batches
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i+batch_size]
        batch_images = []
        batch_ids = []

        # Load and preprocess images
        for sample in batch_samples:
            try:
                img = Image.open(sample.filepath).convert("RGB")
                img_tensor = preprocess(img)
                batch_images.append(img_tensor)
                batch_ids.append(sample.id)
            except Exception as e:
                print(f"  Warning: Could not load {sample.filepath}: {e}")
                continue

        if not batch_images:
            continue

        # Stack into batch tensor
        batch_tensor = torch.stack(batch_images).to(device)

        # Get embeddings
        with torch.no_grad():
            embeddings = model.encode_image(batch_tensor)
            embeddings = embeddings.cpu().numpy()

            # Normalize embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        all_embeddings.append(embeddings)
        sample_ids.extend(batch_ids)

        # Progress
        processed = min(i + batch_size, len(samples))
        if processed % 100 == 0 or processed == len(samples):
            print(f"  {processed}/{len(samples)} samples processed...")

    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    print(f"\nEmbeddings shape: {all_embeddings.shape}")

    # Store embeddings in dataset
    print(f"Storing embeddings in field: {embedding_field}")
    id_to_embedding = dict(zip(sample_ids, all_embeddings))

    for sample in dataset:
        if sample.id in id_to_embedding:
            sample[embedding_field] = id_to_embedding[sample.id].tolist()
            sample.save()

    # Store metadata
    dataset.info["clip_model"] = model_name
    dataset.info["embedding_dim"] = all_embeddings.shape[1]
    dataset.info["embeddings_computed"] = datetime.now().isoformat()
    dataset.save()

    print(f"  Stored {len(id_to_embedding)} embeddings")

    return all_embeddings, sample_ids


def get_embeddings_from_dataset(dataset, embedding_field: str = "clip_embedding"):
    """Extract embeddings from dataset that were previously computed.

    Returns:
        embeddings: numpy array (N x embedding_dim)
        sample_ids: list of sample IDs
        metadata: dict with campo, source_type, etc. for each sample
    """
    embeddings = []
    sample_ids = []
    metadata = {
        'campo': [],
        'field_name': [],
        'source_type': [],
        'device_type': [],
        'filepath': [],
        'gps_latitude': [],
        'gps_longitude': [],
    }

    for sample in dataset:
        emb = sample[embedding_field]
        if emb is not None:
            embeddings.append(emb)
            sample_ids.append(sample.id)

            # Collect metadata (use "unknown" for missing tags so plots still render)
            campo = sample.campo if hasattr(sample, 'campo') else None
            field_name = sample.field_name if hasattr(sample, 'field_name') else None
            source_type = sample.source_type if hasattr(sample, 'source_type') else None
            device_type = sample.device_type if hasattr(sample, 'device_type') else None

            metadata['campo'].append(campo or "unknown")
            metadata['field_name'].append(field_name or "unknown")
            metadata['source_type'].append(source_type or "unknown")
            metadata['device_type'].append(device_type or "unknown")
            metadata['filepath'].append(sample.filepath)
            metadata['gps_latitude'].append(sample.gps_latitude if hasattr(sample, 'gps_latitude') else None)
            metadata['gps_longitude'].append(sample.gps_longitude if hasattr(sample, 'gps_longitude') else None)

    embeddings = np.array(embeddings)
    print(f"Loaded {len(embeddings)} embeddings from dataset")

    return embeddings, sample_ids, metadata


# =============================================================================
# UMAP Dimensionality Reduction
# =============================================================================

def compute_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "cosine"
) -> np.ndarray:
    """Compute UMAP dimensionality reduction.

    Delegates to ragweed_toolkit.embeddings.reduce_embeddings().

    Args:
        embeddings: High-dimensional embeddings (N x D)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        n_components: Output dimensions (2 or 3)
        metric: Distance metric

    Returns:
        Low-dimensional coordinates (N x n_components)
    """
    print("\nComputing UMAP projection...")
    print(f"  Input shape: {embeddings.shape}")
    print(f"  n_neighbors: {n_neighbors}")
    print(f"  min_dist: {min_dist}")
    print(f"  metric: {metric}")

    df = reduce_embeddings(
        embeddings,
        method="umap",
        n_components=n_components,
        umap_n_neighbors=n_neighbors,
        umap_min_dist=min_dist,
        umap_metric=metric,
    )
    cols = ["umap_x", "umap_y"]
    if n_components >= 3:
        cols.append("umap_z")
    coords = df[cols].values

    print(f"  Output shape: {coords.shape}")
    return coords


# =============================================================================
# Visualization
# =============================================================================

def create_interactive_plot(
    coords: np.ndarray,
    metadata: Dict,
    output_path: Path,
    color_by: str = "campo",
    title: str = "CLIP Embeddings - UMAP Visualization"
):
    """Create interactive Plotly scatter plot with image viewer.

    Args:
        coords: 2D UMAP coordinates (N x 2)
        metadata: Dictionary with campo, source_type, etc.
        output_path: Path to save HTML file
        color_by: Field to color points by
        title: Plot title
    """
    import pandas as pd
    import plotly.express as px

    print("\nCreating interactive visualization...")

    # Create DataFrame - include full filepath for image viewing
    df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'campo': metadata['campo'],
        'field_name': metadata['field_name'],
        'source_type': metadata['source_type'],
        'device_type': metadata['device_type'],
        'filename': [Path(p).name for p in metadata['filepath']],
        'filepath': metadata['filepath'],  # Full path for image loading
        'gps_lat': metadata['gps_latitude'],
        'gps_lon': metadata['gps_longitude'],
    })

    # Color palette
    color_map = {
        # Campos
        'campo1_ene_2026': '#e6194b',
        'campo2_ene_2026': '#3cb44b',
        'Campo3_enero26': '#ffe119',
        'campo4_enero26': '#4363d8',
        'campo5_enero26': '#f58231',
        'campo_6': '#911eb4',
        'campo_7': '#42d4f4',
        # Source types
        'drone': '#ff0000',
        'cellular': '#00ff00',
        # Device types
        'dji': '#ff6600',
        'iphone': '#0066ff',
        'android': '#00cc00',
        'iphone_jpg': '#6699ff',
        'unknown': '#999999',
    }

    # Create figure with custom data for click handling
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color=color_by,
        color_discrete_map=color_map,
        hover_data=['campo', 'field_name', 'source_type', 'device_type', 'filename', 'filepath'],
        title=title,
        labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
        custom_data=['filepath'],  # Include full path for click handler
    )

    # Update layout
    fig.update_layout(
        width=1400,
        height=900,
        template='plotly_dark',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)"
        ),
        hoverlabel=dict(
            bgcolor="black",
            font_size=12,
        )
    )

    fig.update_traces(marker=dict(size=5, opacity=0.7))

    # Save HTML with image viewer
    html_content = fig.to_html(include_plotlyjs=True, full_html=True)

    # Inject custom CSS and JavaScript for image viewer
    image_viewer_html = '''
    <style>
        #image-panel {
            position: fixed;
            right: 20px;
            top: 80px;
            width: 450px;
            background: rgba(30, 30, 40, 0.95);
            border: 2px solid #444;
            border-radius: 10px;
            padding: 15px;
            z-index: 1000;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            display: none;
        }
        #image-panel.visible { display: block; }
        #image-panel img {
            max-width: 100%;
            max-height: 400px;
            border-radius: 5px;
            display: block;
            margin: 0 auto;
        }
        #image-panel .close-btn {
            position: absolute;
            right: 10px;
            top: 10px;
            background: #ff4444;
            color: white;
            border: none;
            border-radius: 50%;
            width: 25px;
            height: 25px;
            cursor: pointer;
            font-weight: bold;
        }
        #image-panel .info {
            color: #fff;
            margin-top: 10px;
            font-family: monospace;
            font-size: 11px;
            word-break: break-all;
        }
        #image-panel .info strong { color: #88ccff; }
        #click-hint {
            position: fixed;
            left: 20px;
            bottom: 20px;
            background: rgba(50, 50, 70, 0.9);
            color: #aaa;
            padding: 8px 15px;
            border-radius: 5px;
            font-size: 12px;
            z-index: 999;
        }
    </style>
    <div id="image-panel">
        <button class="close-btn" onclick="document.getElementById('image-panel').classList.remove('visible')">×</button>
        <img id="selected-image" src="" alt="Selected Image">
        <div class="info">
            <div><strong>File:</strong> <span id="info-filename"></span></div>
            <div><strong>Campo:</strong> <span id="info-campo"></span></div>
            <div><strong>Source:</strong> <span id="info-source"></span></div>
            <div><strong>Device:</strong> <span id="info-device"></span></div>
        </div>
    </div>
    <div id="click-hint">💡 Click on any point to view the image</div>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            var plot = document.querySelector('.plotly-graph-div');
            if (plot) {
                plot.on('plotly_click', function(data) {
                    if (data.points && data.points.length > 0) {
                        var pt = data.points[0];
                        var filepath = pt.customdata[0];
                        var hoverData = pt.hovertemplate || '';

                        // Extract info from hover data
                        var campo = pt.data.customdata ? pt.data.hovertext : pt.data.name;

                        // Update image panel
                        var panel = document.getElementById('image-panel');
                        var img = document.getElementById('selected-image');

                        // Convert absolute path to HTTP URL (assumes server running in project root)
                        // Path: /home/.../data/campo1_ene_2026/... -> /data/campo1_ene_2026/...
                        var httpPath = filepath.replace('/home/malezainia1/dev/INIA_DeepLearning_Ubuntu_mod_lleon/', 'http://localhost:8888/');
                        img.src = httpPath;
                        img.onerror = function() {
                            // Fallback to file:// if HTTP fails
                            this.src = 'file://' + filepath;
                            this.onerror = function() {
                                this.src = '';
                                this.alt = 'Cannot load: ' + filepath + ' (HEIC files may not display in browser)';
                            };
                        };

                        // Update info
                        document.getElementById('info-filename').textContent = filepath.split('/').pop();
                        document.getElementById('info-campo').textContent = pt.data.name || 'N/A';
                        document.getElementById('info-source').textContent = pt.customdata ? 'See hover' : 'N/A';
                        document.getElementById('info-device').textContent = 'See hover info';

                        panel.classList.add('visible');
                    }
                });
            }
        });
    </script>
    '''

    # Insert before closing body tag
    html_content = html_content.replace('</body>', image_viewer_html + '</body>')

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"  Saved to: {output_path}")

    return fig


def create_static_plot(
    coords: np.ndarray,
    metadata: Dict,
    output_path: Path,
    color_by: str = "campo",
    title: str = "CLIP Embeddings - UMAP Visualization"
):
    """Create static matplotlib plot.

    Args:
        coords: 2D UMAP coordinates (N x 2)
        metadata: Dictionary with campo, source_type, etc.
        output_path: Path to save PNG file
        color_by: Field to color points by
        title: Plot title
    """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    print("\nCreating static visualization...")

    # Color palette
    color_map = {
        'campo1_ene_2026': '#e6194b',
        'campo2_ene_2026': '#3cb44b',
        'Campo3_enero26': '#ffe119',
        'campo4_enero26': '#4363d8',
        'campo5_enero26': '#f58231',
        'campo_6': '#911eb4',
        'campo_7': '#42d4f4',
        'drone': '#ff0000',
        'cellular': '#00ff00',
        'dji': '#ff6600',
        'iphone': '#0066ff',
        'android': '#00cc00',
        'iphone_jpg': '#6699ff',
        'unknown': '#999999',
    }

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    # Get unique values and plot each
    values = metadata[color_by]
    unique_values = list(set(v for v in values if v is not None))

    for val in unique_values:
        mask = [v == val for v in values]
        x = coords[mask, 0]
        y = coords[mask, 1]
        color = color_map.get(val, '#ffffff')
        ax.scatter(x, y, c=color, label=val, alpha=0.6, s=10)

    ax.set_xlabel('UMAP 1', color='white', fontsize=12)
    ax.set_ylabel('UMAP 2', color='white', fontsize=12)
    ax.set_title(title, color='white', fontsize=16)
    ax.tick_params(colors='white')

    legend = ax.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='white')
    for text in legend.get_texts():
        text.set_color('white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', edgecolor='none')
    plt.close()

    print(f"  Saved to: {output_path}")


def create_multi_view_visualization(
    coords: np.ndarray,
    metadata: Dict,
    output_dir: Path,
    title_prefix: str = "CLIP Embeddings"
):
    """Create multiple visualizations colored by different fields.

    Args:
        coords: 2D UMAP coordinates
        metadata: Dictionary with metadata
        output_dir: Output directory
        title_prefix: Prefix for titles
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    views = [
        ('campo', 'by Campo (Field Location)'),
        ('source_type', 'by Source (Drone vs Cellular)'),
        ('device_type', 'by Device Type'),
    ]

    print(f"\nCreating multi-view visualizations in: {output_dir}")

    for color_by, suffix in views:
        # Interactive HTML
        html_path = output_dir / f"umap_{color_by}.html"
        create_interactive_plot(
            coords, metadata, html_path,
            color_by=color_by,
            title=f"{title_prefix} - {suffix}"
        )

        # Static PNG
        png_path = output_dir / f"umap_{color_by}.png"
        create_static_plot(
            coords, metadata, png_path,
            color_by=color_by,
            title=f"{title_prefix} - {suffix}"
        )

    print("\nVisualization files created:")
    for f in output_dir.glob("umap_*"):
        print(f"  - {f.name}")


# =============================================================================
# Similarity Search
# =============================================================================

def find_similar_images(
    dataset,
    query_sample_id: str,
    embedding_field: str = "clip_embedding",
    k: int = 10
) -> List[Tuple[str, float]]:
    """Find k most similar images to a query sample.

    Args:
        dataset: FiftyOne dataset with embeddings
        query_sample_id: ID of query sample
        embedding_field: Field containing embeddings
        k: Number of similar images to return

    Returns:
        List of (sample_id, similarity_score) tuples
    """
    # Get query embedding
    query_sample = dataset[query_sample_id]
    query_embedding = np.array(query_sample[embedding_field])

    # Compute similarities
    similarities = []
    for sample in dataset:
        if sample.id == query_sample_id:
            continue
        emb = sample[embedding_field]
        if emb is not None:
            emb = np.array(emb)
            sim = np.dot(query_embedding, emb)  # Cosine similarity (embeddings are normalized)
            similarities.append((sample.id, sim, sample.filepath))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:k]


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute CLIP Embeddings and Create UMAP Visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute embeddings
  python compute_embeddings.py --dataset campos_completo --compute

  # Create visualization from existing embeddings
  python compute_embeddings.py --dataset campos_completo --visualize

  # Both compute and visualize
  python compute_embeddings.py --dataset campos_completo --compute --visualize

  # Quick test
  python compute_embeddings.py --dataset campos_completo --compute --max-samples 500 --visualize
        """
    )

    parser.add_argument('--dataset', type=str, required=True, help='FiftyOne dataset name')
    parser.add_argument('--compute', action='store_true', help='Compute CLIP embeddings')
    parser.add_argument('--visualize', action='store_true', help='Create UMAP visualization')
    parser.add_argument('--max-samples', type=int, help='Limit samples for testing')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for embedding computation')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device')
    parser.add_argument('--output-dir', type=str, default='outputs/embeddings', help='Output directory for visualizations')

    args = parser.parse_args()

    fo, F = _ensure_fiftyone()

    # Load dataset
    if not fo.dataset_exists(args.dataset):
        print(f"Error: Dataset '{args.dataset}' not found")
        print("Available datasets:", fo.list_datasets())
        return

    dataset = fo.load_dataset(args.dataset)
    print(f"Loaded dataset: {dataset.name} ({len(dataset)} samples)")

    # Compute embeddings
    if args.compute:
        embeddings, sample_ids = compute_clip_embeddings(
            dataset,
            batch_size=args.batch_size,
            device=args.device,
            max_samples=args.max_samples
        )

    # Create visualization
    if args.visualize:
        # Get embeddings from dataset
        embeddings, sample_ids, metadata = get_embeddings_from_dataset(dataset)

        if len(embeddings) == 0:
            print("Error: No embeddings found. Run with --compute first.")
            return

        # Compute UMAP
        coords = compute_umap(embeddings)

        # Create visualizations
        output_dir = PROJECT_ROOT / args.output_dir
        create_multi_view_visualization(
            coords, metadata, output_dir,
            title_prefix=f"Campo Images ({len(embeddings)} samples)"
        )

        print(f"\n{'='*60}")
        print("VISUALIZATION COMPLETE")
        print(f"{'='*60}")
        print("Open these files in your browser:")
        for html_file in (output_dir).glob("*.html"):
            print(f"  file://{html_file}")


if __name__ == '__main__':
    main()
