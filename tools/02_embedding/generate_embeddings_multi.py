#!/usr/bin/env python3
"""
Multi-Source Image Embedding Generation and Visualization Pipeline
===================================================================

Generates embeddings from multiple image collections (Ambrosia, Drones,
Lentejas_v2) using CLIP or ResNet50 backbones, then applies dimensionality
reduction (UMAP, t-SNE, PCA) and creates interactive visualizations.

Usage:
    # Default (auto-detect model, all sources)
    python generate_embeddings_multi.py

    # Use specific model
    python generate_embeddings_multi.py --model clip
    python generate_embeddings_multi.py --model resnet50

    # Quick test run
    python generate_embeddings_multi.py --max-images 500 --batch-size 32

    # Force regeneration (ignore cache)
    python generate_embeddings_multi.py --force

Output directory: /media/malezainia1/LORENZO/outputs_sugal_25-26/image_embeddings_multi/
    - embeddings.npz           : Raw embeddings + metadata
    - embeddings_reduced.csv   : Reduced coordinates with metadata
    - gps_metadata.csv         : GPS coordinates extracted from EXIF
    - visualization_*.html     : Interactive plotly visualizations
    - config.json              : Run configuration
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ragweed_toolkit.detection import extract_gps

# ragweed_toolkit library imports
from ragweed_toolkit.embeddings import (
    FeatureExtractor,
    default_transform,
    extract_embeddings,
    reduce_embeddings,
    umap_scatter,
)

warnings.filterwarnings("ignore")

# ============================================================================
# DATA SOURCES
# ============================================================================

SOURCES = [
    {
        "name": "ambrosia",
        "path": "/media/malezainia1/LORENZO/Lorenzo/Ambrosia_images/extracted/",
        "description": "Ambrosia/ragweed + other weed datasets",
    },
    {
        "name": "drones",
        "path": "/media/malezainia1/DRONES/imagenes pendiente orden/",
        "description": "Drone flights - lentejas, vuelos, arroz",
    },
    {
        "name": "lentejas_v2",
        "path": "/media/malezainia1/E/Proyecto_lentejas_v2/",
        "description": "Lentil project v2 - inference, fotos, outputs",
    },
]

OUTPUT_DIR = "/media/malezainia1/LORENZO/outputs_sugal_25-26/image_embeddings_multi"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".TIF",
    ".TIFF",
}

# ============================================================================
# MODEL: CLIP BACKEND
# ============================================================================


def _try_load_clip(device: str):
    """
    Attempt to load CLIP model. Try open_clip first, then transformers.
    Returns (model, preprocess, embedding_dim, backend_name) or raises ImportError.
    """
    # Try open_clip
    try:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        model = model.to(device).eval()
        return model, preprocess, 768, "open_clip"
    except (ImportError, Exception):
        pass

    # Try huggingface transformers
    try:
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model = model.to(device).eval()
        return model, processor, 768, "transformers"
    except (ImportError, Exception):
        pass

    raise ImportError(
        "CLIP is not available. Install open_clip_torch or transformers:\n"
        "  pip install open_clip_torch\n"
        "  pip install transformers"
    )


# ============================================================================
# MODEL: RESNET50 BACKEND
# ============================================================================

# ResNet50Extractor replaced by ragweed_toolkit.embeddings.FeatureExtractor.
# Alias for backward compatibility in this script.
ResNet50Extractor = None  # See FeatureExtractor("resnet50") from library


# ============================================================================
# MULTI-SOURCE IMAGE DATASET
# ============================================================================


class MultiSourceImageDataset(Dataset):
    """
    Dataset that loads images from multiple source directories,
    tagging each image with its source_collection and location.
    """

    def __init__(
        self,
        sources: List[Dict],
        transform=None,
        max_images: Optional[int] = None,
    ):
        self.sources = sources
        self.transform = transform
        self.image_paths: List[Path] = []
        self.metadata: List[Dict] = []

        for source in sources:
            src_path = Path(source["path"])
            src_name = source["name"]

            if not src_path.exists():
                print(f"WARNING: Source path does not exist, skipping: {src_path}")
                continue

            print(f"Scanning [{src_name}]: {src_path} ...")
            found = []
            for ext in IMAGE_EXTENSIONS:
                found.extend(list(src_path.rglob(f"*{ext}")))
            found.sort()

            print(f"  Found {len(found)} images in [{src_name}]")

            for img_path in found:
                rel_path = img_path.relative_to(src_path)
                parts = rel_path.parts

                # Location = first subfolder within the source
                location = parts[0] if len(parts) > 1 else "root"

                # Infer source_type from path components
                source_type = "unknown"
                str(img_path).lower()
                for part in parts:
                    part_lower = part.lower()
                    if "celular" in part_lower:
                        source_type = "Celular"
                        break
                    elif "dron" in part_lower or "vuelo" in part_lower or "flight" in part_lower:
                        source_type = "Dron"
                        break
                    elif "media" in part_lower or "camera" in part_lower:
                        source_type = "Camera"
                        break

                self.image_paths.append(img_path)
                self.metadata.append(
                    {
                        "filename": img_path.name,
                        "full_path": str(img_path),
                        "relative_path": str(rel_path),
                        "source_collection": src_name,
                        "location": location,
                        "source_type": source_type,
                    }
                )

        # Sort everything together for reproducibility
        paired = sorted(
            zip(self.image_paths, self.metadata), key=lambda x: str(x[0])
        )
        if paired:
            self.image_paths, self.metadata = map(list, zip(*paired))
        else:
            self.image_paths, self.metadata = [], []

        # Limit for testing
        if max_images is not None and max_images < len(self.image_paths):
            self.image_paths = self.image_paths[:max_images]
            self.metadata = self.metadata[:max_images]

        print(f"\nTotal images across all sources: {len(self.image_paths)}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, idx
        except Exception as e:
            # Return blank tensor for corrupted images
            tqdm.write(f"WARNING: Could not load {img_path}: {e}")
            return torch.zeros(3, 224, 224), idx


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================


def extract_embeddings_resnet(
    dataset: MultiSourceImageDataset,
    model: FeatureExtractor,
    batch_size: int,
    num_workers: int,
    device: str,
) -> np.ndarray:
    """Extract embeddings using ResNet50.

    Delegates to ragweed_toolkit.embeddings.extract_embeddings().
    """
    return extract_embeddings(
        images=dataset,
        model=model,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )


def extract_embeddings_clip(
    dataset: MultiSourceImageDataset,
    clip_model,
    clip_preprocess,
    clip_backend: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> np.ndarray:
    """Extract embeddings using CLIP (open_clip or transformers)."""

    if clip_backend == "open_clip":
        # open_clip: the preprocess is a torchvision transform
        # Override the dataset transform with CLIP's own preprocess
        dataset.transform = clip_preprocess

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
            persistent_workers=(num_workers > 0),
        )

        embeddings = []
        with torch.no_grad():
            for images, indices in tqdm(dataloader, desc="CLIP (open_clip) embedding"):
                images = images.to(device)
                features = clip_model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)
                embeddings.append(features.cpu().numpy())

        return np.vstack(embeddings)

    elif clip_backend == "transformers":
        # transformers CLIPModel: use processor which expects PIL images
        processor = clip_preprocess  # it's actually a CLIPProcessor

        # We need to load images as PIL, not through DataLoader with tensor transform
        embeddings = []
        n = len(dataset)
        for start in tqdm(range(0, n, batch_size), desc="CLIP (transformers) embedding"):
            end = min(start + batch_size, n)
            pil_images = []
            for i in range(start, end):
                try:
                    img = Image.open(dataset.image_paths[i]).convert("RGB")
                    pil_images.append(img)
                except Exception as e:
                    tqdm.write(f"WARNING: Could not load {dataset.image_paths[i]}: {e}")
                    # Use a blank image as fallback
                    pil_images.append(Image.new("RGB", (224, 224)))

            inputs = processor(images=pil_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = clip_model.get_image_features(**inputs)
                outputs = outputs / outputs.norm(dim=-1, keepdim=True)
                embeddings.append(outputs.cpu().numpy())

        return np.vstack(embeddings)

    else:
        raise ValueError(f"Unknown CLIP backend: {clip_backend}")


# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    **kwargs,
) -> np.ndarray:
    """
    Apply dimensionality reduction.

    Delegates to ragweed_toolkit.embeddings.reduce_embeddings() and returns
    raw numpy coordinates.
    """
    df = reduce_embeddings(
        embeddings,
        method=method,
        n_components=n_components,
        umap_n_neighbors=kwargs.get("n_neighbors", 15),
        umap_min_dist=kwargs.get("min_dist", 0.1),
        tsne_perplexity=kwargs.get("perplexity", 30),
    )
    prefix = method.lower()
    cols = [f"{prefix}_x", f"{prefix}_y"]
    if n_components >= 3:
        cols.append(f"{prefix}_z")
    return df[cols].values


# ============================================================================
# GPS EXTRACTION
# ============================================================================


def _get_exif_gps(img_path: Path) -> Optional[Dict]:
    """
    Extract GPS coordinates from an image's EXIF data.

    Delegates to ragweed_toolkit.detection.extract_gps().
    """
    return extract_gps(img_path)


def extract_gps_metadata(
    image_paths: List[Path], metadata: List[Dict]
) -> pd.DataFrame:
    """
    Extract GPS coordinates from all images. Returns DataFrame with
    filename, full_path, source_collection, latitude, longitude, altitude.
    """
    print("Extracting GPS metadata from EXIF ...")
    records = []
    gps_found = 0

    for i, img_path in enumerate(tqdm(image_paths, desc="GPS extraction")):
        gps = _get_exif_gps(img_path)
        rec = {
            "filename": metadata[i]["filename"],
            "full_path": metadata[i]["full_path"],
            "source_collection": metadata[i]["source_collection"],
            "location": metadata[i]["location"],
            "latitude": gps.get("latitude") if gps else None,
            "longitude": gps.get("longitude") if gps else None,
            "altitude": gps.get("altitude") if gps else None,
            "has_gps": gps is not None and "latitude" in gps,
        }
        records.append(rec)
        if rec["has_gps"]:
            gps_found += 1

    print(f"  GPS found in {gps_found}/{len(image_paths)} images ({100*gps_found/max(1,len(image_paths)):.1f}%)")
    return pd.DataFrame(records)


# ============================================================================
# VISUALIZATION
# ============================================================================


def create_visualization(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: Optional[str] = None,
    color_col: str = "source_collection",
    title: str = "Embedding Visualization",
    output_path: str = "visualization.html",
):
    """Create interactive plotly scatter plot (2D or 3D).

    Delegates to ragweed_toolkit.embeddings.umap_scatter().
    """
    return umap_scatter(
        df,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        color_col=color_col,
        title=title,
        output_path=output_path,
        width=1400,
        height=900,
        marker_size=4,
    )


def create_all_visualizations(df: pd.DataFrame, output_dir: Path):
    """Generate all visualization variants."""
    print("\nCreating visualizations ...")

    reduction_methods = [
        ("umap_x", "umap_y", None, "UMAP 2D"),
        ("tsne_x", "tsne_y", None, "t-SNE 2D"),
        ("pca_x", "pca_y", None, "PCA 2D"),
        ("umap_3d_x", "umap_3d_y", "umap_3d_z", "UMAP 3D"),
    ]

    # Visualizations colored by source_collection
    for x, y, z, label in reduction_methods:
        slug = label.lower().replace(" ", "_").replace("-", "")
        create_visualization(
            df,
            x_col=x,
            y_col=y,
            z_col=z,
            color_col="source_collection",
            title=f"{label} - by Source Collection",
            output_path=str(output_dir / f"viz_{slug}_by_collection.html"),
        )

    # Visualizations colored by location (UMAP 2D only, to avoid too many files)
    create_visualization(
        df,
        x_col="umap_x",
        y_col="umap_y",
        color_col="location",
        title="UMAP 2D - by Location",
        output_path=str(output_dir / "viz_umap_2d_by_location.html"),
    )

    # Visualization colored by source_type
    create_visualization(
        df,
        x_col="umap_x",
        y_col="umap_y",
        color_col="source_type",
        title="UMAP 2D - by Source Type",
        output_path=str(output_dir / "viz_umap_2d_by_source_type.html"),
    )


# ============================================================================
# ARGUMENT PARSING
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-source image embedding extraction and visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_embeddings_multi.py                     # Full run, auto model
    python generate_embeddings_multi.py --model clip        # Force CLIP
    python generate_embeddings_multi.py --model resnet50    # Force ResNet50
    python generate_embeddings_multi.py --max-images 500    # Quick test
    python generate_embeddings_multi.py --force             # Regenerate embeddings
    python generate_embeddings_multi.py --skip-gps          # Skip GPS extraction
    python generate_embeddings_multi.py --skip-dimreduce    # Only extract embeddings
        """,
    )
    parser.add_argument(
        "--model",
        choices=["clip", "resnet50"],
        default="clip",
        help="Embedding model (default: clip, falls back to resnet50 if unavailable)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding extraction (default: 64)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: cuda or cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit total images for testing (e.g. 500)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader workers (default: 8)",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if embeddings.npz exists",
    )
    parser.add_argument(
        "--skip-gps",
        action="store_true",
        help="Skip GPS EXIF extraction step",
    )
    parser.add_argument(
        "--skip-dimreduce",
        action="store_true",
        help="Skip dimensionality reduction and visualization (only extract embeddings)",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter (default: 15)",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter (default: 0.1)",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity parameter (default: 30)",
    )
    return parser.parse_args()


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    args = parse_args()

    # Device selection
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MULTI-SOURCE IMAGE EMBEDDING PIPELINE")
    print("=" * 70)
    print(f"  Device       : {device}")
    print(f"  Requested    : {args.model}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Workers      : {args.num_workers}")
    print(f"  Max images   : {args.max_images or 'all'}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Force regen  : {args.force}")
    print(f"  Timestamp    : {datetime.now().isoformat()}")
    print()

    # ------------------------------------------------------------------
    # Resolve model backend
    # ------------------------------------------------------------------
    actual_model = args.model
    clip_model = None
    clip_preprocess = None
    clip_backend = None
    resnet_model = None
    embedding_dim = None

    if args.model == "clip":
        try:
            clip_model, clip_preprocess, embedding_dim, clip_backend = _try_load_clip(device)
            print(f"CLIP loaded via [{clip_backend}], embedding_dim={embedding_dim}")
        except ImportError as e:
            print(f"CLIP not available: {e}")
            print("Falling back to ResNet50 ...")
            actual_model = "resnet50"

    if actual_model == "resnet50":
        resnet_model = FeatureExtractor("resnet50")
        embedding_dim = resnet_model.embedding_dim
        print(f"ResNet50 loaded, embedding_dim={embedding_dim}")

    print()

    # ------------------------------------------------------------------
    # STEP 1: Scan images from all sources
    # ------------------------------------------------------------------
    print("-" * 70)
    print("STEP 1: Scanning images from all sources")
    print("-" * 70)

    # Standard ImageNet transform for ResNet50 (from ragweed_toolkit.embeddings)
    resnet_transform = default_transform()

    # For CLIP with open_clip, the transform comes from the model
    # For CLIP with transformers, we handle PIL images directly
    # For ResNet50, use standard ImageNet transform
    if actual_model == "resnet50" or (actual_model == "clip" and clip_backend == "transformers"):
        ds_transform = resnet_transform
    else:
        # open_clip provides its own transform
        ds_transform = clip_preprocess

    dataset = MultiSourceImageDataset(
        sources=SOURCES,
        transform=ds_transform,
        max_images=args.max_images,
    )

    if len(dataset) == 0:
        print("ERROR: No images found across any source!")
        sys.exit(1)

    # Print source breakdown
    from collections import Counter
    src_counts = Counter(m["source_collection"] for m in dataset.metadata)
    print("\nImages per source:")
    for src, count in src_counts.most_common():
        print(f"  {src}: {count:,}")
    print()

    # ------------------------------------------------------------------
    # STEP 2: Extract or load embeddings
    # ------------------------------------------------------------------
    print("-" * 70)
    print("STEP 2: Extracting embeddings")
    print("-" * 70)

    embeddings_path = output_dir / "embeddings.npz"

    if embeddings_path.exists() and not args.force:
        print(f"Loading cached embeddings from {embeddings_path}")
        data = np.load(embeddings_path, allow_pickle=True)
        embeddings = data["embeddings"]
        metadata = data["metadata"].tolist()
        print(f"Loaded {len(embeddings)} embeddings, shape: {embeddings.shape}")

        # Verify the cache matches current dataset
        if len(embeddings) != len(dataset):
            print(
                f"WARNING: Cache has {len(embeddings)} entries but dataset has "
                f"{len(dataset)}. Re-extracting..."
            )
            args.force = True

    if not embeddings_path.exists() or args.force:
        if actual_model == "clip":
            embeddings = extract_embeddings_clip(
                dataset=dataset,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                clip_backend=clip_backend,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
        else:
            embeddings = extract_embeddings_resnet(
                dataset=dataset,
                model=resnet_model,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )

        metadata = dataset.metadata

        # Save cache
        np.savez_compressed(
            embeddings_path,
            embeddings=embeddings,
            metadata=np.array(metadata),
        )
        print(f"Saved embeddings to: {embeddings_path}")

    print(f"Embeddings shape: {embeddings.shape}")
    print()

    # Save config
    config = {
        "model": actual_model,
        "clip_backend": clip_backend,
        "embedding_dim": embedding_dim,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_images": args.max_images,
        "device": device,
        "n_images": len(embeddings),
        "sources": SOURCES,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ------------------------------------------------------------------
    # STEP 3: GPS extraction
    # ------------------------------------------------------------------
    if not args.skip_gps:
        print("-" * 70)
        print("STEP 3: GPS EXIF extraction")
        print("-" * 70)

        gps_path = output_dir / "gps_metadata.csv"
        gps_df = extract_gps_metadata(dataset.image_paths, metadata)
        gps_df.to_csv(gps_path, index=False)
        print(f"Saved GPS metadata to: {gps_path}")
        print()
    else:
        print("Skipping GPS extraction (--skip-gps)")
        print()

    # ------------------------------------------------------------------
    # STEP 4: Dimensionality reduction
    # ------------------------------------------------------------------
    if args.skip_dimreduce:
        print("Skipping dimensionality reduction (--skip-dimreduce)")
        print()
        _print_summary(output_dir, embeddings, metadata)
        return

    print("-" * 70)
    print("STEP 4: Dimensionality reduction")
    print("-" * 70)

    print("\n[1/4] UMAP 2D")
    umap_2d = reduce_dimensions(
        embeddings,
        method="umap",
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )

    print("\n[2/4] UMAP 3D")
    umap_3d = reduce_dimensions(
        embeddings,
        method="umap",
        n_components=3,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )

    print("\n[3/4] t-SNE 2D")
    tsne_2d = reduce_dimensions(
        embeddings,
        method="tsne",
        n_components=2,
        perplexity=args.tsne_perplexity,
    )

    print("\n[4/4] PCA 2D")
    pca_2d = reduce_dimensions(embeddings, method="pca", n_components=2)
    print()

    # ------------------------------------------------------------------
    # STEP 5: Build combined DataFrame
    # ------------------------------------------------------------------
    print("-" * 70)
    print("STEP 5: Building combined dataset")
    print("-" * 70)

    df = pd.DataFrame(metadata)
    df["umap_x"] = umap_2d[:, 0]
    df["umap_y"] = umap_2d[:, 1]
    df["umap_3d_x"] = umap_3d[:, 0]
    df["umap_3d_y"] = umap_3d[:, 1]
    df["umap_3d_z"] = umap_3d[:, 2]
    df["tsne_x"] = tsne_2d[:, 0]
    df["tsne_y"] = tsne_2d[:, 1]
    df["pca_x"] = pca_2d[:, 0]
    df["pca_y"] = pca_2d[:, 1]

    csv_path = output_dir / "embeddings_reduced.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Statistics
    print("\nDataset statistics:")
    print(f"  Total images: {len(df):,}")
    print("\n  By source_collection:")
    for src, count in df["source_collection"].value_counts().items():
        print(f"    {src}: {count:,}")
    print("\n  By location (top 20):")
    for loc, count in df["location"].value_counts().head(20).items():
        print(f"    {loc}: {count:,}")
    print("\n  By source_type:")
    for st, count in df["source_type"].value_counts().items():
        print(f"    {st}: {count:,}")
    print()

    # ------------------------------------------------------------------
    # STEP 6: Visualizations
    # ------------------------------------------------------------------
    print("-" * 70)
    print("STEP 6: Creating visualizations")
    print("-" * 70)

    create_all_visualizations(df, output_dir)
    print()

    # ------------------------------------------------------------------
    # DONE
    # ------------------------------------------------------------------
    _print_summary(output_dir, embeddings, metadata)


def _print_summary(output_dir: Path, embeddings: np.ndarray, metadata: list):
    """Print final summary."""
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"  Output directory: {output_dir}")
    print(f"  Total images:     {len(embeddings):,}")
    print(f"  Embedding shape:  {embeddings.shape}")
    print()
    print("Generated files:")
    for f in sorted(output_dir.glob("*")):
        size = f.stat().st_size / 1024
        unit = "KB"
        if size > 1024:
            size /= 1024
            unit = "MB"
        print(f"  {f.name}: {size:.1f} {unit}")


if __name__ == "__main__":
    main()
