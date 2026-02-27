"""
Feature Extraction with Pre-trained CNN Backbones
===================================================

Extracts fixed-size embeddings from images using ImageNet-pretrained models.
The default backbone is ResNet-50 (ImageNet V2 weights) which produces
2048-dimensional feature vectors via global average pooling.

This corresponds to the feature extraction stage described in Chapter 6
(Eq. 1: phi(x) -> R^d, where d=2048 for ResNet-50).

Usage::

    from ragweed_toolkit.embeddings import FeatureExtractor, extract_embeddings

    extractor = FeatureExtractor("resnet50")
    embeddings = extract_embeddings(image_paths, extractor, device="cuda:0")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default image size for all backbones
DEFAULT_IMAGE_SIZE = 224


class FeatureExtractor(nn.Module):
    """Extract fixed-size embeddings from images using a pre-trained CNN backbone.

    Removes the classification head and applies global average pooling to produce
    a single feature vector per image.

    Parameters
    ----------
    model_name : str
        Backbone architecture. One of ``"resnet50"`` (2048-D),
        ``"efficientnet_b0"`` (1280-D), or ``"efficientnet_b2"`` (1408-D).

    Attributes
    ----------
    embedding_dim : int
        Dimensionality of the output feature vector.
    backbone : nn.Module
        Truncated backbone (without classification head).

    Examples
    --------
    >>> extractor = FeatureExtractor("resnet50")
    >>> extractor.embedding_dim
    2048
    >>> dummy = torch.randn(1, 3, 224, 224)
    >>> extractor.eval()
    >>> with torch.no_grad():
    ...     features = extractor(dummy)
    >>> features.shape
    torch.Size([1, 2048])
    """

    def __init__(self, model_name: str = "resnet50") -> None:
        super().__init__()

        if model_name == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            self.embedding_dim = 2048
        elif model_name == "efficientnet_b0":
            base = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            self.embedding_dim = 1280
        elif model_name == "efficientnet_b2":
            base = models.efficientnet_b2(
                weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
            )
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            self.embedding_dim = 1408
        else:
            raise ValueError(
                f"Unknown model: {model_name!r}. "
                f"Choose from: resnet50, efficientnet_b0, efficientnet_b2"
            )

        self._pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of images, shape ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Feature vectors, shape ``(B, embedding_dim)``.
        """
        features = self.backbone(x)
        if features.ndim == 4:
            features = self._pool(features)
        return features.flatten(1)


def default_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """Return the standard inference transform (resize + ImageNet normalization).

    No augmentation is applied -- this is for deterministic feature extraction.

    Parameters
    ----------
    image_size : int
        Target spatial size (both height and width).

    Returns
    -------
    torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class _ImageListDataset(Dataset):
    """Internal dataset that loads images from a list of file paths."""

    IMAGE_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".tif", ".tiff",
        ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF",
    }

    def __init__(
        self,
        paths: Sequence[Union[str, Path]],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.paths = [Path(p) for p in paths]
        self.transform = transform or default_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return image, idx
        except Exception:
            # Return a blank tensor for corrupted images
            return torch.zeros(3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), idx


def scan_image_directory(
    root: Union[str, Path],
    max_images: Optional[int] = None,
) -> List[Path]:
    """Recursively find all image files under *root*.

    Parameters
    ----------
    root : path-like
        Directory to scan.
    max_images : int, optional
        If given, return at most this many paths (sorted alphabetically).

    Returns
    -------
    list[Path]
        Sorted list of image file paths.
    """
    root = Path(root)
    paths: List[Path] = []
    for ext in _ImageListDataset.IMAGE_EXTENSIONS:
        paths.extend(root.rglob(f"*{ext}"))
    paths.sort()
    if max_images is not None:
        paths = paths[:max_images]
    return paths


def extract_embeddings(
    images: Union[Sequence[Union[str, Path]], Dataset],
    model: FeatureExtractor,
    *,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
    transform: Optional[transforms.Compose] = None,
) -> np.ndarray:
    """Extract embeddings for a collection of images.

    Parameters
    ----------
    images : sequence of paths or torch Dataset
        Either a list of image file paths, or a ``torch.utils.data.Dataset``
        that yields ``(tensor, index)`` tuples.
    model : FeatureExtractor
        The backbone model.
    batch_size : int
        Batch size for the DataLoader.
    num_workers : int
        Number of DataLoader workers.
    device : str
        PyTorch device string (e.g. ``"cuda:0"``, ``"cpu"``).
    transform : transforms.Compose, optional
        Image transform. Only used when *images* is a list of paths.
        Defaults to :func:`default_transform`.

    Returns
    -------
    np.ndarray
        Embeddings array of shape ``(N, embedding_dim)``.
    """
    if isinstance(images, Dataset):
        dataset = images
    else:
        dataset = _ImageListDataset(images, transform=transform)

    model = model.to(device)
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=("cuda" in device),
    )

    all_embeddings: List[np.ndarray] = []

    with torch.no_grad():
        for batch_images, _ in tqdm(dataloader, desc="Extracting embeddings"):
            batch_images = batch_images.to(device)
            features = model(batch_images)
            all_embeddings.append(features.cpu().numpy())

    return np.vstack(all_embeddings)


def main():
    """CLI entry point for feature extraction."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract CNN embeddings from a directory of images."
    )
    parser.add_argument("--source", required=True, help="Directory containing images")
    parser.add_argument("--output", required=True, help="Output .npy file for embeddings")
    parser.add_argument("--backbone", default="resnet50",
                        choices=["resnet50", "efficientnet_b0", "efficientnet_b2"],
                        help="CNN backbone (default: resnet50)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to process")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4)")
    args = parser.parse_args()

    image_paths = scan_image_directory(args.source, max_images=args.max_images)
    if not image_paths:
        print(f"No images found in {args.source}")
        return

    print(f"Found {len(image_paths)} images in {args.source}")
    print(f"Backbone: {args.backbone}, device: {args.device}")

    extractor = FeatureExtractor(args.backbone)
    embeddings = extract_embeddings(
        image_paths,
        extractor,
        batch_size=args.batch_size,
        num_workers=args.workers,
        device=args.device,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), embeddings)
    print(f"\nSaved {embeddings.shape} embeddings to {out_path}")


if __name__ == "__main__":
    main()
