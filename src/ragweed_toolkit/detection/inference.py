"""
SAHI Sliced Inference
=====================

Wraps the SAHI library for sliced aided hyper inference on high-resolution
field imagery (Section 3.3 -- Sliced Inference Protocol).

Key parameters from the chapter:
    slice_size=640, overlap_ratio=0.2, confidence_threshold=0.25,
    nms_iou=0.5, max_det=10000.

Usage::

    from ragweed_toolkit.detection.inference import SahiConfig, run_sahi, run_sahi_batch

    cfg = SahiConfig(model_path="weights/best.pt", device="cuda:0")
    detections = run_sahi("image.jpg", cfg)

    # Batch mode
    results = run_sahi_batch(["img1.jpg", "img2.jpg"], cfg)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # Optional dependency; checked at runtime in run_sahi()


@dataclass
class SahiConfig:
    """SAHI inference parameters matching Section 3.3 of the chapter.

    Attributes:
        model_path: Path to a trained YOLO ``.pt`` checkpoint.
        slice_size: Height and width of each inference slice in pixels.
        overlap_ratio: Fractional overlap between adjacent slices.
        confidence_threshold: Minimum detection confidence to keep.
        nms_iou: NMS IoU threshold applied by SAHI post-merge.
        max_det: Maximum detections per image (default 10000 for dense
            agricultural scenes; the ultralytics default of 300 is too low).
        device: CUDA device string (e.g. ``"cuda:0"``, ``"cpu"``).
    """

    model_path: str = ""
    slice_size: int = 640
    overlap_ratio: float = 0.2
    confidence_threshold: float = 0.25
    nms_iou: float = 0.5
    max_det: int = 10000
    device: str = "cuda:0"


def _load_sahi_model(config: SahiConfig) -> Any:
    """Load a SAHI ``AutoDetectionModel`` from a YOLO checkpoint.

    Also overrides ``max_det`` inside the underlying ultralytics model
    because the default (300) silently drops detections in dense scenes.
    """
    from sahi import AutoDetectionModel

    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=config.model_path,
        confidence_threshold=config.confidence_threshold,
        device=config.device,
    )
    # Override max_det for dense agricultural scenes
    try:
        sahi_model.model.model.args["max_det"] = config.max_det
    except Exception:
        pass
    return sahi_model


def run_sahi(
    image: Union[str, Path],
    config: SahiConfig,
    *,
    _sahi_model: Any = None,
) -> List[Dict[str, Any]]:
    """Run SAHI sliced inference on a single image.

    Implements the sliced inference protocol described in Section 3.3:
    the image is divided into overlapping tiles of ``slice_size`` pixels,
    each tile is passed through the YOLO detector, and results are merged
    with NMS to remove duplicates at tile boundaries.

    Args:
        image: Path to the input image file.
        config: A :class:`SahiConfig` with inference parameters.
        _sahi_model: Pre-loaded SAHI model (for batch reuse). If ``None``
            the model is loaded from ``config.model_path``.

    Returns:
        List of detection dictionaries, each containing:

        - ``"class_id"`` (int): Numeric class index.
        - ``"class_name"`` (str): Human-readable class name.
        - ``"bbox"`` (list[float]): ``[x1, y1, x2, y2]`` in pixel coords.
        - ``"confidence"`` (float): Detection confidence score.
    """
    from sahi.predict import get_sliced_prediction

    if _sahi_model is None:
        _sahi_model = _load_sahi_model(config)

    _sahi_model.confidence_threshold = config.confidence_threshold

    result = get_sliced_prediction(
        str(image),
        _sahi_model,
        slice_height=config.slice_size,
        slice_width=config.slice_size,
        overlap_height_ratio=config.overlap_ratio,
        overlap_width_ratio=config.overlap_ratio,
        verbose=0,
    )

    detections: List[Dict[str, Any]] = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox.to_xyxy()
        detections.append(
            {
                "class_id": pred.category.id,
                "class_name": pred.category.name,
                "bbox": [float(c) for c in bbox],
                "confidence": float(pred.score.value),
            }
        )
    return detections


def run_sahi_batch(
    images: Sequence[Union[str, Path]],
    config: SahiConfig,
    *,
    verbose: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run SAHI inference on multiple images, loading the model once.

    Args:
        images: Sequence of image file paths.
        config: A :class:`SahiConfig` with inference parameters.
        verbose: Print progress to stdout.

    Returns:
        Dictionary mapping each image filename (stem) to its list of
        detection dicts (same format as :func:`run_sahi`).
    """
    sahi_model = _load_sahi_model(config)

    results: Dict[str, List[Dict[str, Any]]] = {}
    for idx, img_path in enumerate(images):
        img_path = Path(img_path)
        if verbose:
            print(f"[{idx + 1}/{len(images)}] {img_path.name}")
        dets = run_sahi(img_path, config, _sahi_model=sahi_model)
        results[img_path.stem] = dets
        if verbose:
            print(f"  {len(dets)} detections")

    return results


def main():
    """CLI entry point for SAHI sliced inference."""
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(
        description="Run SAHI sliced inference on images with a YOLO model."
    )
    parser.add_argument("--model", required=True, help="Path to trained YOLO .pt checkpoint")
    parser.add_argument("--images", required=True, nargs="+", help="Image file(s) or directory")
    parser.add_argument("--slice-size", type=int, default=640, help="Slice size in pixels (default: 640)")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio (default: 0.2)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--device", default="cuda:0", help="Device (default: cuda:0)")
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    # Expand directories into image lists
    image_paths = []
    for img_arg in args.images:
        p = Path(img_arg)
        if p.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
                image_paths.extend(sorted(p.glob(ext)))
                image_paths.extend(sorted(p.glob(ext.upper())))
        elif p.is_file():
            image_paths.append(p)
        else:
            print(f"Warning: {img_arg} not found, skipping")

    if not image_paths:
        print("No images found.")
        return

    cfg = SahiConfig(
        model_path=args.model,
        slice_size=args.slice_size,
        overlap_ratio=args.overlap,
        confidence_threshold=args.conf,
        device=args.device,
    )

    print(f"SAHI inference: {len(image_paths)} images, slice={cfg.slice_size}, conf={cfg.confidence_threshold}")
    results = run_sahi_batch(image_paths, cfg)

    total_dets = sum(len(dets) for dets in results.values())
    print(f"\nTotal: {total_dets} detections across {len(results)} images")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            _json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
