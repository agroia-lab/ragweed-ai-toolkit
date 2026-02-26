"""
Model Evaluation
================

Wraps ultralytics ``model.val()`` to compute standard COCO-protocol
detection metrics: mAP50, mAP50:95, Precision, and Recall.

Reference: Section 3.4 -- Evaluation Metrics, Table 4.

Usage::

    from ragweed_toolkit.detection.evaluation import evaluate

    metrics = evaluate(
        model_path="weights/best.pt",
        data="path/to/data.yaml",
    )
    print(metrics["mAP50"])      # 0.943
    print(metrics["mAP50_95"])   # 0.726
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from ultralytics import YOLO


def evaluate(
    model_path: Union[str, Path],
    data: Union[str, Path],
    *,
    imgsz: int = 640,
    batch: int = 64,
    device: str = "0",
    split: str = "test",
    conf: float = 0.001,
    iou: float = 0.6,
    max_det: int = 10000,
    verbose: bool = True,
    extra_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate a trained YOLO model on a dataset split.

    Runs the ultralytics validation loop and extracts the four headline
    metrics reported in the chapter (Section 3.4, Table 4).

    Args:
        model_path: Path to a trained ``.pt`` checkpoint.
        data: Path to the dataset YAML (must contain the requested split).
        imgsz: Inference image size in pixels.
        batch: Batch size for validation.
        device: CUDA device (e.g. ``"0"``, ``"cpu"``).
        split: Dataset split to evaluate (``"val"`` or ``"test"``).
        conf: Confidence threshold for metrics computation.
        iou: IoU threshold for NMS.
        max_det: Maximum detections per image.
        verbose: Print ultralytics progress output.
        extra_args: Additional keyword arguments forwarded to
            ``model.val()``.

    Returns:
        Dictionary with keys:

        - ``"mAP50"`` (float): Mean AP at IoU = 0.5.
        - ``"mAP50_95"`` (float): Mean AP at IoU = 0.5:0.95.
        - ``"precision"`` (float): Mean precision.
        - ``"recall"`` (float): Mean recall.
        - ``"per_class"`` (dict): Per-class metrics keyed by class name.
        - ``"speed"`` (dict): Inference speed breakdown from ultralytics.
        - ``"raw"`` -- Full ``results_dict`` from ultralytics for advanced use.
    """
    model = YOLO(str(model_path))

    val_kwargs: Dict[str, Any] = {
        "data": str(data),
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "split": split,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "verbose": verbose,
    }
    if extra_args:
        val_kwargs.update(extra_args)

    results = model.val(**val_kwargs)

    # Extract headline metrics
    metrics: Dict[str, Any] = {
        "mAP50": float(getattr(results, "map50", 0) or 0),
        "mAP50_95": float(getattr(results, "map", 0) or 0),
        "precision": float(getattr(results, "mp", 0) or 0),
        "recall": float(getattr(results, "mr", 0) or 0),
    }

    # Per-class metrics
    per_class: Dict[str, Dict[str, float]] = {}
    class_names = model.names or {}
    if hasattr(results, "ap50") and results.ap50 is not None:
        for i, ap50_val in enumerate(results.ap50):
            cls_name = class_names.get(i, f"class_{i}")
            per_class[cls_name] = {
                "AP50": float(ap50_val),
            }
    # Add per-class P/R if available
    if hasattr(results, "p") and results.p is not None:
        for i, p_val in enumerate(results.p):
            cls_name = class_names.get(i, f"class_{i}")
            if cls_name not in per_class:
                per_class[cls_name] = {}
            per_class[cls_name]["precision"] = float(p_val)
    if hasattr(results, "r") and results.r is not None:
        for i, r_val in enumerate(results.r):
            cls_name = class_names.get(i, f"class_{i}")
            if cls_name not in per_class:
                per_class[cls_name] = {}
            per_class[cls_name]["recall"] = float(r_val)

    metrics["per_class"] = per_class

    # Speed info
    if hasattr(results, "speed"):
        metrics["speed"] = dict(results.speed)
    else:
        metrics["speed"] = {}

    # Raw results dict for advanced usage
    if hasattr(results, "results_dict"):
        metrics["raw"] = dict(results.results_dict)
    else:
        metrics["raw"] = {}

    return metrics
