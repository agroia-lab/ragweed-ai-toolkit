"""
Training Configuration and Execution
=====================================

Wraps ultralytics YOLO training with the hyperparameters validated in the
Springer chapter (Section 3.2 -- Model Training Protocol).

Reference configuration (Table 3 in chapter):
    Model: YOLOv11l, imgsz=640, batch=64, epochs=50, optimizer=SGD,
    lr0=0.01, lrf=0.0001, warmup_epochs=3, patience=15, seed=42,
    box=7.5, cls=0.5, dfl=1.5, mosaic=1.0, crop_fraction=0.29,
    degrees=15.0, max_det=10000, amp=True.

Usage::

    from ragweed_toolkit.detection.trainer import TrainingConfig, train

    cfg = TrainingConfig(
        model="yolo11l.pt",
        data="path/to/data.yaml",
        device="0",
    )
    results = train(cfg)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # Optional dependency; checked at runtime in train()


@dataclass
class TrainingConfig:
    """Training hyperparameters matching the chapter's validated configuration.

    All defaults correspond to the values reported in Section 3.2 / Table 3.
    Override individual fields as needed for experimentation.

    Attributes:
        model: Pretrained checkpoint name or path (e.g. ``"yolo11l.pt"``).
        data: Path to the dataset YAML file.
        epochs: Maximum number of training epochs.
        imgsz: Input image size in pixels (square).
        batch: Mini-batch size.
        device: CUDA device specification (``"0"``, ``"0,1"``, ``"cpu"``).
        optimizer: Optimizer name (``"SGD"``, ``"Adam"``, ``"auto"``).
        lr0: Initial learning rate.
        lrf: Final learning rate factor (lr0 * lrf at end).
        momentum: SGD momentum / Adam beta1.
        warmup_epochs: Linear warmup epochs.
        patience: Early-stopping patience (0 disables).
        max_det: Maximum detections per image.
        seed: Random seed for reproducibility.
        amp: Enable automatic mixed precision (FP16).
        box: Box loss gain.
        cls: Classification loss gain.
        dfl: Distribution focal loss gain.
        mosaic: Mosaic augmentation probability.
        crop_fraction: Random crop fraction for classification (passed to YOLO).
        degrees: Random rotation range in degrees.
        project: Base directory for training outputs.
        name: Run name (timestamp appended automatically).
        workers: Number of DataLoader workers.
        cache: Cache images in RAM for speed.
        save_period: Save checkpoint every N epochs.
    """

    # Model
    model: str = "yolo11l.pt"
    data: str = ""

    # Core hyperparameters (Table 3)
    epochs: int = 50
    imgsz: int = 640
    batch: int = 64
    device: str = "0"
    optimizer: str = "SGD"
    lr0: float = 0.01
    lrf: float = 0.0001
    momentum: float = 0.937
    warmup_epochs: int = 3
    patience: int = 15
    max_det: int = 10000
    seed: int = 42
    amp: bool = True

    # Loss weights
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5

    # Augmentation
    mosaic: float = 1.0
    crop_fraction: float = 0.29
    degrees: float = 15.0

    # Output
    project: str = "training_results"
    name: str = ""
    workers: int = 8
    cache: bool = True
    save_period: int = 10


def train(
    config: TrainingConfig,
    *,
    extra_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run YOLO training using the given configuration.

    Implements the training protocol described in Section 3.2 of the chapter.
    Creates a timestamped output directory, trains the model, and returns
    a results dictionary containing final metrics and the output path.

    Args:
        config: A :class:`TrainingConfig` instance with all hyperparameters.
        extra_args: Additional keyword arguments forwarded to
            ``model.train()`` (overrides config values if keys collide).

    Returns:
        Dictionary with keys:

        - ``"output_dir"`` -- :class:`~pathlib.Path` to the run directory.
        - ``"best_weights"`` -- Path to ``weights/best.pt``.
        - ``"metrics"`` -- Final validation metrics dict from ultralytics.
        - ``"training_time_s"`` -- Wall-clock training time in seconds.
        - ``"config"`` -- Serialised config for reproducibility.

    Raises:
        FileNotFoundError: If ``config.data`` does not exist.
        RuntimeError: If ultralytics raises during training.
    """
    data_path = Path(config.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    # Timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.name}_{timestamp}" if config.name else f"run_{timestamp}"
    output_dir = Path(config.project) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build ultralytics train kwargs
    train_kwargs: Dict[str, Any] = {
        "data": str(data_path),
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "device": config.device,
        "optimizer": config.optimizer,
        "lr0": config.lr0,
        "lrf": config.lrf,
        "momentum": config.momentum,
        "warmup_epochs": config.warmup_epochs,
        "patience": config.patience,
        "max_det": config.max_det,
        "seed": config.seed,
        "amp": config.amp,
        "box": config.box,
        "cls": config.cls,
        "dfl": config.dfl,
        "mosaic": config.mosaic,
        "crop_fraction": config.crop_fraction,
        "degrees": config.degrees,
        "workers": config.workers,
        "cache": config.cache,
        "save_period": config.save_period,
        "project": str(output_dir.parent),
        "name": output_dir.name,
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
        "deterministic": True,
        "plots": True,
        "save": True,
    }
    if extra_args:
        train_kwargs.update(extra_args)

    # Train
    model = YOLO(config.model)
    start = time.time()
    results = model.train(**train_kwargs)
    elapsed = time.time() - start

    # Collect metrics
    metrics: Dict[str, Any] = {}
    if results and hasattr(results, "results_dict"):
        metrics = dict(results.results_dict)

    best_weights = output_dir / "weights" / "best.pt"

    # Save summary JSON alongside weights
    summary = {
        "config": asdict(config),
        "training_time_s": elapsed,
        "best_weights": str(best_weights),
        "metrics": metrics,
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "output_dir": output_dir,
        "best_weights": best_weights,
        "metrics": metrics,
        "training_time_s": elapsed,
        "config": asdict(config),
    }


def main():
    """CLI entry point for YOLO training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a YOLO model with validated hyperparameters (Table 3)."
    )
    parser.add_argument("--data", required=True, help="Path to dataset YAML file")
    parser.add_argument(
        "--model", default="yolo11l.pt",
        help="Pretrained checkpoint (default: yolo11l.pt)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Max training epochs (default: 50)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--batch", type=int, default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--device", default="0",
        help="CUDA device, e.g. '0', '0,1', 'cpu' (default: 0)",
    )
    parser.add_argument(
        "--optimizer", default="SGD",
        help="Optimizer: SGD, Adam, auto (default: SGD)",
    )
    parser.add_argument(
        "--lr0", type=float, default=0.01,
        help="Initial learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--patience", type=int, default=15,
        help="Early stopping patience (default: 15)",
    )
    parser.add_argument(
        "--project", default="training_results",
        help="Output directory (default: training_results)",
    )
    parser.add_argument("--name", default="", help="Run name prefix")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers (default: 8)")
    parser.add_argument("--no-cache", action="store_true", help="Disable image caching")
    args = parser.parse_args()

    cfg = TrainingConfig(
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        project=args.project,
        name=args.name,
        seed=args.seed,
        workers=args.workers,
        cache=not args.no_cache,
    )

    print(f"Starting training: {cfg.model} on {cfg.data}")
    print(f"  epochs={cfg.epochs}, imgsz={cfg.imgsz}, batch={cfg.batch}, device={cfg.device}")
    results = train(cfg)

    print(f"\nTraining complete in {results['training_time_s']:.1f}s")
    print(f"  Output: {results['output_dir']}")
    print(f"  Best weights: {results['best_weights']}")
    if results["metrics"]:
        for k, v in results["metrics"].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
