"""
Cross-Domain Evaluation
========================

Run a YOLO model on a multi-database test set and compute per-database
mAP breakdown using a manifest file that maps images to source databases.

Functions
---------
load_manifest
    Load manifest.csv mapping filenames to source databases.
count_labels_per_image
    Count ground truth annotations per image from YOLO label files.
compute_per_database_metrics
    Group images by source database and compute detection statistics.
run_crossdomain_evaluation
    End-to-end evaluation: YOLO val + per-database breakdown.
"""

import csv
import json
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Manifest / label helpers
# ---------------------------------------------------------------------------

# Short display names for known databases
SHORT_NAMES = {
    "ND_Indiv": "ND Individual",
    "ND_Aerial": "ND Aerial",
    "MI_3Season": "MI 3-Season",
    "Purdue": "Purdue 4-Weed",
    "WeedCube": "USDA WeedCube",
    "WeedCrop": "WeedCrop PrecAg",
    "CL_Seba": "CL Seba",
    "CL_Alberto": "CL Alberto",
    "CL_StaRosa": "CL StaRosa",
}

# Known filename prefixes for fallback grouping
KNOWN_PREFIXES = list(SHORT_NAMES.keys())


def load_manifest(manifest_path: str) -> Dict[str, str]:
    """Load manifest.csv mapping filename stems to source databases.

    Args:
        manifest_path: Path to ``manifest.csv`` produced by dataset builder.

    Returns:
        Dict mapping filename stem (no extension) to ``source_db`` string.
    """
    mapping: Dict[str, str] = {}
    with open(manifest_path, "r") as f:
        for row in csv.DictReader(f):
            stem = Path(row.get("filename", "")).stem
            mapping[stem] = row.get("source_db", "unknown")
    return mapping


def infer_db_from_filename(filename: str) -> str:
    """Infer source database from filename prefix convention.

    The build scripts prefix images as ``{db_prefix}_{original_name}.jpg``.

    Args:
        filename: Image filename stem.

    Returns:
        Database name or ``"unknown"``.
    """
    for prefix in KNOWN_PREFIXES:
        if filename.startswith(prefix):
            return prefix
    return "unknown"


def count_labels_per_image(labels_dir: str) -> Dict[str, int]:
    """Count ground truth annotations per image from YOLO label files.

    Args:
        labels_dir: Directory containing ``.txt`` YOLO label files.

    Returns:
        Dict mapping filename stem to annotation count.
    """
    counts: Dict[str, int] = {}
    labels_path = Path(labels_dir)
    if not labels_path.exists():
        return counts
    for lf in labels_path.glob("*.txt"):
        n = sum(1 for line in lf.read_text().splitlines() if line.strip())
        counts[lf.stem] = n
    return counts


def compute_per_database_metrics(
    gt_labels_dir: str,
    pred_labels_dir: Optional[str] = None,
    manifest_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict]:
    """Compute per-database detection statistics.

    Groups images by source database (via manifest or filename prefix) and
    computes counts of ground truth and predicted annotations.

    Args:
        gt_labels_dir: Directory with ground truth ``.txt`` label files.
        pred_labels_dir: Directory with prediction label files (optional).
        manifest_map: Filename stem to database mapping (optional).

    Returns:
        Dict mapping database name to metrics dict with keys
        ``n_images``, ``n_gt``, ``n_pred``, ``gt_per_image``,
        ``pred_per_image``, ``detection_ratio``, ``display_name``.
    """
    gt_counts = count_labels_per_image(gt_labels_dir)
    pred_counts = count_labels_per_image(pred_labels_dir) if pred_labels_dir else {}

    db_images: Dict[str, List[str]] = defaultdict(list)
    for stem in gt_counts:
        if manifest_map:
            db = manifest_map.get(stem, infer_db_from_filename(stem))
        else:
            db = infer_db_from_filename(stem)
        db_images[db].append(stem)

    db_metrics: Dict[str, Dict] = {}
    for db_name, stems in sorted(db_images.items()):
        n_images = len(stems)
        n_gt = sum(gt_counts.get(s, 0) for s in stems)
        n_pred = sum(pred_counts.get(s, 0) for s in stems)
        db_metrics[db_name] = {
            "n_images": n_images,
            "n_gt": n_gt,
            "n_pred": n_pred,
            "gt_per_image": round(n_gt / n_images, 1) if n_images else 0,
            "pred_per_image": round(n_pred / n_images, 1) if n_images else 0,
            "detection_ratio": round(n_pred / n_gt, 3) if n_gt else 0,
            "display_name": SHORT_NAMES.get(db_name, db_name),
        }
    return db_metrics


def run_crossdomain_evaluation(
    model_path: str,
    data_yaml: str,
    output_dir: str,
    *,
    manifest_path: Optional[str] = None,
    imgsz: int = 640,
    conf: float = 0.001,
    batch: int = 16,
    device: str = "0",
) -> Dict[str, any]:
    """Run YOLO validation and compute cross-domain metrics.

    Args:
        model_path: Path to trained ``.pt`` model.
        data_yaml: YOLO data config YAML with ``test`` or ``val`` split.
        output_dir: Where to save JSON metrics, markdown report, and plots.
        manifest_path: Optional manifest.csv for per-database grouping.
        imgsz: Inference image size.
        conf: Confidence threshold for val.
        batch: Batch size.
        device: CUDA device string.

    Returns:
        Dict with ``overall`` metrics and ``per_database`` breakdown.
    """
    from ultralytics import YOLO

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    t0 = time.time()

    results = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        device=device,
        project=str(out),
        name="yolo_val",
        exist_ok=True,
        plots=True,
        save_json=True,
        save_txt=True,
        verbose=True,
    )
    elapsed = time.time() - t0

    overall = {
        "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
        "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
        "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
    }

    # Resolve labels directory from data yaml
    data_cfg = yaml.safe_load(Path(data_yaml).read_text())
    dataset_root = Path(data_cfg.get("path", ""))
    test_split = data_cfg.get("test", data_cfg.get("val", ""))
    test_images_dir = dataset_root / test_split
    test_labels_dir = Path(str(test_images_dir).replace("/images", "/labels"))

    manifest_map = {}
    if manifest_path and Path(manifest_path).exists():
        manifest_map = load_manifest(manifest_path)

    db_metrics = {}
    val_save_dir = out / "yolo_val"
    pred_dir = val_save_dir / "labels"

    if test_labels_dir.exists():
        gt_counts = count_labels_per_image(str(test_labels_dir))
        overall["n_images"] = len(gt_counts)
        overall["n_gt"] = sum(gt_counts.values())

        db_metrics = compute_per_database_metrics(
            str(test_labels_dir),
            str(pred_dir) if pred_dir.exists() else None,
            manifest_map or None,
        )

    # Save metrics JSON
    payload = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": str(model_path),
            "data": str(data_yaml),
            "imgsz": imgsz,
            "conf": conf,
        },
        "elapsed_seconds": round(elapsed, 1),
        "overall": overall,
        "per_database": db_metrics,
    }
    (out / "crossdomain_metrics.json").write_text(json.dumps(payload, indent=2))

    # Copy YOLO val artifacts
    if val_save_dir.exists():
        for name in ("PR_curve.png", "confusion_matrix.png", "results.csv"):
            src = val_save_dir / name
            if src.exists():
                shutil.copy2(src, out / name)

    return payload
