"""
Active Learning — Uncertainty Mining
======================================

Score detections by uncertainty (confidence near threshold), rank images
for human review, and export a review set with thumbnails and YOLO labels.

Functions
---------
score_images
    Compute composite uncertainty scores for all images.
export_uncertain_panels
    Generate annotated panels for top-uncertain images.
export_yolo_labels
    Export top-N images + YOLO labels for re-annotation.
generate_uncertainty_report
    Generate a JSON summary report of the mining session.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def _score_low_confidence(
    detections: List[Dict],
    lo: float = 0.25,
    hi: float = 0.50,
) -> float:
    """Fraction of detections in the uncertain confidence zone [lo, hi]."""
    if not detections:
        return 0.0
    n_uncertain = sum(1 for d in detections if lo <= d["confidence"] <= hi)
    return n_uncertain / len(detections)


def _score_confidence_entropy(detections: List[Dict]) -> float:
    """Entropy of the confidence distribution (higher = more uncertain)."""
    if not detections:
        return 0.0
    confs = np.array([d["confidence"] for d in detections])
    confs = np.clip(confs, 1e-9, 1 - 1e-9)
    entropy = -np.mean(confs * np.log(confs) + (1 - confs) * np.log(1 - confs))
    return float(entropy)


def _score_borderline_density(
    n_detections: int,
    median_count: float,
    mad: float,
) -> float:
    """How far the detection count deviates from the median (MAD-scaled)."""
    if mad < 1:
        return 0.0
    z = abs(n_detections - median_count) / mad
    return float(min(z / 3.0, 1.0))


def _score_class_ambiguity(
    detections: List[Dict],
    ambiguous_classes: List[str],
    threshold: float = 0.50,
) -> float:
    """Fraction of uncertain detections that are ambiguous classes."""
    if not detections or not ambiguous_classes:
        return 0.0
    uncertain_amb = sum(
        1 for d in detections
        if d["confidence"] <= threshold and d.get("class_name") in ambiguous_classes
    )
    return uncertain_amb / len(detections)


def _compute_dataset_stats(images_data: Dict) -> Tuple[float, float]:
    """Compute median and MAD of detection counts across all images."""
    counts = [len(img.get("detections", [])) for img in images_data.values()]
    if not counts:
        return 0.0, 1.0
    median = float(np.median(counts))
    mad = float(np.median(np.abs(np.array(counts) - median)))
    return median, max(mad, 1.0)


def score_images(
    images_data: Dict,
    *,
    weights: Optional[Dict[str, float]] = None,
    conf_lo: float = 0.25,
    conf_hi: float = 0.50,
    ambiguous_classes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Score all images by composite uncertainty.

    Args:
        images_data: Dict of ``image_name -> {detections, width, height}``.
            Each detection has ``confidence``, ``class_name``, ``bbox``.
        weights: Scoring weights per criterion. Defaults to equal weights.
        conf_lo: Lower confidence threshold for uncertain zone.
        conf_hi: Upper boundary of uncertain zone.
        ambiguous_classes: Classes prone to visual confusion.

    Returns:
        DataFrame sorted by ``composite_score`` descending, with columns
        ``image``, ``n_detections``, ``composite_score``, and per-criterion
        scores.
    """
    if weights is None:
        weights = {
            "low_confidence": 0.35,
            "confidence_entropy": 0.20,
            "borderline_density": 0.20,
            "class_ambiguity": 0.25,
        }
    if ambiguous_classes is None:
        ambiguous_classes = []

    median_count, mad = _compute_dataset_stats(images_data)
    rows: List[Dict] = []

    for img_name, img_data in images_data.items():
        dets = img_data.get("detections", [])
        n = len(dets)

        scores = {
            "low_confidence": _score_low_confidence(dets, conf_lo, conf_hi),
            "confidence_entropy": _score_confidence_entropy(dets),
            "borderline_density": _score_borderline_density(n, median_count, mad),
            "class_ambiguity": _score_class_ambiguity(dets, ambiguous_classes, conf_hi),
        }

        composite = sum(scores.get(k, 0) * weights.get(k, 0) for k in weights)

        row = {"image": img_name, "n_detections": n, "composite_score": composite}
        row.update(scores)

        if dets:
            confs = [d["confidence"] for d in dets]
            row["mean_confidence"] = float(np.mean(confs))
            row["min_confidence"] = float(np.min(confs))
        else:
            row["mean_confidence"] = None
            row["min_confidence"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


def export_uncertain_panels(
    ranking: pd.DataFrame,
    images_data: Dict,
    images_dir: str,
    output_dir: str,
    *,
    top_n: int = 50,
    conf_hi: float = 0.50,
) -> int:
    """Generate annotated panels for top-uncertain images.

    Draws bounding boxes color-coded by confidence: red (low),
    orange (medium), green (high).

    Args:
        ranking: DataFrame from :func:`score_images`.
        images_data: Detection data dict.
        images_dir: Directory containing original images.
        output_dir: Output directory for panels.
        top_n: Number of top images to export.
        conf_hi: Confidence threshold separating low/medium.

    Returns:
        Number of panels generated.
    """
    panels_dir = Path(output_dir) / "uncertain_panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(images_dir)

    generated = 0
    for _, row in ranking.head(top_n).iterrows():
        img_name = row["image"]
        img_path = img_dir / img_name
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        dets = images_data.get(img_name, {}).get("detections", [])
        overlay = image.copy()

        for det in dets:
            conf = det["confidence"]
            if conf < conf_hi:
                color = (0, 0, 255)  # red BGR
            elif conf < 0.70:
                color = (0, 165, 255)  # orange BGR
            else:
                color = (0, 200, 0)  # green BGR

            x1, y1, x2, y2 = (int(c) for c in det["bbox"])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label = f"{det.get('class_name', '')} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        image = cv2.addWeighted(overlay, 0.2, image, 0.8, 0)

        rank = int(row["rank"])
        score = row["composite_score"]
        title = f"#{rank} | {img_name} | score={score:.3f} | dets={len(dets)}"
        cv2.rectangle(image, (0, 0), (len(title) * 11 + 20, 30), (0, 0, 0), -1)
        cv2.putText(image, title, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imwrite(str(panels_dir / f"rank{rank:03d}_{img_name}"), image)
        generated += 1

    return generated


def export_yolo_labels(
    ranking: pd.DataFrame,
    images_data: Dict,
    images_dir: str,
    output_dir: str,
    *,
    top_n: int = 50,
) -> int:
    """Export top-N uncertain images + YOLO labels for re-annotation.

    Args:
        ranking: DataFrame from :func:`score_images`.
        images_data: Detection data dict.
        images_dir: Directory containing original images.
        output_dir: Output directory.
        top_n: Number of top images to export.

    Returns:
        Number of images exported.
    """
    export_dir = Path(output_dir) / "uncertain_yolo_export"
    images_out = export_dir / "images"
    labels_out = export_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    img_dir = Path(images_dir)

    exported = 0
    for _, row in ranking.head(top_n).iterrows():
        img_name = row["image"]
        img_path = img_dir / img_name
        if not img_path.exists():
            continue

        img_data = images_data.get(img_name, {})
        dets = img_data.get("detections", [])
        w = img_data.get("width", 1)
        h = img_data.get("height", 1)

        shutil.copy2(str(img_path), str(images_out / img_name))

        stem = Path(img_name).stem
        with open(labels_out / f"{stem}.txt", "w") as f:
            for det in dets:
                x1, y1, x2, y2 = det["bbox"]
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{det['class_id']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        exported += 1

    return exported


def generate_uncertainty_report(
    ranking: pd.DataFrame,
    top_n: int,
    output_dir: str,
    metadata: Optional[Dict] = None,
) -> Dict:
    """Generate a JSON summary report of the mining session.

    Args:
        ranking: DataFrame from :func:`score_images`.
        top_n: Number of top images selected.
        output_dir: Where to write ``uncertainty_report.json``.
        metadata: Optional metadata dict (model path, source, etc.).

    Returns:
        Report dict.
    """
    top = ranking.head(top_n)
    report = {
        "timestamp": datetime.now().isoformat(),
        "source": (metadata or {}).get("source", "unknown"),
        "model": (metadata or {}).get("model", "unknown"),
        "total_images": len(ranking),
        "top_n": top_n,
        "score_distribution": {
            "mean": float(ranking["composite_score"].mean()),
            "std": float(ranking["composite_score"].std()),
            "median": float(ranking["composite_score"].median()),
            "max": float(ranking["composite_score"].max()),
            "min": float(ranking["composite_score"].min()),
        },
        "top_n_stats": {
            "mean_score": float(top["composite_score"].mean()),
            "mean_detections": float(top["n_detections"].mean()),
        },
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "uncertainty_report.json").write_text(json.dumps(report, indent=2))

    return report
