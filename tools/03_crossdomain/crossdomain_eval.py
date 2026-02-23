#!/usr/bin/env python3
"""
Cross-Domain Evaluation CLI
============================

Evaluates a trained YOLO model on a cross-domain test set (built by
build_international_dataset.py) and generates per-database performance
breakdown using the manifest.csv file.

The manifest.csv maps each image to its source database (e.g., ND_Indiv,
MI_3Season, CL_Seba, etc.), enabling fine-grained analysis of how well
a Chilean-trained model generalizes to international data.

Usage:
    # Full evaluation with per-database breakdown
    python scripts/cli/crossdomain_eval.py \\
        --model training_results/malezainia2/ambel_seba/run1_seba640_b32_1gpu/weights/best.pt \\
        --data configs/yaml_config/data_ambel-international-v1-test.yaml \\
        --manifest /media/malezainia2/E/ProcessingData/ambel-international-v1/manifest.csv \\
        --output malezainia2/ambel_crossdomain/eval_seba_on_international

    # Quick test with custom confidence threshold
    python scripts/cli/crossdomain_eval.py \\
        --model path/to/best.pt \\
        --data path/to/data.yaml \\
        --manifest path/to/manifest.csv \\
        --output path/to/output/ \\
        --conf 0.25 --imgsz 640

    # Without manifest (overall metrics only, no per-database breakdown)
    python scripts/cli/crossdomain_eval.py \\
        --model path/to/best.pt \\
        --data path/to/data.yaml \\
        --output path/to/output/

Output:
    crossdomain_metrics.json     - Full metrics: overall + per-database
    crossdomain_report.md        - Markdown report with tables
    pr_curve.png                 - Precision-Recall curve (from YOLO val)
    confusion_matrix.png         - Confusion matrix (from YOLO val)
"""

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# CONSTANTS
# ============================================================================

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


# ============================================================================
# MANIFEST LOADING
# ============================================================================


def load_manifest(manifest_path: Path) -> Dict[str, str]:
    """
    Load manifest.csv mapping image filenames to source databases.

    The manifest is produced by build_international_dataset.py with columns:
        filename, source_db, original_path, split

    Args:
        manifest_path: Path to manifest.csv

    Returns:
        Dict mapping filename (stem, no extension) to source_db string
    """
    mapping = {}
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename", "")
            source_db = row.get("source_db", "unknown")
            # Store by stem (without extension) for matching with YOLO results
            stem = Path(filename).stem
            mapping[stem] = source_db
    return mapping


def infer_db_from_filename(filename: str) -> str:
    """
    Infer source database from filename prefix convention.

    The build script prefixes images as: {db_prefix}_{original_name}.jpg
    E.g., ND_Indiv_ragweed_001.jpg -> ND_Indiv

    Falls back to parsing known prefixes from the filename.

    Args:
        filename: Image filename (stem, no extension)

    Returns:
        Source database name or "unknown"
    """
    known_prefixes = [
        "ND_Indiv",
        "ND_Aerial",
        "MI_3Season",
        "Purdue",
        "WeedCube",
        "WeedCrop",
        "CL_Seba",
        "CL_Alberto",
        "CL_StaRosa",
    ]
    for prefix in known_prefixes:
        if filename.startswith(prefix):
            return prefix
    return "unknown"


# ============================================================================
# PER-IMAGE LABEL COUNTING
# ============================================================================


def count_labels_per_image(labels_dir: Path) -> Dict[str, int]:
    """
    Count ground truth annotations per image from YOLO label files.

    Args:
        labels_dir: Directory containing .txt YOLO label files

    Returns:
        Dict mapping filename stem to number of annotations
    """
    counts = {}
    if not labels_dir.exists():
        return counts

    for label_file in labels_dir.glob("*.txt"):
        n_lines = 0
        with open(label_file, "r") as f:
            for line in f:
                if line.strip():
                    n_lines += 1
        counts[label_file.stem] = n_lines
    return counts


# ============================================================================
# PER-DATABASE METRICS COMPUTATION
# ============================================================================


def compute_per_database_metrics(
    val_results,
    manifest_map: Dict[str, str],
    labels_dir: Path,
    predictions_dir: Optional[Path] = None,
) -> Dict[str, Dict]:
    """
    Compute per-database breakdown of evaluation metrics.

    Uses ground truth label counts and prediction label counts grouped by
    source database to compute per-database detection statistics.

    For single-class detection (AMBEL), this computes:
    - n_images: Number of images from this database
    - n_gt: Total ground truth annotations
    - n_pred: Total predictions (from YOLO prediction labels)
    - gt_per_image: Average GT annotations per image

    Args:
        val_results: YOLO validation results object
        manifest_map: Filename stem -> source_db mapping
        labels_dir: Directory with ground truth .txt label files
        predictions_dir: Directory with prediction .txt label files (from val)

    Returns:
        Dict mapping database name to metrics dict
    """
    # Group images by database
    db_images = defaultdict(list)

    gt_counts = count_labels_per_image(labels_dir)

    for stem in gt_counts:
        if manifest_map:
            db = manifest_map.get(stem, infer_db_from_filename(stem))
        else:
            db = infer_db_from_filename(stem)
        db_images[db].append(stem)

    # Count predictions per image if prediction labels are available
    pred_counts = {}
    if predictions_dir and predictions_dir.exists():
        pred_counts = count_labels_per_image(predictions_dir)

    # Compute per-database statistics
    db_metrics = {}
    for db_name, stems in sorted(db_images.items()):
        n_images = len(stems)
        n_gt = sum(gt_counts.get(s, 0) for s in stems)
        n_pred = sum(pred_counts.get(s, 0) for s in stems)
        gt_per_image = n_gt / n_images if n_images > 0 else 0

        db_metrics[db_name] = {
            "n_images": n_images,
            "n_gt": n_gt,
            "n_pred": n_pred,
            "gt_per_image": round(gt_per_image, 1),
            "pred_per_image": round(n_pred / n_images, 1) if n_images > 0 else 0,
            "detection_ratio": round(n_pred / n_gt, 3) if n_gt > 0 else 0,
            "display_name": SHORT_NAMES.get(db_name, db_name),
        }

    return db_metrics


# ============================================================================
# REPORT GENERATION
# ============================================================================


def generate_markdown_report(
    overall_metrics: Dict,
    db_metrics: Dict[str, Dict],
    args,
    elapsed_seconds: float,
) -> str:
    """
    Generate a markdown report with overall and per-database metrics.

    Args:
        overall_metrics: Overall mAP, precision, recall, etc.
        db_metrics: Per-database metrics from compute_per_database_metrics
        args: CLI arguments (for recording configuration)
        elapsed_seconds: Wall-clock time for evaluation

    Returns:
        Markdown string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Cross-Domain Evaluation Report",
        "",
        f"**Generated:** {timestamp}",
        f"**Model:** `{args.model}`",
        f"**Data config:** `{args.data}`",
        f"**Image size:** {args.imgsz}",
        f"**Confidence threshold:** {args.conf}",
        f"**Evaluation time:** {elapsed_seconds:.1f}s",
        "",
        "---",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| mAP@0.5 | {overall_metrics.get('mAP50', 0):.4f} |",
        f"| mAP@0.5:0.95 | {overall_metrics.get('mAP50-95', 0):.4f} |",
        f"| Precision | {overall_metrics.get('precision', 0):.4f} |",
        f"| Recall | {overall_metrics.get('recall', 0):.4f} |",
        f"| Total images | {overall_metrics.get('n_images', 0)} |",
        f"| Total GT annotations | {overall_metrics.get('n_gt', 0)} |",
        "",
    ]

    if db_metrics:
        lines.extend([
            "---",
            "",
            "## Per-Database Breakdown",
            "",
            "| Database | Images | GT | Pred | GT/img | Pred/img | Det. Ratio |",
            "|----------|-------:|---:|-----:|-------:|---------:|-----------:|",
        ])

        for db_name, m in sorted(db_metrics.items()):
            display = m.get("display_name", db_name)
            lines.append(
                f"| {display} | {m['n_images']} | {m['n_gt']} | {m['n_pred']} "
                f"| {m['gt_per_image']} | {m['pred_per_image']} | {m['detection_ratio']:.3f} |"
            )

        # Totals
        total_images = sum(m["n_images"] for m in db_metrics.values())
        total_gt = sum(m["n_gt"] for m in db_metrics.values())
        total_pred = sum(m["n_pred"] for m in db_metrics.values())
        total_ratio = total_pred / total_gt if total_gt > 0 else 0
        lines.append(
            f"| **TOTAL** | **{total_images}** | **{total_gt}** | **{total_pred}** "
            f"| | | **{total_ratio:.3f}** |"
        )

        lines.extend([
            "",
            "**Detection Ratio** = Predictions / Ground Truth. Values near 1.0 indicate",
            "good detection rate. Values >> 1 may indicate false positives; values << 1",
            "indicate missed detections.",
            "",
        ])

        # Interpretation
        lines.extend([
            "---",
            "",
            "## Interpretation Guide",
            "",
            "- **High det. ratio (>0.8):** Model generalizes well to this database",
            "- **Low det. ratio (<0.5):** Significant domain gap; consider fine-tuning",
            "  or adding samples from this distribution",
            "- **Very high det. ratio (>1.5):** Possible false positives; review",
            "  predictions visually",
            "",
        ])

    lines.extend([
        "---",
        "",
        f"*Report generated by `scripts/cli/crossdomain_eval.py`*",
    ])

    return "\n".join(lines)


def save_metrics_json(
    overall_metrics: Dict,
    db_metrics: Dict[str, Dict],
    args,
    elapsed_seconds: float,
    output_path: Path,
):
    """
    Save all metrics to a JSON file.

    Args:
        overall_metrics: Overall model metrics
        db_metrics: Per-database breakdown
        args: CLI arguments
        elapsed_seconds: Evaluation wall-clock time
        output_path: Path to write JSON
    """
    data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": str(args.model),
            "data": str(args.data),
            "manifest": str(args.manifest) if args.manifest else None,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "batch": args.batch,
            "device": args.device,
        },
        "elapsed_seconds": round(elapsed_seconds, 1),
        "overall": overall_metrics,
        "per_database": db_metrics,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


# ============================================================================
# COPY YOLO VAL ARTIFACTS
# ============================================================================


def copy_val_artifacts(val_save_dir: Path, output_dir: Path):
    """
    Copy useful artifacts from YOLO val output to our output directory.

    Looks for PR curve, confusion matrix, and other plots generated by
    model.val().

    Args:
        val_save_dir: YOLO's val output directory
        output_dir: Our output directory
    """
    artifacts = [
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "results.csv",
    ]

    copied = []
    for name in artifacts:
        src = val_save_dir / name
        if src.exists():
            dst = output_dir / name
            shutil.copy2(src, dst)
            copied.append(name)

    if copied:
        print(f"[+] Copied {len(copied)} artifacts: {', '.join(copied)}")
    else:
        print("[!] No YOLO val artifacts found to copy")


# ============================================================================
# MAIN EVALUATION
# ============================================================================


def run_evaluation(args):
    """
    Run cross-domain evaluation.

    Steps:
    1. Load YOLO model and run val() on the test dataset
    2. Extract overall metrics (mAP50, mAP50-95, precision, recall)
    3. Load manifest.csv and compute per-database breakdown
    4. Save JSON metrics + markdown report
    5. Copy PR curve and confusion matrix from YOLO output

    Args:
        args: Parsed command-line arguments

    Returns:
        Path to output directory
    """
    import time

    # Lazy import to allow --help without loading torch
    from ultralytics import YOLO

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"[-] Data config not found: {data_yaml}")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[-] Model not found: {model_path}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  CROSS-DOMAIN EVALUATION")
    print("=" * 70)
    print(f"  Model:     {args.model}")
    print(f"  Data:      {args.data}")
    print(f"  Manifest:  {args.manifest or '(none - prefix-based grouping)'}")
    print(f"  Output:    {output_dir}")
    print(f"  ImgSize:   {args.imgsz}")
    print(f"  Conf:      {args.conf}")
    print(f"  Batch:     {args.batch}")
    print(f"  Device:    {args.device}")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Step 1: Run YOLO validation
    # ------------------------------------------------------------------
    print("-" * 70)
    print("  Step 1: Running YOLO validation")
    print("-" * 70)

    model = YOLO(str(model_path))
    print(f"[+] Model loaded: {model_path.name}")
    print(f"[+] Model classes: {model.names}")

    start_time = time.time()

    results = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        device=args.device,
        project=str(output_dir),
        name="yolo_val",
        exist_ok=True,
        plots=True,
        save_json=True,
        save_txt=True,
        verbose=True,
    )

    elapsed = time.time() - start_time
    print(f"\n[+] Validation completed in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: Extract overall metrics
    # ------------------------------------------------------------------
    print()
    print("-" * 70)
    print("  Step 2: Extracting metrics")
    print("-" * 70)

    overall_metrics = {
        "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
        "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
        "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
    }

    print(f"  mAP@0.5:      {overall_metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {overall_metrics['mAP50-95']:.4f}")
    print(f"  Precision:     {overall_metrics['precision']:.4f}")
    print(f"  Recall:        {overall_metrics['recall']:.4f}")

    # ------------------------------------------------------------------
    # Step 3: Per-database breakdown
    # ------------------------------------------------------------------
    manifest_map = {}
    db_metrics = {}

    # Resolve labels directory from the data yaml
    import yaml

    with open(data_yaml, "r") as f:
        data_config = yaml.safe_load(f)

    dataset_root = Path(data_config.get("path", ""))
    test_split = data_config.get("test", data_config.get("val", ""))
    # Labels dir mirrors images dir: images/ -> labels/
    test_images_dir = dataset_root / test_split
    test_labels_dir = Path(str(test_images_dir).replace("/images", "/labels"))

    # Load manifest if provided
    if args.manifest:
        manifest_path = Path(args.manifest)
        if manifest_path.exists():
            manifest_map = load_manifest(manifest_path)
            print(f"\n[+] Loaded manifest: {len(manifest_map)} entries")
        else:
            print(f"[!] Manifest not found: {manifest_path}")
            print("[!] Falling back to filename prefix grouping")

    # Check for YOLO prediction labels
    val_save_dir = output_dir / "yolo_val"
    predictions_dir = val_save_dir / "labels"

    if test_labels_dir.exists():
        print()
        print("-" * 70)
        print("  Step 3: Per-database breakdown")
        print("-" * 70)

        gt_counts = count_labels_per_image(test_labels_dir)
        overall_metrics["n_images"] = len(gt_counts)
        overall_metrics["n_gt"] = sum(gt_counts.values())

        db_metrics = compute_per_database_metrics(
            results,
            manifest_map,
            test_labels_dir,
            predictions_dir if predictions_dir.exists() else None,
        )

        print(f"\n  Databases found: {len(db_metrics)}")
        for db_name, m in sorted(db_metrics.items()):
            display = m.get("display_name", db_name)
            print(
                f"    {display:20s}  {m['n_images']:>5d} imgs  "
                f"{m['n_gt']:>6d} GT  {m['n_pred']:>6d} pred  "
                f"ratio={m['detection_ratio']:.3f}"
            )
    else:
        print(f"\n[!] Labels dir not found: {test_labels_dir}")
        print("[!] Skipping per-database breakdown")

    # ------------------------------------------------------------------
    # Step 4: Save results
    # ------------------------------------------------------------------
    print()
    print("-" * 70)
    print("  Step 4: Saving results")
    print("-" * 70)

    # JSON metrics
    json_path = output_dir / "crossdomain_metrics.json"
    save_metrics_json(overall_metrics, db_metrics, args, elapsed, json_path)
    print(f"[+] Metrics saved: {json_path}")

    # Markdown report
    md_path = output_dir / "crossdomain_report.md"
    report = generate_markdown_report(overall_metrics, db_metrics, args, elapsed)
    with open(md_path, "w") as f:
        f.write(report)
    print(f"[+] Report saved: {md_path}")

    # Copy YOLO val artifacts (PR curve, confusion matrix, etc.)
    if val_save_dir.exists():
        copy_val_artifacts(val_save_dir, output_dir)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Output directory: {output_dir}")
    print(f"  mAP@0.5:         {overall_metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95:    {overall_metrics['mAP50-95']:.4f}")
    if db_metrics:
        print(f"  Databases:        {len(db_metrics)}")
    print("=" * 70)

    return output_dir


# ============================================================================
# CLI ARGUMENT PARSER
# ============================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cross-domain evaluation: YOLO model on international ragweed test set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Chilean model on international test set
  python scripts/cli/crossdomain_eval.py \\
      --model training_results/malezainia2/ambel_seba/run1_seba640_b32_1gpu/weights/best.pt \\
      --data configs/yaml_config/data_ambel-international-v1-test.yaml \\
      --manifest /media/malezainia2/E/ProcessingData/ambel-international-v1/manifest.csv \\
      --output malezainia2/ambel_crossdomain/eval_seba_on_international

  # Without manifest (uses filename prefix grouping)
  python scripts/cli/crossdomain_eval.py \\
      --model path/to/best.pt \\
      --data path/to/data.yaml \\
      --output path/to/output/

  # Custom settings
  python scripts/cli/crossdomain_eval.py \\
      --model path/to/best.pt \\
      --data path/to/data.yaml \\
      --manifest path/to/manifest.csv \\
      --output path/to/output/ \\
      --conf 0.25 --imgsz 1024 --device cuda:1
        """,
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained YOLO model (.pt file)",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to YOLO data config YAML (with test split)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for metrics, report, and plots",
    )

    # Optional arguments
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest.csv mapping images to source databases (default: infer from filename prefix)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold for val (default: 0.001, standard for mAP computation)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help='Device: 0, 1, "0,1", or "cpu" (default: 0)',
    )

    return parser.parse_args()


def main():
    """Entry point for CLI execution."""
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
