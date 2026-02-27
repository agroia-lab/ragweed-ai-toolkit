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
    python tools/03_crossdomain/crossdomain_eval.py \\
        --model training_results/malezainia2/ambel_seba/run1_seba640_b32_1gpu/weights/best.pt \\
        --data configs/yaml_config/data_ambel-international-v1-test.yaml \\
        --manifest /media/malezainia2/E/ProcessingData/ambel-international-v1/manifest.csv \\
        --output malezainia2/ambel_crossdomain/eval_seba_on_international

    # Quick test with custom confidence threshold
    python tools/03_crossdomain/crossdomain_eval.py \\
        --model path/to/best.pt \\
        --data path/to/data.yaml \\
        --manifest path/to/manifest.csv \\
        --output path/to/output/ \\
        --conf 0.25 --imgsz 640

    # Without manifest (overall metrics only, no per-database breakdown)
    python tools/03_crossdomain/crossdomain_eval.py \\
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
import sys
from datetime import datetime
from pathlib import Path

from ragweed_toolkit.crossdomain import run_crossdomain_evaluation

# ============================================================================
# REPORT GENERATION (CLI-specific formatting)
# ============================================================================


def generate_markdown_report(
    overall_metrics, db_metrics, args, elapsed_seconds,
):
    """
    Generate a markdown report with overall and per-database metrics.

    Args:
        overall_metrics: Overall mAP, precision, recall, etc.
        db_metrics: Per-database metrics dict.
        args: CLI arguments (for recording configuration).
        elapsed_seconds: Wall-clock time for evaluation.

    Returns:
        Markdown string.
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
        "*Report generated by `tools/03_crossdomain/crossdomain_eval.py`*",
    ])

    return "\n".join(lines)


# ============================================================================
# MAIN EVALUATION
# ============================================================================


def run_evaluation(args):
    """
    Run cross-domain evaluation.

    Delegates core work to ragweed_toolkit.crossdomain.run_crossdomain_evaluation,
    then generates a custom markdown report with the results.

    Args:
        args: Parsed command-line arguments.
    """
    output_dir = Path(args.output)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[-] Model not found: {model_path}")
        sys.exit(1)

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"[-] Data config not found: {data_yaml}")
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
    # Run evaluation via library
    # ------------------------------------------------------------------
    print("-" * 70)
    print("  Running YOLO validation + per-database breakdown")
    print("-" * 70)

    payload = run_crossdomain_evaluation(
        model_path=str(args.model),
        data_yaml=str(args.data),
        output_dir=str(output_dir),
        manifest_path=str(args.manifest) if args.manifest else None,
        imgsz=args.imgsz,
        conf=args.conf,
        batch=args.batch,
        device=args.device,
    )

    overall_metrics = payload.get("overall", {})
    db_metrics = payload.get("per_database", {})
    elapsed = payload.get("elapsed_seconds", 0)

    # ------------------------------------------------------------------
    # Print metrics
    # ------------------------------------------------------------------
    print(f"\n  mAP@0.5:      {overall_metrics.get('mAP50', 0):.4f}")
    print(f"  mAP@0.5:0.95: {overall_metrics.get('mAP50-95', 0):.4f}")
    print(f"  Precision:     {overall_metrics.get('precision', 0):.4f}")
    print(f"  Recall:        {overall_metrics.get('recall', 0):.4f}")

    if db_metrics:
        print(f"\n  Databases found: {len(db_metrics)}")
        for db_name, m in sorted(db_metrics.items()):
            display = m.get("display_name", db_name)
            print(
                f"    {display:20s}  {m['n_images']:>5d} imgs  "
                f"{m['n_gt']:>6d} GT  {m['n_pred']:>6d} pred  "
                f"ratio={m['detection_ratio']:.3f}"
            )

    # ------------------------------------------------------------------
    # Generate markdown report (CLI-specific)
    # ------------------------------------------------------------------
    md_path = output_dir / "crossdomain_report.md"
    report = generate_markdown_report(overall_metrics, db_metrics, args, elapsed)
    md_path.write_text(report)
    print(f"\n[+] Report saved: {md_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Output directory: {output_dir}")
    print(f"  mAP@0.5:         {overall_metrics.get('mAP50', 0):.4f}")
    print(f"  mAP@0.5:0.95:    {overall_metrics.get('mAP50-95', 0):.4f}")
    if db_metrics:
        print(f"  Databases:        {len(db_metrics)}")
    print("=" * 70)


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
  python tools/03_crossdomain/crossdomain_eval.py \\
      --model training_results/malezainia2/ambel_seba/run1_seba640_b32_1gpu/weights/best.pt \\
      --data configs/yaml_config/data_ambel-international-v1-test.yaml \\
      --manifest /media/malezainia2/E/ProcessingData/ambel-international-v1/manifest.csv \\
      --output malezainia2/ambel_crossdomain/eval_seba_on_international

  # Without manifest (uses filename prefix grouping)
  python tools/03_crossdomain/crossdomain_eval.py \\
      --model path/to/best.pt \\
      --data path/to/data.yaml \\
      --output path/to/output/

  # Custom settings
  python tools/03_crossdomain/crossdomain_eval.py \\
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
