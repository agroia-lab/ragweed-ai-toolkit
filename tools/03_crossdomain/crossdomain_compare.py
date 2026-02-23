#!/usr/bin/env python3
"""
Cross-Domain Comparison CLI
=============================

Loads evaluation JSON files from multiple cross-domain evaluations
(produced by crossdomain_eval.py) and generates a comparison table
and grouped bar chart.

Usage:
    # Compare multiple evaluations
    python scripts/cli/crossdomain_compare.py \\
        --evals \\
            malezainia2/ambel_crossdomain/phase1_eval/crossdomain_metrics.json \\
            malezainia2/ambel_crossdomain/phase2_eval_intl/crossdomain_metrics.json \\
            malezainia2/ambel_crossdomain/phase2_eval_cl/crossdomain_metrics.json \\
        --labels \\
            "CL Model -> INTL" \\
            "Combined -> INTL" \\
            "Combined -> CL" \\
        --output malezainia2/ambel_crossdomain/comparison

    # With custom baseline reference line
    python scripts/cli/crossdomain_compare.py \\
        --evals eval1/crossdomain_metrics.json eval2/crossdomain_metrics.json \\
        --output comparison/ \\
        --baseline-map50 0.90

    # Auto-labels from filenames
    python scripts/cli/crossdomain_compare.py \\
        --evals eval1/crossdomain_metrics.json eval2/crossdomain_metrics.json \\
        --output comparison/

Output:
    comparison_report.md     - Markdown comparison table
    comparison_chart.png     - Grouped bar chart (mAP50, mAP50-95, P, R)
    comparison_data.json     - Combined data from all evaluations
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# CONSTANTS
# ============================================================================

METRICS_KEYS = ["mAP50", "mAP50-95", "precision", "recall"]
METRICS_LABELS = ["mAP50", "mAP50-95", "Precision", "Recall"]
METRICS_COLORS = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c"]  # blue, orange, green, red


# ============================================================================
# DATA LOADING
# ============================================================================


def load_eval_json(path: Path) -> Dict:
    """
    Load a crossdomain_metrics.json file.

    Args:
        path: Path to JSON file produced by crossdomain_eval.py

    Returns:
        Parsed JSON dict with 'overall', 'config', etc.
    """
    with open(path, "r") as f:
        return json.load(f)


def extract_overall_metrics(data: Dict) -> Dict[str, float]:
    """
    Extract the four main metrics from an evaluation JSON.

    Args:
        data: Parsed JSON from crossdomain_metrics.json

    Returns:
        Dict with mAP50, mAP50-95, precision, recall as floats
    """
    overall = data.get("overall", {})
    return {
        "mAP50": float(overall.get("mAP50", 0)),
        "mAP50-95": float(overall.get("mAP50-95", 0)),
        "precision": float(overall.get("precision", 0)),
        "recall": float(overall.get("recall", 0)),
    }


# ============================================================================
# COMPARISON TABLE (MARKDOWN)
# ============================================================================


def generate_comparison_table(
    labels: List[str],
    metrics_list: List[Dict[str, float]],
) -> str:
    """
    Generate a markdown comparison table.

    Args:
        labels: Display labels for each evaluation
        metrics_list: List of metric dicts (one per evaluation)

    Returns:
        Markdown string with the comparison table
    """
    lines = [
        "| Evaluation | mAP50 | mAP50-95 | Precision | Recall |",
        "|------------|------:|---------:|----------:|-------:|",
    ]

    for label, metrics in zip(labels, metrics_list):
        lines.append(
            f"| {label} "
            f"| {metrics['mAP50']:.3f} "
            f"| {metrics['mAP50-95']:.3f} "
            f"| {metrics['precision']:.3f} "
            f"| {metrics['recall']:.3f} |"
        )

    return "\n".join(lines)


def generate_markdown_report(
    labels: List[str],
    metrics_list: List[Dict[str, float]],
    eval_paths: List[Path],
    baseline_map50: float,
) -> str:
    """
    Generate a full markdown comparison report.

    Args:
        labels: Display labels for each evaluation
        metrics_list: List of metric dicts
        eval_paths: Original paths to JSON files
        baseline_map50: Baseline mAP50 for reference

    Returns:
        Full markdown report string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Cross-Domain Comparison Report",
        "",
        f"**Generated:** {timestamp}",
        f"**Evaluations compared:** {len(labels)}",
        f"**Baseline mAP50 (CL):** {baseline_map50:.3f}",
        "",
        "---",
        "",
        "## Comparison Table",
        "",
        generate_comparison_table(labels, metrics_list),
        "",
        "---",
        "",
        "## Source Files",
        "",
    ]

    for label, path in zip(labels, eval_paths):
        lines.append(f"- **{label}:** `{path}`")

    lines.extend([
        "",
        "---",
        "",
        "## Interpretation",
        "",
        f"- **Baseline reference:** CL model on CL test set = {baseline_map50:.3f} mAP50",
        "- Values close to baseline indicate good cross-domain generalization",
        "- Large drops from baseline indicate significant domain gap",
        "- Improvements after fine-tuning on combined data validate the training strategy",
        "",
        "---",
        "",
        "*Report generated by `scripts/cli/crossdomain_compare.py`*",
    ])

    return "\n".join(lines)


# ============================================================================
# BAR CHART
# ============================================================================


def generate_bar_chart(
    labels: List[str],
    metrics_list: List[Dict[str, float]],
    output_path: Path,
    baseline_map50: float,
):
    """
    Generate a grouped bar chart comparing evaluations.

    Creates one group per evaluation with 4 bars each (mAP50, mAP50-95,
    Precision, Recall). Adds a horizontal dashed line at the baseline mAP50.

    Args:
        labels: Display labels for each evaluation
        metrics_list: List of metric dicts
        output_path: Path to save PNG
        baseline_map50: Baseline mAP50 for reference line
    """
    n_evals = len(labels)
    n_metrics = len(METRICS_KEYS)
    x = np.arange(n_evals)
    bar_width = 0.18

    fig, ax = plt.subplots(figsize=(max(8, n_evals * 2.5), 6))

    for i, (key, color, label) in enumerate(
        zip(METRICS_KEYS, METRICS_COLORS, METRICS_LABELS)
    ):
        values = [m[key] for m in metrics_list]
        offset = (i - (n_metrics - 1) / 2) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=label, color=color, alpha=0.85)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45,
            )

    # Baseline reference line
    ax.axhline(
        y=baseline_map50,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"CL Baseline ({baseline_map50:.3f})",
    )

    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Score")
    ax.set_title("Cross-Domain Evaluation Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================


def run_comparison(args):
    """
    Load evaluation JSONs, generate comparison outputs.

    Args:
        args: Parsed CLI arguments
    """
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_paths = [Path(p) for p in args.evals]

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    for p in eval_paths:
        if not p.exists():
            print(f"[-] Evaluation file not found: {p}")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    if args.labels:
        if len(args.labels) != len(eval_paths):
            print(
                f"[-] Number of labels ({len(args.labels)}) must match "
                f"number of eval files ({len(eval_paths)})"
            )
            sys.exit(1)
        labels = args.labels
    else:
        # Auto-generate labels from parent directory names
        labels = [p.parent.name for p in eval_paths]

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  CROSS-DOMAIN COMPARISON")
    print("=" * 70)
    print(f"  Evaluations:  {len(eval_paths)}")
    print(f"  Output:       {output_dir}")
    print(f"  Baseline:     {args.baseline_map50:.3f} mAP50")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Step 1: Load evaluation data
    # ------------------------------------------------------------------
    print("-" * 70)
    print("  Step 1: Loading evaluation data")
    print("-" * 70)

    all_data = []
    metrics_list = []
    for label, path in zip(labels, eval_paths):
        data = load_eval_json(path)
        all_data.append(data)
        metrics = extract_overall_metrics(data)
        metrics_list.append(metrics)
        print(
            f"[+] {label:30s}  mAP50={metrics['mAP50']:.3f}  "
            f"mAP50-95={metrics['mAP50-95']:.3f}  "
            f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}"
        )

    # ------------------------------------------------------------------
    # Step 2: Print comparison table
    # ------------------------------------------------------------------
    print()
    print("-" * 70)
    print("  Step 2: Comparison table")
    print("-" * 70)
    print()
    table = generate_comparison_table(labels, metrics_list)
    print(table)

    # ------------------------------------------------------------------
    # Step 3: Save outputs
    # ------------------------------------------------------------------
    print()
    print("-" * 70)
    print("  Step 3: Saving outputs")
    print("-" * 70)

    # Markdown report
    md_path = output_dir / "comparison_report.md"
    report = generate_markdown_report(
        labels, metrics_list, eval_paths, args.baseline_map50
    )
    with open(md_path, "w") as f:
        f.write(report)
    print(f"[+] Report saved: {md_path}")

    # Bar chart
    chart_path = output_dir / "comparison_chart.png"
    generate_bar_chart(labels, metrics_list, chart_path, args.baseline_map50)
    print(f"[+] Chart saved: {chart_path}")

    # Combined JSON
    json_path = output_dir / "comparison_data.json"
    combined = {
        "timestamp": datetime.now().isoformat(),
        "baseline_map50": args.baseline_map50,
        "evaluations": [],
    }
    for label, path, data, metrics in zip(labels, eval_paths, all_data, metrics_list):
        combined["evaluations"].append({
            "label": label,
            "source_file": str(path),
            "metrics": metrics,
            "config": data.get("config", {}),
        })
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"[+] Combined data saved: {json_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  COMPARISON COMPLETE")
    print("=" * 70)
    print(f"  Output directory: {output_dir}")
    print(f"  Files generated:")
    print(f"    - {md_path.name}")
    print(f"    - {chart_path.name}")
    print(f"    - {json_path.name}")
    print("=" * 70)


# ============================================================================
# CLI ARGUMENT PARSER
# ============================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare multiple cross-domain evaluation results side by side.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare evaluations with custom labels
  python scripts/cli/crossdomain_compare.py \\
      --evals \\
          malezainia2/ambel_crossdomain/phase1_eval/crossdomain_metrics.json \\
          malezainia2/ambel_crossdomain/phase2_eval_intl/crossdomain_metrics.json \\
      --labels "CL Model -> INTL" "Combined -> INTL" \\
      --output malezainia2/ambel_crossdomain/comparison

  # Auto-label from directory names
  python scripts/cli/crossdomain_compare.py \\
      --evals eval1/crossdomain_metrics.json eval2/crossdomain_metrics.json \\
      --output comparison/

  # Custom baseline
  python scripts/cli/crossdomain_compare.py \\
      --evals eval1/crossdomain_metrics.json eval2/crossdomain_metrics.json \\
      --output comparison/ --baseline-map50 0.90
        """,
    )

    parser.add_argument(
        "--evals",
        type=str,
        nargs="+",
        required=True,
        help="Paths to crossdomain_metrics.json files to compare",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Display labels for each evaluation (default: infer from directory names)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for comparison report, chart, and data",
    )
    parser.add_argument(
        "--baseline-map50",
        type=float,
        default=0.886,
        help="Chilean model baseline mAP50 for reference line on chart (default: 0.886)",
    )

    return parser.parse_args()


def main():
    """Entry point for CLI execution."""
    args = parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()
