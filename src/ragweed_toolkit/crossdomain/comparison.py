"""
Cross-Domain Comparison
========================

Load evaluation results from multiple cross-domain runs, generate a
comparison table and grouped bar chart.

Functions
---------
load_eval_results
    Load a ``crossdomain_metrics.json`` file.
extract_overall_metrics
    Extract mAP50, mAP50-95, precision, recall from an eval dict.
compare_evaluations
    Generate comparison table, bar chart, and combined JSON.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Metric keys and display info
METRICS_KEYS = ["mAP50", "mAP50-95", "precision", "recall"]
METRICS_LABELS = ["mAP50", "mAP50-95", "Precision", "Recall"]
METRICS_COLORS = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c"]


def load_eval_results(json_path: str) -> Dict:
    """Load a ``crossdomain_metrics.json`` file.

    Args:
        json_path: Path to evaluation JSON.

    Returns:
        Parsed JSON dict.
    """
    return json.loads(Path(json_path).read_text())


def extract_overall_metrics(data: Dict) -> Dict[str, float]:
    """Extract the four main metrics from an evaluation dict.

    Args:
        data: Parsed JSON from ``crossdomain_metrics.json``.

    Returns:
        Dict with ``mAP50``, ``mAP50-95``, ``precision``, ``recall``.
    """
    overall = data.get("overall", {})
    return {k: float(overall.get(k, 0)) for k in METRICS_KEYS}


def _generate_comparison_table(
    labels: List[str],
    metrics_list: List[Dict[str, float]],
) -> str:
    """Generate a markdown comparison table."""
    lines = [
        "| Evaluation | mAP50 | mAP50-95 | Precision | Recall |",
        "|------------|------:|---------:|----------:|-------:|",
    ]
    for label, m in zip(labels, metrics_list):
        lines.append(
            f"| {label} "
            f"| {m['mAP50']:.3f} "
            f"| {m['mAP50-95']:.3f} "
            f"| {m['precision']:.3f} "
            f"| {m['recall']:.3f} |"
        )
    return "\n".join(lines)


def _generate_bar_chart(
    labels: List[str],
    metrics_list: List[Dict[str, float]],
    output_path: Path,
    baseline_map50: float,
) -> None:
    """Generate a grouped bar chart comparing evaluations."""
    n_evals = len(labels)
    n_metrics = len(METRICS_KEYS)
    x = np.arange(n_evals)
    bar_width = 0.18

    fig, ax = plt.subplots(figsize=(max(8, n_evals * 2.5), 6))

    for i, (key, color, mlabel) in enumerate(
        zip(METRICS_KEYS, METRICS_COLORS, METRICS_LABELS)
    ):
        values = [m[key] for m in metrics_list]
        offset = (i - (n_metrics - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, values, bar_width, label=mlabel, color=color, alpha=0.85
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=45,
            )

    ax.axhline(
        y=baseline_map50, color="gray", linestyle="--", linewidth=1.5,
        alpha=0.7, label=f"Baseline ({baseline_map50:.3f})",
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
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()


def compare_evaluations(
    eval_paths: List[str],
    output_dir: str,
    *,
    labels: Optional[List[str]] = None,
    baseline_map50: float = 0.886,
) -> Dict:
    """Compare multiple cross-domain evaluation results.

    Generates a markdown report, grouped bar chart, and combined JSON.

    Args:
        eval_paths: Paths to ``crossdomain_metrics.json`` files.
        output_dir: Destination for comparison outputs.
        labels: Display labels (one per eval). Defaults to parent dir names.
        baseline_map50: Reference mAP50 for the baseline model.

    Returns:
        Combined comparison dict.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in eval_paths]

    if labels is None:
        labels = [p.parent.name for p in paths]

    all_data = [load_eval_results(str(p)) for p in paths]
    metrics_list = [extract_overall_metrics(d) for d in all_data]

    # Markdown report
    table = _generate_comparison_table(labels, metrics_list)
    report_lines = [
        "# Cross-Domain Comparison Report",
        "",
        f"**Generated:** {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"**Evaluations:** {len(labels)}",
        f"**Baseline mAP50:** {baseline_map50:.3f}",
        "",
        "## Comparison Table",
        "",
        table,
        "",
        "## Source Files",
        "",
    ]
    for lbl, p in zip(labels, paths):
        report_lines.append(f"- **{lbl}:** `{p}`")
    (out / "comparison_report.md").write_text("\n".join(report_lines))

    # Bar chart
    _generate_bar_chart(labels, metrics_list, out / "comparison_chart.png", baseline_map50)

    # Combined JSON
    combined = {
        "timestamp": datetime.now().isoformat(),
        "baseline_map50": baseline_map50,
        "evaluations": [
            {
                "label": lbl,
                "source_file": str(p),
                "metrics": m,
                "config": d.get("config", {}),
            }
            for lbl, p, d, m in zip(labels, paths, all_data, metrics_list)
        ],
    }
    (out / "comparison_data.json").write_text(json.dumps(combined, indent=2))

    return combined
