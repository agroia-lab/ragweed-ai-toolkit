#!/usr/bin/env python3
"""
Cross-Domain Comparison CLI
=============================

Loads evaluation JSON files from multiple cross-domain evaluations
(produced by crossdomain_eval.py) and generates a comparison table
and grouped bar chart.

Usage:
    # Compare multiple evaluations
    python tools/03_crossdomain/crossdomain_compare.py \\
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
    python tools/03_crossdomain/crossdomain_compare.py \\
        --evals eval1/crossdomain_metrics.json eval2/crossdomain_metrics.json \\
        --output comparison/ \\
        --baseline-map50 0.90

    # Auto-labels from filenames
    python tools/03_crossdomain/crossdomain_compare.py \\
        --evals eval1/crossdomain_metrics.json eval2/crossdomain_metrics.json \\
        --output comparison/

Output:
    comparison_report.md     - Markdown comparison table
    comparison_chart.png     - Grouped bar chart (mAP50, mAP50-95, P, R)
    comparison_data.json     - Combined data from all evaluations
"""

import argparse
import sys
from pathlib import Path

from ragweed_toolkit.crossdomain import (
    compare_evaluations,
    extract_overall_metrics,
    load_eval_results,
)


# ============================================================================
# MAIN
# ============================================================================


def run_comparison(args):
    """
    Load evaluation JSONs, generate comparison outputs via library.

    Args:
        args: Parsed CLI arguments.
    """
    output_dir = Path(args.output)
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
    # Step 1: Load and display evaluation data
    # ------------------------------------------------------------------
    print("-" * 70)
    print("  Step 1: Loading evaluation data")
    print("-" * 70)

    for label, path in zip(labels, eval_paths):
        data = load_eval_results(str(path))
        metrics = extract_overall_metrics(data)
        print(
            f"[+] {label:30s}  mAP50={metrics['mAP50']:.3f}  "
            f"mAP50-95={metrics['mAP50-95']:.3f}  "
            f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}"
        )

    # ------------------------------------------------------------------
    # Step 2: Generate comparison via library
    # ------------------------------------------------------------------
    print()
    print("-" * 70)
    print("  Step 2: Generating comparison outputs")
    print("-" * 70)

    combined = compare_evaluations(
        eval_paths=[str(p) for p in eval_paths],
        output_dir=str(output_dir),
        labels=labels,
        baseline_map50=args.baseline_map50,
    )

    md_path = output_dir / "comparison_report.md"
    chart_path = output_dir / "comparison_chart.png"
    json_path = output_dir / "comparison_data.json"

    print(f"[+] Report saved: {md_path}")
    print(f"[+] Chart saved: {chart_path}")
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
  python tools/03_crossdomain/crossdomain_compare.py \\
      --evals \\
          malezainia2/ambel_crossdomain/phase1_eval/crossdomain_metrics.json \\
          malezainia2/ambel_crossdomain/phase2_eval_intl/crossdomain_metrics.json \\
      --labels "CL Model -> INTL" "Combined -> INTL" \\
      --output malezainia2/ambel_crossdomain/comparison

  # Auto-label from directory names
  python tools/03_crossdomain/crossdomain_compare.py \\
      --evals eval1/crossdomain_metrics.json eval2/crossdomain_metrics.json \\
      --output comparison/

  # Custom baseline
  python tools/03_crossdomain/crossdomain_compare.py \\
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
