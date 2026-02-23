#!/usr/bin/env python3
"""
Uncertainty Mining for Active Learning

Identifies images where the YOLO model is most uncertain, so they can be
prioritized for human review and re-annotation.

Three input modes (mutually exclusive):
  --detections  : Read pre-saved detections.json (fastest)
  --images      : Run SAHI inference on new images, then score
  --sahi-output : Score from an existing SAHI output directory

Outputs:
  uncertain_ranking.csv       - All images ranked by composite uncertainty score
  uncertain_panels/           - Top-N images with confidence-coded boxes
  uncertain_yolo_export/      - Top-N images + YOLO labels for re-annotation
  uncertainty_report.json     - Summary statistics

Usage:
    # From pre-saved detections (fastest)
    python scripts/cli/mine_uncertain_samples.py \\
        --detections outputs/sahi_inference_1024px_*/detections.json \\
        --top 50 --output outputs/active_learning/run1

    # From new images (runs inference internally)
    python scripts/cli/mine_uncertain_samples.py \\
        --images /path/to/new/campo/images \\
        --model training_results/v4_4140tiles/run_20260118_132318/weights/best.pt \\
        --top 30 --output outputs/active_learning/run2

    # From existing SAHI output (re-runs inference to get confidences)
    python scripts/cli/mine_uncertain_samples.py \\
        --sahi-output outputs/sahi_inference_1024px_20260210 \\
        --top 20 --output outputs/active_learning/run3

    # With CLIP distribution shift scoring
    python scripts/cli/mine_uncertain_samples.py \\
        --detections outputs/test/detections.json \\
        --reference-data configs/yaml_config/data_tomate-orobanche-v4-4488tiles.yaml \\
        --top 50 --output outputs/active_learning/run4
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.active_learning.uncertainty_scorer import (
    compute_composite_score,
    compute_dataset_stats,
    score_borderline_density,
    score_class_ambiguity_proxy,
    score_confidence_entropy,
    score_distribution_shift,
    score_low_confidence,
)


# ─── Configuration ───────────────────────────────────────────────────────────

def load_config() -> Dict:
    """Load active learning config from sugal_orobanche_config.yaml."""
    config_path = PROJECT_ROOT / 'configs' / 'sugal_orobanche_config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        return full_config.get('active_learning', {}), full_config
    return {}, {}


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_detections_json(detections_path: str) -> Tuple[Dict, Dict]:
    """Load detections from a pre-saved detections.json file.

    Returns:
        (images_data, metadata) where images_data maps image_name -> {detections, width, height}
        and metadata has model info.
    """
    with open(detections_path, 'r') as f:
        data = json.load(f)

    metadata = {
        'model': data.get('model', 'unknown'),
        'slice_size': data.get('slice_size'),
        'confidence_threshold': data.get('confidence_threshold'),
        'source': str(detections_path),
    }

    return data['images'], metadata


def load_from_sahi_output(sahi_dir: str) -> Tuple[Dict, Dict]:
    """Load detections from an existing SAHI output directory.

    Looks for detections.json first; falls back to summary.json (no confidences).
    """
    sahi_path = Path(sahi_dir)

    # Look for detections.json (has per-detection confidence)
    det_path = sahi_path / 'detections.json'
    if det_path.exists():
        print(f"Found detections.json in {sahi_path.name}")
        return load_detections_json(str(det_path))

    # Fallback: summary.json (only has counts, no confidences)
    summary_path = sahi_path / 'summary.json'
    if summary_path.exists():
        print(f"WARNING: Only summary.json found in {sahi_path.name}")
        print("  summary.json has class counts but NOT per-detection confidences.")
        print("  Re-run SAHI with --save-detections for full uncertainty scoring.")
        print("  Falling back to count-based scoring only (borderline_density).")

        with open(summary_path, 'r') as f:
            summary = json.load(f)

        metadata = {
            'model': summary.get('model', 'unknown'),
            'slice_size': summary.get('slice_size'),
            'confidence_threshold': summary.get('confidence_threshold'),
            'source': str(summary_path),
            'limited': True,
        }

        # Build images_data from per_image_results (no actual detections)
        images_data = {}
        for result in summary.get('per_image_results', []):
            name = result['image']
            # Create fake detections with no confidence (just counts)
            images_data[name] = {
                'detections': [],
                'sahi_total': result.get('sahi_total', 0),
            }

        return images_data, metadata

    raise FileNotFoundError(
        f"No detections.json or summary.json found in {sahi_dir}"
    )


def run_inference_and_collect(
    images_dir: str,
    model_path: str,
    slice_size: int,
    overlap: float,
    confidence: float,
    device: str,
) -> Tuple[Dict, Dict]:
    """Run SAHI inference on images and return per-detection data.

    Returns same format as load_detections_json.
    """
    from PIL import Image as PILImage

    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    images_path = Path(images_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = sorted([
        f for f in images_path.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    print(f"Running SAHI inference on {len(image_files)} images...")
    print(f"  Model: {model_path}")
    print(f"  Slice: {slice_size}px, Conf: {confidence}")

    # Load model once
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=confidence,
        device=device,
    )

    CLASS_NAMES = {0: 'LYPES-G', 1: 'LYPES-O', 2: 'LYPES-R', 3: 'LYPES-Y', 4: 'ORARA'}

    images_data = {}
    t0 = time.perf_counter()

    for idx, img_path in enumerate(image_files):
        # Get dimensions cheaply
        with PILImage.open(img_path) as im:
            w, h = im.size

        result = get_sliced_prediction(
            str(img_path),
            sahi_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            verbose=0,
        )

        detections = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            detections.append({
                'class_id': pred.category.id,
                'class_name': pred.category.name or CLASS_NAMES.get(pred.category.id, f'class_{pred.category.id}'),
                'bbox': [float(c) for c in bbox],
                'confidence': float(pred.score.value),
            })

        images_data[img_path.name] = {
            'detections': detections,
            'width': w,
            'height': h,
        }

        if (idx + 1) % 50 == 0 or idx == len(image_files) - 1:
            elapsed = time.perf_counter() - t0
            rate = (idx + 1) / elapsed
            print(f"  [{idx+1}/{len(image_files)}] {rate:.1f} img/s")

    metadata = {
        'model': model_path,
        'slice_size': slice_size,
        'confidence_threshold': confidence,
        'source': str(images_dir),
    }

    return images_data, metadata


# ─── Scoring ─────────────────────────────────────────────────────────────────

def score_all_images(
    images_data: Dict,
    weights: Dict[str, float],
    conf_lo: float,
    conf_hi: float,
    ambiguous_classes: List[str],
    reference: Optional[Tuple[np.ndarray, float]] = None,
    embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """Score all images on uncertainty criteria.

    Args:
        images_data: Dict of image_name -> {detections, width, height}.
        weights: Scoring weights per criterion.
        conf_lo: Lower confidence threshold.
        conf_hi: Upper uncertain zone boundary.
        ambiguous_classes: Classes prone to visual confusion.
        reference: (centroid, std) from training distribution, or None.
        embeddings: Dict of image_name -> CLIP embedding, or None.

    Returns:
        DataFrame with one row per image, sorted by composite_score descending.
    """
    # Dataset-level stats
    median_count, mad = compute_dataset_stats(images_data)
    print(f"Dataset stats: median={median_count:.0f} detections, MAD={mad:.1f}")

    has_reference = reference is not None and embeddings is not None
    if not has_reference:
        print("No CLIP reference provided - skipping distribution_shift criterion")

    rows = []
    for img_name, img_data in images_data.items():
        dets = img_data.get('detections', [])
        n_dets = len(dets)

        # Compute individual scores
        s_low = score_low_confidence(dets, lo=conf_lo, hi=conf_hi) if dets else 0.0
        s_entropy = score_confidence_entropy(dets) if dets else 0.0
        s_density = score_borderline_density(n_dets, median_count, mad)
        s_ambig = score_class_ambiguity_proxy(dets, ambiguous_classes, threshold=conf_hi) if dets else 0.0

        s_shift = None
        if has_reference and img_name in embeddings:
            centroid, std = reference
            s_shift = score_distribution_shift(embeddings[img_name], centroid, std)

        scores = {
            'low_confidence': s_low,
            'confidence_entropy': s_entropy,
            'borderline_density': s_density,
            'class_ambiguity': s_ambig,
        }
        if s_shift is not None:
            scores['distribution_shift'] = s_shift

        composite = compute_composite_score(scores, weights)

        row = {
            'image': img_name,
            'n_detections': n_dets,
            'composite_score': composite,
            'low_confidence': s_low,
            'confidence_entropy': s_entropy,
            'borderline_density': s_density,
            'class_ambiguity': s_ambig,
        }
        if s_shift is not None:
            row['distribution_shift'] = s_shift

        # Add mean/min confidence for diagnostics
        if dets:
            confs = [d['confidence'] for d in dets]
            row['mean_confidence'] = np.mean(confs)
            row['min_confidence'] = np.min(confs)
        else:
            row['mean_confidence'] = None
            row['min_confidence'] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1

    return df


# ─── Output Generation ──────────────────────────────────────────────────────

def generate_uncertainty_panels(
    top_images: pd.DataFrame,
    images_data: Dict,
    images_dir: Optional[str],
    output_dir: Path,
    conf_hi: float,
    panel_colors: Dict,
):
    """Generate annotated panels for top uncertain images.

    Boxes are color-coded by confidence: red (low), orange (medium), green (high).
    """
    panels_dir = output_dir / 'uncertain_panels'
    panels_dir.mkdir(parents=True, exist_ok=True)

    # Parse panel colors
    color_low = tuple(panel_colors.get('low', [255, 0, 0]))
    color_med = tuple(panel_colors.get('medium', [255, 165, 0]))
    color_high = tuple(panel_colors.get('high', [0, 200, 0]))

    generated = 0
    for _, row in top_images.iterrows():
        img_name = row['image']
        img_data = images_data.get(img_name, {})
        dets = img_data.get('detections', [])

        # Find the image file
        img_path = _find_image(img_name, images_dir)
        if img_path is None:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        overlay = image.copy()

        for det in dets:
            conf = det['confidence']
            if conf < conf_hi:
                color_bgr = (color_low[2], color_low[1], color_low[0])
            elif conf < 0.70:
                color_bgr = (color_med[2], color_med[1], color_med[0])
            else:
                color_bgr = (color_high[2], color_high[1], color_high[0])

            x1, y1, x2, y2 = [int(c) for c in det['bbox']]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
            cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, 2)

            label = f"{det['class_name']} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color_bgr, -1)
            cv2.putText(image, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Blend overlay
        image = cv2.addWeighted(overlay, 0.2, image, 0.8, 0)

        # Title bar
        rank = int(row['rank'])
        score = row['composite_score']
        title = f"#{rank} | {img_name} | score={score:.3f} | dets={len(dets)}"
        cv2.rectangle(image, (0, 0), (len(title) * 11 + 20, 30), (0, 0, 0), -1)
        cv2.putText(image, title, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        panel_path = panels_dir / f"rank{rank:03d}_{img_name}"
        cv2.imwrite(str(panel_path), image)
        generated += 1

    print(f"Generated {generated} uncertainty panels in {panels_dir}")


def export_yolo_labels(
    top_images: pd.DataFrame,
    images_data: Dict,
    images_dir: Optional[str],
    output_dir: Path,
):
    """Export top-N images + YOLO labels for re-annotation tools."""
    export_dir = output_dir / 'uncertain_yolo_export'
    images_out = export_dir / 'images'
    labels_out = export_dir / 'labels'
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    exported = 0
    for _, row in top_images.iterrows():
        img_name = row['image']
        img_data = images_data.get(img_name, {})
        dets = img_data.get('detections', [])

        # Copy image
        img_path = _find_image(img_name, images_dir)
        if img_path is None:
            continue

        import shutil
        shutil.copy2(str(img_path), str(images_out / img_name))

        # Write YOLO label
        w = img_data.get('width', 1)
        h = img_data.get('height', 1)
        stem = Path(img_name).stem
        label_path = labels_out / f"{stem}.txt"

        with open(label_path, 'w') as f:
            for det in dets:
                x1, y1, x2, y2 = det['bbox']
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{det['class_id']} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

        exported += 1

    print(f"Exported {exported} images + labels to {export_dir}")


def generate_report(
    ranking: pd.DataFrame,
    metadata: Dict,
    top_n: int,
    output_dir: Path,
):
    """Generate summary report JSON."""
    n_total = len(ranking)
    top = ranking.head(top_n)

    report = {
        'timestamp': datetime.now().isoformat(),
        'source': metadata.get('source', 'unknown'),
        'model': metadata.get('model', 'unknown'),
        'total_images': n_total,
        'top_n': top_n,
        'score_distribution': {
            'mean': float(ranking['composite_score'].mean()),
            'std': float(ranking['composite_score'].std()),
            'median': float(ranking['composite_score'].median()),
            'max': float(ranking['composite_score'].max()),
            'min': float(ranking['composite_score'].min()),
            'p90': float(ranking['composite_score'].quantile(0.90)),
            'p95': float(ranking['composite_score'].quantile(0.95)),
        },
        'top_n_stats': {
            'mean_score': float(top['composite_score'].mean()),
            'mean_detections': float(top['n_detections'].mean()),
            'mean_confidence': float(top['mean_confidence'].dropna().mean()) if top['mean_confidence'].notna().any() else None,
        },
        'criteria_means': {
            'low_confidence': float(ranking['low_confidence'].mean()),
            'confidence_entropy': float(ranking['confidence_entropy'].mean()),
            'borderline_density': float(ranking['borderline_density'].mean()),
            'class_ambiguity': float(ranking['class_ambiguity'].mean()),
        },
    }

    if 'distribution_shift' in ranking.columns:
        report['criteria_means']['distribution_shift'] = float(ranking['distribution_shift'].mean())

    report_path = output_dir / 'uncertainty_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to {report_path}")
    return report


def create_fiftyone_dataset(
    ranking: pd.DataFrame,
    images_data: Dict,
    images_dir: Optional[str],
    top_n: int,
    dataset_name: str,
):
    """Create FiftyOne dataset for interactive review."""
    try:
        import fiftyone as fo
    except ImportError:
        print("FiftyOne not available - skipping dataset creation")
        return

    top = ranking.head(top_n)
    samples = []

    for _, row in top.iterrows():
        img_name = row['image']
        img_path = _find_image(img_name, images_dir)
        if img_path is None:
            continue

        sample = fo.Sample(filepath=str(img_path))
        sample['uncertainty_rank'] = int(row['rank'])
        sample['composite_score'] = float(row['composite_score'])
        sample['n_detections'] = int(row['n_detections'])

        for criterion in ['low_confidence', 'confidence_entropy',
                          'borderline_density', 'class_ambiguity']:
            if criterion in row:
                sample[criterion] = float(row[criterion])

        # Add detections as FiftyOne Detections
        img_data = images_data.get(img_name, {})
        dets = img_data.get('detections', [])
        w = img_data.get('width', 1)
        h = img_data.get('height', 1)

        fo_dets = []
        for det in dets:
            x1, y1, x2, y2 = det['bbox']
            fo_dets.append(
                fo.Detection(
                    label=det['class_name'],
                    bounding_box=[x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                    confidence=det['confidence'],
                )
            )
        sample['predictions'] = fo.Detections(detections=fo_dets)

        # Tag by uncertainty level
        score = row['composite_score']
        if score > ranking['composite_score'].quantile(0.95):
            sample.tags.append('very_uncertain')
        elif score > ranking['composite_score'].quantile(0.75):
            sample.tags.append('uncertain')

        samples.append(sample)

    # Delete existing dataset if present
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)
    dataset.add_samples(samples)
    dataset.persistent = True

    print(f"FiftyOne dataset '{dataset_name}' created with {len(samples)} samples")
    print(f"  Launch: python -c \"import fiftyone as fo; fo.load_dataset('{dataset_name}').launch()\"")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _find_image(img_name: str, images_dir: Optional[str]) -> Optional[Path]:
    """Locate an image file by name in the images directory."""
    if images_dir is None:
        return None
    p = Path(images_dir) / img_name
    if p.exists():
        return p
    return None


def _resolve_images_dir(
    images_data: Dict,
    metadata: Dict,
    cli_images_dir: Optional[str],
) -> Optional[str]:
    """Determine where the original images are located."""
    if cli_images_dir:
        return cli_images_dir

    source = metadata.get('source', '')
    if source and Path(source).is_dir():
        return source

    # Try to infer from detections.json path
    if source and Path(source).is_file():
        # detections.json is inside a sahi output dir, images may be in --images
        # Can't reliably infer - return None
        pass

    return None


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Uncertainty Mining for Active Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From pre-saved detections
  python scripts/cli/mine_uncertain_samples.py \\
      --detections outputs/test/sahi_*/detections.json --top 20

  # From new images
  python scripts/cli/mine_uncertain_samples.py \\
      --images /path/to/images --top 30

  # With CLIP distribution shift
  python scripts/cli/mine_uncertain_samples.py \\
      --detections outputs/test/detections.json \\
      --reference-data configs/yaml_config/data_tomate-orobanche-v4-4488tiles.yaml
        """,
    )

    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--detections', type=str,
                             help='Path to detections.json from --save-detections')
    input_group.add_argument('--images', type=str,
                             help='Path to directory of new images (runs SAHI inference)')
    input_group.add_argument('--sahi-output', type=str,
                             help='Path to existing SAHI output directory')

    # Inference parameters (used with --images)
    parser.add_argument('--model', type=str, default=None,
                        help='Model path (default: from config)')
    parser.add_argument('--slice', type=int, default=None,
                        help='SAHI slice size (default: from config)')
    parser.add_argument('--conf', type=float, default=None,
                        help='Confidence threshold (default: from config)')
    parser.add_argument('--overlap', type=float, default=0.2,
                        help='Overlap ratio (default: 0.2)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: from config)')

    # Scoring parameters
    parser.add_argument('--conf-uncertain', type=float, default=None,
                        help='Upper bound of uncertain zone (default: from config)')
    parser.add_argument('--top', type=int, default=None,
                        help='Number of top uncertain images to export (default: from config)')
    parser.add_argument('--ambiguous-classes', type=str, nargs='+', default=None,
                        help='Classes prone to visual confusion (default: from config). '
                             'E.g., --ambiguous-classes AMBEL POLAV POLPE for Lencu weeds')

    # CLIP reference (optional)
    parser.add_argument('--reference-data', type=str, default=None,
                        help='YOLO dataset YAML for building CLIP reference')
    parser.add_argument('--reference-cache', type=str, default=None,
                        help='Path to cached .npz reference (skip recomputation)')

    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: from config)')
    parser.add_argument('--images-dir', type=str, default=None,
                        help='Directory containing original images (for panel generation)')

    # Feature flags
    parser.add_argument('--no-panels', dest='panels', action='store_false',
                        help='Skip annotated panel generation')
    parser.set_defaults(panels=True)
    parser.add_argument('--no-fiftyone', dest='fiftyone', action='store_false',
                        help='Skip FiftyOne dataset creation')
    parser.set_defaults(fiftyone=True)
    parser.add_argument('--no-yolo-export', dest='yolo_export', action='store_false',
                        help='Skip YOLO label export')
    parser.set_defaults(yolo_export=True)

    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────
    al_config, full_config = load_config()
    inf_config = full_config.get('inference', {})

    # Resolve parameters (CLI > config > defaults)
    model_path = args.model or full_config.get('model', {}).get('path')
    slice_size = args.slice or inf_config.get('slice_size', 1024)
    conf_threshold = args.conf or inf_config.get('confidence', 0.25)
    device = args.device or inf_config.get('device', 'cuda:0')
    conf_uncertain = args.conf_uncertain or al_config.get('confidence_uncertain_upper', 0.50)
    top_n = args.top or al_config.get('default_top_n', 50)
    output_base = args.output or al_config.get('output_base', 'outputs/active_learning')
    weights = al_config.get('weights', {
        'low_confidence': 0.30,
        'confidence_entropy': 0.15,
        'borderline_density': 0.15,
        'class_ambiguity': 0.15,
        'distribution_shift': 0.25,
    })
    ambiguous_classes = (args.ambiguous_classes
                         or al_config.get('ambiguous_classes',
                                          ['LYPES-G', 'LYPES-O', 'LYPES-R', 'LYPES-Y']))
    panel_colors = al_config.get('panel_colors', {
        'high': [0, 200, 0],
        'medium': [255, 165, 0],
        'low': [255, 0, 0],
    })

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base)
    if not str(output_dir).endswith(timestamp[:8]):
        output_dir = output_dir / f"mining_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UNCERTAINTY MINING FOR ACTIVE LEARNING")
    print("=" * 60)

    # ── Step 1: Load or generate detections ──────────────────────────────
    t0 = time.perf_counter()

    if args.detections:
        print(f"\nLoading detections from: {args.detections}")
        images_data, metadata = load_detections_json(args.detections)
    elif args.sahi_output:
        print(f"\nLoading from SAHI output: {args.sahi_output}")
        images_data, metadata = load_from_sahi_output(args.sahi_output)
    elif args.images:
        if model_path is None:
            parser.error("--images requires --model (or set model.path in config)")
        print(f"\nRunning inference on: {args.images}")
        images_data, metadata = run_inference_and_collect(
            args.images, model_path, slice_size, args.overlap, conf_threshold, device,
        )
    else:
        parser.error("One of --detections, --images, or --sahi-output is required")

    n_images = len(images_data)
    t_load = time.perf_counter() - t0
    print(f"Loaded {n_images} images in {t_load:.1f}s")

    # Resolve images directory for panels/export
    images_dir = _resolve_images_dir(images_data, metadata, args.images_dir or args.images)

    # ── Step 2: Build CLIP reference (optional) ──────────────────────────
    reference = None
    embeddings = None

    if args.reference_cache or args.reference_data:
        print("\nBuilding CLIP reference distribution...")
        from scripts.active_learning.reference_distribution import (
            build_reference_from_training,
            compute_image_embeddings_batched,
            load_reference,
        )

        cache_path = args.reference_cache
        if cache_path is None:
            cache_path = str(output_dir / 'clip_reference.npz')

        if args.reference_cache and Path(args.reference_cache).exists():
            reference = load_reference(args.reference_cache)
        elif args.reference_data:
            reference = build_reference_from_training(
                args.reference_data,
                split='train',
                cache_path=cache_path,
                device=device.replace('cuda:', 'cuda') if 'cuda' in device else device,
            )

        # Compute embeddings for input images
        if reference is not None and images_dir:
            print("Computing CLIP embeddings for input images...")
            image_paths = []
            image_names = []
            for img_name in images_data.keys():
                p = _find_image(img_name, images_dir)
                if p is not None:
                    image_paths.append(p)
                    image_names.append(img_name)

            if image_paths:
                emb_array = compute_image_embeddings_batched(
                    image_paths,
                    device=device.replace('cuda:', 'cuda') if 'cuda' in device else device,
                )
                embeddings = dict(zip(image_names, emb_array))
                print(f"Computed embeddings for {len(embeddings)} images")

    # ── Step 3: Score all images ─────────────────────────────────────────
    print(f"\nScoring {n_images} images...")
    t_score0 = time.perf_counter()

    ranking = score_all_images(
        images_data=images_data,
        weights=weights,
        conf_lo=conf_threshold,
        conf_hi=conf_uncertain,
        ambiguous_classes=ambiguous_classes,
        reference=reference,
        embeddings=embeddings,
    )

    t_score = time.perf_counter() - t_score0
    print(f"Scoring complete in {t_score:.1f}s")

    # ── Step 4: Save ranking CSV ─────────────────────────────────────────
    csv_path = output_dir / 'uncertain_ranking.csv'
    ranking.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nRanking saved to: {csv_path}")

    # ── Step 5: Generate outputs ─────────────────────────────────────────
    top_images = ranking.head(top_n)

    print(f"\nTop {min(top_n, len(ranking))} most uncertain images:")
    print(f"{'Rank':<6} {'Image':<35} {'Score':<8} {'Dets':<6} {'MeanConf':<9}")
    print("-" * 65)
    for _, row in top_images.head(10).iterrows():
        mc = f"{row['mean_confidence']:.3f}" if row['mean_confidence'] is not None else "N/A"
        print(f"{int(row['rank']):<6} {row['image'][:34]:<35} {row['composite_score']:<8.4f} "
              f"{int(row['n_detections']):<6} {mc:<9}")
    if len(top_images) > 10:
        print(f"  ... and {len(top_images) - 10} more")

    # Generate panels
    if args.panels and images_dir:
        print(f"\nGenerating uncertainty panels for top {top_n}...")
        generate_uncertainty_panels(
            top_images, images_data, images_dir, output_dir, conf_uncertain, panel_colors,
        )
    elif args.panels and not images_dir:
        print("\nSkipping panels: cannot locate original images (use --images-dir)")

    # Export YOLO labels
    if args.yolo_export and images_dir:
        print(f"\nExporting YOLO labels for top {top_n}...")
        export_yolo_labels(top_images, images_data, images_dir, output_dir)
    elif args.yolo_export and not images_dir:
        print("\nSkipping YOLO export: cannot locate original images (use --images-dir)")

    # Generate report
    report = generate_report(ranking, metadata, top_n, output_dir)

    # FiftyOne dataset
    if args.fiftyone and images_dir:
        ds_name = f"uncertainty_mining_{timestamp}"
        create_fiftyone_dataset(ranking, images_data, images_dir, top_n, ds_name)
    elif args.fiftyone and not images_dir:
        print("\nSkipping FiftyOne: cannot locate original images (use --images-dir)")

    # ── Summary ──────────────────────────────────────────────────────────
    total_time = time.perf_counter() - t0
    print("\n" + "=" * 60)
    print("UNCERTAINTY MINING COMPLETE")
    print("=" * 60)
    print(f"Total images analyzed: {n_images}")
    print(f"Top-{top_n} exported to:    {output_dir}")
    print(f"Score range:            {ranking['composite_score'].min():.4f} - {ranking['composite_score'].max():.4f}")
    print(f"Total time:             {total_time:.1f}s")
    print(f"\nOutputs:")
    print(f"  {csv_path}")
    if args.panels and images_dir:
        print(f"  {output_dir / 'uncertain_panels/'}")
    if args.yolo_export and images_dir:
        print(f"  {output_dir / 'uncertain_yolo_export/'}")
    print(f"  {output_dir / 'uncertainty_report.json'}")


if __name__ == '__main__':
    main()
