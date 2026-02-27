#!/usr/bin/env python3
"""
SAHI Inference CLI with Ground Truth vs Prediction Comparison Panels

Usage:
    python scripts/sahi_inference_cli.py --model path/to/best.pt --images path/to/test/images --slice 1024
    python scripts/sahi_inference_cli.py --model path/to/best.pt --images path/to/test --labels path/to/test/labels --slice 640
"""

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import torch
from sahi import AutoDetectionModel
from ultralytics import YOLO

# ragweed_toolkit library imports
from ragweed_toolkit.detection import SahiConfig, run_sahi

# Default class colors (RGB) - extended palette for any number of classes
DEFAULT_COLORS = [
    (0, 255, 0),      # Green
    (255, 165, 0),    # Orange
    (255, 0, 0),      # Red
    (255, 255, 0),    # Yellow
    (128, 0, 128),    # Purple
    (0, 0, 255),      # Blue
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (165, 42, 42),    # Brown
    (0, 128, 0),      # Dark Green
]

# These will be populated dynamically from the model
CLASS_COLORS = {}
CLASS_NAMES = {}


def init_class_info_from_model(model_path: str):
    """Load class names and colors from the YOLO model."""
    global CLASS_NAMES, CLASS_COLORS
    model = YOLO(model_path)
    CLASS_NAMES = dict(model.names)
    CLASS_COLORS = {i: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in CLASS_NAMES}
    del model
    torch.cuda.empty_cache()
    return CLASS_NAMES


def load_yolo_labels(label_path: Path, img_width: int, img_height: int) -> List[Dict]:
    """Load YOLO format labels and convert to pixel coordinates."""
    labels = []
    if not label_path.exists():
        return labels

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height

                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                labels.append({
                    'class_id': cls_id,
                    'class_name': CLASS_NAMES.get(cls_id, f'class_{cls_id}'),
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 1.0  # GT has 100% confidence
                })
    return labels


def draw_boxes_on_image(
    image: np.ndarray,
    detections: List[Dict],
    title: str = "",
    show_confidence: bool = True,
    alpha: float = 0.3
) -> np.ndarray:
    """Draw bounding boxes on image with semi-transparent fill."""
    img = image.copy()
    overlay = img.copy()

    # Draw boxes
    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det['bbox']]
        cls_id = det['class_id']
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))

        # Convert RGB to BGR for OpenCV
        color_bgr = (color[2], color[1], color[0])

        # Draw filled rectangle on overlay
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)

        # Draw border on main image
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)

        # Label
        label = det['class_name']
        if show_confidence and 'confidence' in det:
            label += f" {det['confidence']:.2f}"

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color_bgr, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Blend overlay
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Add title
    if title:
        cv2.rectangle(img, (0, 0), (len(title) * 12 + 20, 30), (0, 0, 0), -1)
        cv2.putText(img, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img


def create_comparison_panel(
    image: np.ndarray,
    gt_detections: List[Dict],
    pred_detections: List[Dict],
    image_name: str,
    slice_size: int
) -> np.ndarray:
    """Create side-by-side comparison panel: Original | GT | Prediction."""
    h, w = image.shape[:2]

    # Create three versions
    original = image.copy()
    gt_image = draw_boxes_on_image(image, gt_detections, f"Ground Truth ({len(gt_detections)} objects)", show_confidence=False)
    pred_image = draw_boxes_on_image(image, pred_detections, f"SAHI Pred (slice={slice_size}, {len(pred_detections)} objects)", show_confidence=True)

    # Add labels to original
    cv2.rectangle(original, (0, 0), (200, 30), (0, 0, 0), -1)
    cv2.putText(original, "Original", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Stack horizontally
    panel = np.hstack([original, gt_image, pred_image])

    # Add overall title
    title_bar = np.zeros((50, panel.shape[1], 3), dtype=np.uint8)
    title = f"{image_name} | GT: {len(gt_detections)} | Pred: {len(pred_detections)} | Diff: {len(pred_detections) - len(gt_detections):+d}"
    cv2.putText(title_bar, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    panel = np.vstack([title_bar, panel])

    return panel


def create_inference_panel(
    image: np.ndarray,
    pred_detections: List[Dict],
    image_name: str,
    slice_size: int
) -> np.ndarray:
    """Create side-by-side panel for unlabeled data: Original | SAHI Prediction."""
    h, w = image.shape[:2]

    # Create two versions
    original = image.copy()
    pred_image = draw_boxes_on_image(image, pred_detections, f"SAHI (slice={slice_size}, {len(pred_detections)} detections)", show_confidence=True)

    # Add label to original
    cv2.rectangle(original, (0, 0), (200, 30), (0, 0, 0), -1)
    cv2.putText(original, "Original", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Stack horizontally
    panel = np.hstack([original, pred_image])

    # Add overall title
    title_bar = np.zeros((50, panel.shape[1], 3), dtype=np.uint8)
    title = f"{image_name} | Detections: {len(pred_detections)}"
    cv2.putText(title_bar, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    panel = np.vstack([title_bar, panel])

    return panel


def create_class_comparison_chart(
    gt_counts: Dict[str, int],
    pred_counts: Dict[str, int],
    image_name: str,
    output_path: Path
):
    """Create bar chart comparing GT vs Prediction counts per class."""
    classes = list(CLASS_NAMES.values())
    gt_values = [gt_counts.get(cls, 0) for cls in classes]
    pred_values = [pred_counts.get(cls, 0) for cls in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, gt_values, width, label='Ground Truth', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, pred_values, width, label='Prediction', color='blue', alpha=0.7)

    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(f'Object Count Comparison: {image_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def count_by_class(detections: List[Dict]) -> Dict[str, int]:
    """Count detections by class."""
    counts = defaultdict(int)
    for det in detections:
        counts[det['class_name']] += 1
    return dict(counts)


def run_sahi_inference(
    detection_model: AutoDetectionModel,
    image_path: Path,
    slice_size: int,
    overlap_ratio: float,
    confidence_threshold: float
) -> List[Dict]:
    """Run SAHI sliced inference on a single image.

    Delegates to ragweed_toolkit.detection.run_sahi() for the core inference,
    reusing the pre-loaded SAHI model for efficiency.
    """
    cfg = SahiConfig(
        slice_size=slice_size,
        overlap_ratio=overlap_ratio,
        confidence_threshold=confidence_threshold,
    )
    return run_sahi(image_path, cfg, _sahi_model=detection_model)


def run_direct_inference(
    model: YOLO,
    image_path: Path,
    imgsz: int,
    confidence_threshold: float
) -> List[Dict]:
    """Run direct YOLO inference without SAHI."""
    results = model.predict(
        str(image_path),
        imgsz=imgsz,
        conf=confidence_threshold,
        verbose=False
    )

    detections = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            bbox = boxes.xyxy[i].cpu().numpy()

            detections.append({
                'class_id': cls_id,
                'class_name': CLASS_NAMES.get(cls_id, f'class_{cls_id}'),
                'bbox': [float(c) for c in bbox],
                'confidence': conf
            })

    return detections


def process_dataset(args):
    """Process all images in a dataset."""

    # Setup paths
    images_dir = Path(args.images)
    labels_dir = Path(args.labels) if args.labels else images_dir.parent / 'labels'
    model_path = args.model
    inference_mode = args.mode == 'inference'

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "inference" if inference_mode else "comparison"
    output_dir = Path(args.output) / f"sahi_{mode_suffix}_{args.slice}px_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    panels_dir = output_dir / 'comparison_panels'
    panels_dir.mkdir(exist_ok=True)

    charts_dir = output_dir / 'count_charts'
    charts_dir.mkdir(exist_ok=True)

    # Also create labels output dir in inference mode
    if inference_mode:
        labels_output_dir = output_dir / 'labels'
        labels_output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    if inference_mode:
        print("🔍 SAHI Inference (No Ground Truth)")
    else:
        print("🔍 SAHI Inference with GT Comparison")
    print("=" * 60)
    print(f"📁 Images: {images_dir}")
    if not inference_mode:
        print(f"📁 Labels: {labels_dir}")
    print(f"🤖 Model: {model_path}")
    print(f"🔪 Slice size: {args.slice}px")
    print(f"📏 Overlap: {args.overlap}")
    print(f"📁 Output: {output_dir}")
    print("=" * 60)

    # Initialize class info from model
    print("\n🏷️  Loading class names from model...")
    class_names = init_class_info_from_model(model_path)
    print(f"   Classes: {list(class_names.values())}")

    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions]
    print(f"\n📷 Found {len(image_files)} images")

    # Also load model for direct inference comparison (only in comparison mode)
    direct_model = YOLO(model_path) if (not inference_mode and args.direct) else None

    # Load SAHI model ONCE (big speedup vs loading per-image)
    print("\n🤖 Loading SAHI model (one-time)...")
    t_model0 = time.perf_counter()
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=args.conf,
        device=args.device
    )
    t_model1 = time.perf_counter()
    print(f"✅ SAHI model loaded in {(t_model1 - t_model0):.2f}s")

    # Override max_det for dense agricultural scenes (default 300 drops detections)
    max_det = args.max_det
    try:
        sahi_model.model.model.args['max_det'] = max_det
        print(f"🔧 max_det overridden to {max_det} (default was 300)")
    except Exception as e:
        print(f"⚠️  Could not override max_det: {e}")

    # Process each image
    all_results = []
    total_gt = defaultdict(int)
    total_pred_sahi = defaultdict(int)
    total_pred_direct = defaultdict(int)

    # Timing stats
    t_total0 = time.perf_counter()
    t_sahi_sum = 0.0
    t_direct_sum = 0.0
    t_panels_sum = 0.0
    t_charts_sum = 0.0
    t_labels_sum = 0.0

    # Per-detection data for active learning (--save-detections)
    all_detections_data = {} if args.save_detections else None

    for idx, image_path in enumerate(image_files):
        print(f"\n[{idx+1}/{len(image_files)}] Processing: {image_path.name}")
        t_img0 = time.perf_counter()

        # Load image only if needed for panels; otherwise avoid decoding cost.
        image = None
        if args.panels:
            image = cv2.imread(str(image_path))
            if image is None:
                print("  ⚠️ Could not load image, skipping")
                continue
            h, w = image.shape[:2]
        else:
            # Need width/height for label export; get it cheaply via PIL.
            with Image.open(image_path) as im:
                w, h = im.size

        # Load ground truth (only in comparison mode)
        gt_detections = []
        if not inference_mode:
            label_path = labels_dir / f"{image_path.stem}.txt"
            gt_detections = load_yolo_labels(label_path, w, h)
            print(f"  📋 GT: {len(gt_detections)} objects")

        # Run SAHI inference
        print(f"  🔍 Running SAHI (slice={args.slice}px)...")
        t0 = time.perf_counter()
        sahi_detections = run_sahi_inference(sahi_model, image_path, args.slice, args.overlap, args.conf)
        t1 = time.perf_counter()
        t_sahi_sum += (t1 - t0)
        print(f"  🎯 SAHI: {len(sahi_detections)} detections")

        # Accumulate per-detection data for active learning
        if all_detections_data is not None:
            all_detections_data[image_path.name] = {
                'detections': sahi_detections,
                'width': w,
                'height': h,
            }

        # Run direct inference for comparison (only in comparison mode)
        direct_detections = []
        if (not inference_mode) and args.direct and (direct_model is not None):
            print("  🔍 Running direct inference...")
            t0 = time.perf_counter()
            direct_detections = run_direct_inference(direct_model, image_path, args.slice, args.conf)
            t1 = time.perf_counter()
            t_direct_sum += (t1 - t0)
            print(f"  🎯 Direct: {len(direct_detections)} detections")

        # Count by class
        gt_counts = count_by_class(gt_detections)
        sahi_counts = count_by_class(sahi_detections)
        direct_counts = count_by_class(direct_detections)

        # Update totals
        for cls, count in gt_counts.items():
            total_gt[cls] += count
        for cls, count in sahi_counts.items():
            total_pred_sahi[cls] += count
        for cls, count in direct_counts.items():
            total_pred_direct[cls] += count

        # Create panel based on mode (optional - can be a major time sink)
        if args.panels:
            t0 = time.perf_counter()
            if inference_mode:
                # 2-panel: Original | SAHI
                panel = create_inference_panel(image, sahi_detections, image_path.name, args.slice)
            else:
                # 3-panel: Original | GT | SAHI
                panel = create_comparison_panel(image, gt_detections, sahi_detections, image_path.name, args.slice)
            panel_path = panels_dir / f"{image_path.stem}_comparison.jpg"
            cv2.imwrite(str(panel_path), panel)
            t1 = time.perf_counter()
            t_panels_sum += (t1 - t0)

        # Create count chart (only in comparison mode)
        if (not inference_mode) and args.charts:
            chart_path = charts_dir / f"{image_path.stem}_counts.png"
            t0 = time.perf_counter()
            create_class_comparison_chart(gt_counts, sahi_counts, image_path.name, chart_path)
            t1 = time.perf_counter()
            t_charts_sum += (t1 - t0)

        # Export YOLO labels in inference mode
        if inference_mode:
            label_output_path = labels_output_dir / f"{image_path.stem}.txt"
            t0 = time.perf_counter()
            with open(label_output_path, 'w') as f:
                for det in sahi_detections:
                    x1, y1, x2, y2 = det['bbox']
                    # Convert to YOLO format (normalized)
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    f.write(f"{det['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            t1 = time.perf_counter()
            t_labels_sum += (t1 - t0)

        # Store results
        all_results.append({
            'image': image_path.name,
            'gt_total': len(gt_detections),
            'sahi_total': len(sahi_detections),
            'direct_total': len(direct_detections),
            'gt_by_class': gt_counts,
            'sahi_by_class': sahi_counts,
            'direct_by_class': direct_counts
        })

        t_img1 = time.perf_counter()
        print(f"  ⏱️ Image total: {(t_img1 - t_img0):.2f}s")

    # Create summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    summary = {
        'model': model_path,
        'mode': args.mode,
        'slice_size': args.slice,
        'overlap': args.overlap,
        'confidence_threshold': args.conf,
        'total_images': len(all_results),
        'totals': {
            'sahi_predictions': dict(total_pred_sahi),
        },
        'per_image_results': all_results
    }

    if not inference_mode:
        summary['totals']['ground_truth'] = dict(total_gt)
        summary['totals']['direct_predictions'] = dict(total_pred_direct)

    # Save summary JSON
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save per-detection data for active learning
    if all_detections_data is not None:
        detections_output = {
            'model': model_path,
            'slice_size': args.slice,
            'overlap': args.overlap,
            'confidence_threshold': args.conf,
            'device': args.device,
            'timestamp': datetime.now().isoformat(),
            'total_images': len(all_detections_data),
            'images': all_detections_data,
        }
        detections_path = output_dir / 'detections.json'
        with open(detections_path, 'w') as f:
            json.dump(detections_output, f, indent=2)
        print(f"\n📊 Detections saved to: {detections_path}")

    # Print summary table
    if inference_mode:
        # Simplified table for inference mode
        print(f"\n{'Class':<12} {'SAHI':>10}")
        print("-" * 25)
        for cls in CLASS_NAMES.values():
            sahi = total_pred_sahi.get(cls, 0)
            print(f"{cls:<12} {sahi:>10}")
        print("-" * 25)
        sahi_total = sum(total_pred_sahi.values())
        print(f"{'TOTAL':<12} {sahi_total:>10}")

        # Create inference summary chart (optional)
        if args.summary_chart:
            create_inference_summary_chart(total_pred_sahi, output_dir / 'detection_summary.png')
    else:
        print(f"\n{'Class':<12} {'GT':>8} {'SAHI':>8} {'Direct':>8} {'SAHI Diff':>10} {'Direct Diff':>12}")
        print("-" * 60)
        for cls in CLASS_NAMES.values():
            gt = total_gt.get(cls, 0)
            sahi = total_pred_sahi.get(cls, 0)
            direct = total_pred_direct.get(cls, 0)
            sahi_diff = sahi - gt
            direct_diff = direct - gt
            print(f"{cls:<12} {gt:>8} {sahi:>8} {direct:>8} {sahi_diff:>+10} {direct_diff:>+12}")

        print("-" * 60)
        gt_total = sum(total_gt.values())
        sahi_total = sum(total_pred_sahi.values())
        direct_total = sum(total_pred_direct.values())
        print(f"{'TOTAL':<12} {gt_total:>8} {sahi_total:>8} {direct_total:>8} {sahi_total-gt_total:>+10} {direct_total-gt_total:>+12}")

        # Create overall comparison chart (optional)
        if args.summary_chart:
            create_overall_comparison_chart(total_gt, total_pred_sahi, total_pred_direct, output_dir / 'overall_comparison.png')

    print(f"\n✅ Results saved to: {output_dir}")

    if inference_mode:
        print(f"📁 YOLO labels exported to: {labels_output_dir}")

    # Timing summary
    t_total1 = time.perf_counter()
    total_elapsed = t_total1 - t_total0
    n = max(1, len(all_results))
    print("\n" + "=" * 60)
    print("⏱️ TIMING (approx)")
    print("=" * 60)
    print(f"Total elapsed: {total_elapsed:.1f}s  |  Images: {len(all_results)}  |  {len(all_results)/max(1e-9,total_elapsed):.2f} img/s")
    print(f"SAHI inference total: {t_sahi_sum:.1f}s  ({t_sahi_sum/n:.2f}s/img)")
    if (not inference_mode) and args.direct:
        print(f"Direct inference total: {t_direct_sum:.1f}s  ({t_direct_sum/n:.2f}s/img)")
    if args.panels:
        print(f"Panels write total: {t_panels_sum:.1f}s  ({t_panels_sum/n:.2f}s/img)")
    if (not inference_mode) and args.charts:
        print(f"Charts total: {t_charts_sum:.1f}s  ({t_charts_sum/n:.2f}s/img)")
    if inference_mode:
        print(f"Label export total: {t_labels_sum:.1f}s  ({t_labels_sum/n:.3f}s/img)")

    return output_dir


def create_overall_comparison_chart(
    gt_counts: Dict[str, int],
    sahi_counts: Dict[str, int],
    direct_counts: Dict[str, int],
    output_path: Path
):
    """Create overall comparison chart."""
    classes = list(CLASS_NAMES.values())
    gt_values = [gt_counts.get(cls, 0) for cls in classes]
    sahi_values = [sahi_counts.get(cls, 0) for cls in classes]
    direct_values = [direct_counts.get(cls, 0) for cls in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - width, gt_values, width, label='Ground Truth', color='green', alpha=0.7)
    bars2 = ax.bar(x, sahi_values, width, label='SAHI', color='blue', alpha=0.7)
    bars3 = ax.bar(x + width, direct_values, width, label='Direct', color='orange', alpha=0.7)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Total Count', fontsize=12)
    ax.set_title('Overall Object Detection Comparison: GT vs SAHI vs Direct', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_inference_summary_chart(
    sahi_counts: Dict[str, int],
    output_path: Path
):
    """Create summary bar chart for inference mode (SAHI detections only)."""
    classes = list(CLASS_NAMES.values())
    sahi_values = [sahi_counts.get(cls, 0) for cls in classes]

    # Colors matching class colors
    colors = [
        '#00FF00',  # LYPES-G - Green
        '#FFA500',  # LYPES-O - Orange
        '#FF0000',  # LYPES-R - Red
        '#FFFF00',  # LYPES-Y - Yellow
        '#800080',  # ORARA - Purple
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(classes, sahi_values, color=colors, alpha=0.8, edgecolor='black')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Detection Count', fontsize=12)
    ax.set_title('SAHI Detection Summary by Class', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add total annotation
    total = sum(sahi_values)
    ax.annotate(f'Total: {total}', xy=(0.98, 0.98), xycoords='axes fraction',
                ha='right', va='top', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='SAHI Inference with GT vs Prediction Comparison')

    # Required arguments
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (best.pt)')
    parser.add_argument('--images', type=str, required=True, help='Path to test images directory')

    # Optional arguments
    parser.add_argument('--labels', type=str, default=None, help='Path to labels directory (default: ../labels)')
    parser.add_argument('--output', type=str, default='outputs/sahi_comparison', help='Output directory')
    parser.add_argument('--slice', type=int, default=1024, help='SAHI slice size (640 or 1024)')
    parser.add_argument('--overlap', type=float, default=0.2, help='Overlap ratio')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (cuda:0 or cpu)')
    parser.add_argument('--max-det', type=int, default=10000, help='Max detections per image (default: 10000)')
    parser.add_argument('--mode', type=str, default='comparison', choices=['comparison', 'inference'],
                        help='Mode: comparison (3-panel with GT) or inference (2-panel, no GT)')
    parser.add_argument('--no-panels', dest='panels', action='store_false',
                        help='Do not write comparison panels (big speedup).')
    parser.set_defaults(panels=True)
    parser.add_argument('--no-charts', dest='charts', action='store_false',
                        help='Do not write per-image count charts (comparison mode only).')
    parser.set_defaults(charts=True)
    parser.add_argument('--no-summary-chart', dest='summary_chart', action='store_false',
                        help='Do not write summary charts (detection_summary/overall_comparison).')
    parser.set_defaults(summary_chart=True)
    parser.add_argument('--no-direct', dest='direct', action='store_false',
                        help='Skip direct YOLO inference in comparison mode (speedup).')
    parser.set_defaults(direct=True)
    parser.add_argument('--save-detections', dest='save_detections', action='store_true',
                        help='Save per-detection confidence data to detections.json (for active learning).')
    parser.set_defaults(save_detections=False)

    args = parser.parse_args()
    process_dataset(args)


if __name__ == '__main__':
    main()
