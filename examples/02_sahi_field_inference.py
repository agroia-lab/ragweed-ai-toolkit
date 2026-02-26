"""
Example 02: SAHI Sliced Inference Configuration
================================================

Shows how to configure SAHI (Slicing Aided Hyper Inference) for
high-resolution field images using the chapter's validated parameters
(Section 3.3).

SAHI divides large images into overlapping tiles, runs detection on
each tile, and merges results with NMS -- critical for dense weed
scenes where a single pass misses small objects.

This script prints the configuration and demonstrates the batch
inference API pattern.  Actual inference requires a trained model
checkpoint and input images.
"""

from ragweed_toolkit.detection.inference import SahiConfig

# ------------------------------------------------------------------ #
# 1.  Create a config with chapter defaults (Section 3.3)
# ------------------------------------------------------------------ #
cfg = SahiConfig(
    model_path="weights/best.pt",
    slice_size=640,
    overlap_ratio=0.2,
    confidence_threshold=0.25,
    nms_iou=0.5,
    max_det=10000,   # agricultural scenes are dense!
    device="cuda:0",
)

print("=== SAHI Config (Chapter Section 3.3) ===")
print(f"  Model          : {cfg.model_path}")
print(f"  Slice size     : {cfg.slice_size}px")
print(f"  Overlap        : {cfg.overlap_ratio:.0%}")
print(f"  Confidence     : {cfg.confidence_threshold}")
print(f"  NMS IoU        : {cfg.nms_iou}")
print(f"  Max detections : {cfg.max_det}")
print(f"  Device         : {cfg.device}")

# ------------------------------------------------------------------ #
# 2.  Why max_det=10000 matters
# ------------------------------------------------------------------ #
print("\n=== Why max_det=10000? ===")
print("  YOLO's default max_det is 300.")
print("  Dense ragweed fields easily produce 1000+ detections per image.")
print("  With default settings, detections are silently dropped.")
print("  The toolkit overrides this inside the SAHI model loader.")

# ------------------------------------------------------------------ #
# 3.  Batch inference pattern (requires model + images)
# ------------------------------------------------------------------ #
print("\n=== Batch inference usage ===")
print("""
  from ragweed_toolkit.detection.inference import SahiConfig, run_sahi_batch

  cfg = SahiConfig(model_path="weights/best.pt", device="cuda:0")

  images = ["field_001.jpg", "field_002.jpg", "field_003.jpg"]
  results = run_sahi_batch(images, cfg)

  for name, detections in results.items():
      print(f"{name}: {len(detections)} ragweed detections")
""")

# ------------------------------------------------------------------ #
# 4.  Single image inference pattern
# ------------------------------------------------------------------ #
print("=== Single image usage ===")
print("""
  from ragweed_toolkit.detection.inference import SahiConfig, run_sahi

  cfg = SahiConfig(model_path="weights/best.pt", device="cuda:0")
  detections = run_sahi("field_001.jpg", cfg)

  for det in detections[:5]:
      print(f"  class={det['class_name']}, conf={det['confidence']:.2f}, "
            f"bbox={det['bbox']}")
""")
