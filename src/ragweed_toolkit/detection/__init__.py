"""
Detection Module
================

YOLOv11 object detection with SAHI sliced inference for high-resolution
field imagery. Includes GPS extraction from EXIF metadata and COCO-protocol
evaluation metrics.

Key components:
    - trainer: YOLOv11 training configuration and execution
    - inference: SAHI sliced inference for images beyond training resolution
    - gps: EXIF GPS coordinate extraction to GeoDataFrame
    - evaluation: mAP50, mAP50:95, precision, recall metrics
"""

from ragweed_toolkit.detection.trainer import TrainingConfig, train
from ragweed_toolkit.detection.inference import SahiConfig, run_sahi, run_sahi_batch
from ragweed_toolkit.detection.gps import extract_gps, extract_gps_geodataframe
from ragweed_toolkit.detection.evaluation import evaluate

__all__ = [
    "TrainingConfig",
    "train",
    "SahiConfig",
    "run_sahi",
    "run_sahi_batch",
    "extract_gps",
    "extract_gps_geodataframe",
    "evaluate",
]
