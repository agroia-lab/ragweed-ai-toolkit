"""
Orthomosaic Module
==================

Processing pipeline for georeferenced drone orthomosaics: tiling for
YOLO inference, converting detections to GeoPackage point layers, and
uncertainty-based active learning for label mining.

Key components:
    - tiling: Split large orthomosaics into inference-ready tiles
    - geo_export: Convert pixel-space detections to georeferenced points
    - active_learning: Mine uncertain predictions for human review

Requires: pip install ragweed-ai-toolkit[orthomosaic]
"""

__all__ = [
    "tile_orthomosaic",
    "compute_detection_summary",
    "detections_to_geopackage",
    "load_tile_manifest",
    "export_uncertain_panels",
    "export_yolo_labels",
    "generate_uncertainty_report",
    "score_images",
]

try:
    from ragweed_toolkit.orthomosaic.tiling import tile_orthomosaic
except ImportError:
    pass

try:
    from ragweed_toolkit.orthomosaic.geo_export import (
        compute_detection_summary,
        detections_to_geopackage,
        load_tile_manifest,
    )
except ImportError:
    pass

try:
    from ragweed_toolkit.orthomosaic.active_learning import (
        export_uncertain_panels,
        export_yolo_labels,
        generate_uncertainty_report,
        score_images,
    )
except ImportError:
    pass
