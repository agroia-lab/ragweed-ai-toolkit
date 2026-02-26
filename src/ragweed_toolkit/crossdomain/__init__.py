"""
Cross-Domain Module
===================

Tools for building multi-source datasets and evaluating cross-domain
generalization of detection models. Implements the two-phase transfer
learning experiment from the chapter.

Key components:
    - dataset_builder: Unify heterogeneous annotation formats (YOLO, VOC, COCO)
    - evaluation: Cross-domain mAP evaluation with per-database breakdown
    - comparison: Multi-model benchmark visualization
"""

__all__ = [
    "build_combined_dataset",
    "build_international_dataset",
    "process_voc_xml",
    "process_yolo_labels",
    "validate_labels",
    "compute_per_database_metrics",
    "count_labels_per_image",
    "load_manifest",
    "run_crossdomain_evaluation",
    "compare_evaluations",
    "extract_overall_metrics",
    "load_eval_results",
]

try:
    from ragweed_toolkit.crossdomain.dataset_builder import (
        build_combined_dataset,
        build_international_dataset,
        process_voc_xml,
        process_yolo_labels,
        validate_labels,
    )
except ImportError:
    pass

try:
    from ragweed_toolkit.crossdomain.evaluation import (
        compute_per_database_metrics,
        count_labels_per_image,
        load_manifest,
        run_crossdomain_evaluation,
    )
except ImportError:
    pass

try:
    from ragweed_toolkit.crossdomain.comparison import (
        compare_evaluations,
        extract_overall_metrics,
        load_eval_results,
    )
except ImportError:
    pass
