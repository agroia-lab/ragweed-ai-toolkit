# Tutorial: Detection Workflow

This tutorial walks through the complete weed detection pipeline: training a YOLOv11 detector, running SAHI sliced inference on field images, extracting GPS coordinates, and exporting results as a shapefile for QGIS visualization.

**Corresponding notebook:** `notebooks/01_detection_workflow.ipynb`

---

## Step 1: Configure Training

The `TrainingConfig` dataclass stores all hyperparameters validated in the chapter (Section 3.2, Table 3). Defaults match the production settings, so you only need to override what you want to change.

```python
from ragweed_toolkit.detection import TrainingConfig, train

config = TrainingConfig(
    model="yolo11l.pt",
    data="path/to/data.yaml",
    epochs=50,
    imgsz=640,
    batch=64,
    device="0,1",  # dual GPU DataParallel
)

result = train(config)
print(f"Best mAP50: {result['metrics']['mAP50']:.3f}")
print(f"Weights: {result['best_weights']}")
```

Key parameters from the chapter:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `model` | `yolo11l.pt` | Best accuracy/speed trade-off |
| `imgsz` | `640` | Matches SAHI slice size |
| `batch` | `64` | Saturates dual RTX 4090 |
| `optimizer` | `SGD` | More stable convergence |
| `patience` | `15` | Early stopping prevents overfitting |

Or use the CLI:

```bash
ragweed-train --data configs/ragweed_v4.yaml --device 0,1
```

## Step 2: Run SAHI Sliced Inference

SAHI divides large images into overlapping tiles, runs detection on each tile, and merges results with NMS. This is critical for dense weed scenes where single-pass inference misses small objects.

```python
from ragweed_toolkit.detection import SahiConfig, run_sahi_batch

config = SahiConfig(
    model_path="weights/best.pt",
    slice_size=640,
    overlap_ratio=0.2,
    confidence_threshold=0.25,
    max_det=10000,  # YOLO default of 300 drops detections in dense fields
)

results = run_sahi_batch(image_paths, config)
# results: dict mapping image_path -> list of detections
```

Each detection is a dictionary:

```python
{
    "class_id": 0,
    "class_name": "AMBEL",
    "bbox": [x1, y1, x2, y2],
    "confidence": 0.87,
}
```

Or use the CLI:

```bash
ragweed-sahi --model weights/best.pt --images field_photos/ --output detections.json
```

## Step 3: Extract GPS Coordinates

Convert detection results into a GeoDataFrame by extracting GPS coordinates from image EXIF metadata.

```python
from ragweed_toolkit.detection import extract_gps_geodataframe

gdf = extract_gps_geodataframe(image_paths)
# gdf columns: image, latitude, longitude, geometry
```

## Step 4: Export to Shapefile

Merge detection counts with GPS coordinates and export for GIS visualization.

```python
# Add detection counts to the GeoDataFrame
gdf["nr_ambel"] = [len(results.get(str(p), [])) for p in image_paths]

# Classify severity for symbology
def classify(count):
    if count == 0: return "Absent"
    elif count <= 5: return "Low"
    elif count <= 10: return "Medium"
    elif count <= 30: return "High"
    else: return "Very High"

gdf["severity"] = gdf["nr_ambel"].apply(classify)

# Export
gdf.to_file("detections.gpkg", driver="GPKG")
```

## Step 5: Visualize in QGIS

Load the exported GeoPackage in QGIS and apply the standard severity symbology:

| Severity | Color | Size | Count Range |
|----------|-------|------|-------------|
| Absent | Gray (#CCCCCC) | 3 mm | 0 |
| Low | Green (#2ECC71) | 4 mm | 1--5 |
| Medium | Yellow (#F1C40F) | 5 mm | 6--10 |
| High | Orange (#E67E22) | 6 mm | 11--30 |
| Very High | Red (#E74C3C) | 8 mm | > 30 |

Or use the `ragweed_toolkit.viz` module for matplotlib plots:

```python
from ragweed_toolkit.viz.style import set_publication_style, ORARA_COLORS

set_publication_style()
# ... plot with ORARA_COLORS for consistent styling
```

## What's Next

- **[Cross-Domain Analysis](crossdomain.md)** -- Assess whether your model will work on new field data
- **[Spatial Mapping](spatial.md)** -- Interpolate detections into density surfaces with kriging
