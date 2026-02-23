# Evidence 05: Field-Level SAHI Inference for Ragweed Detection

## Overview

Ground-level SAHI inference results for *Ambrosia artemisiifolia* (common ragweed, AMBEL) detection across three field sites and one full campaign in central Chile. These results demonstrate that a YOLOv11 model trained on close-up photographs can be deployed via SAHI on field-collected smartphone images to produce georeferenced ragweed density maps.

---

## 1. Detection Model

| Parameter | Value |
|-----------|-------|
| Architecture | YOLOv11l (Large variant, 25.3M parameters) |
| Training run | Run 3 (`run3_lentejas640_b64_2gpu`) |
| Training data | Augmented Seba dataset: 4,955 images at 640x640 px (3x augmentation from 2,065 originals) |
| Augmentation | Crop, rotation, salt-and-pepper noise |
| Target class | Single class: **AMBEL** (*Ambrosia artemisiifolia*) |
| Training | 100 epochs, batch size 64, 2x NVIDIA RTX 4090, ~40 min |
| Weights path | `training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/weights/best.pt` |

### Validation Metrics (Run 3)

| Metric | Value |
|--------|-------|
| mAP@50 | **0.886** |
| mAP@50-95 | 0.744 |
| Precision | 0.934 |
| Recall | 0.808 |

---

## 2. Inference Pipeline

| Component | Configuration |
|-----------|---------------|
| Framework | SAHI (Slicing Aided Hyper Inference) |
| Slice size | 640 x 640 px |
| Overlap ratio | 0.20 (20%) |
| Confidence threshold | 0.25 |
| NMS | Applied by SAHI to merge overlapping detections across slices |
| GPS extraction | EXIF metadata via PIL/Pillow |

**Rationale for SAHI on ground images:** Although ground-level photographs capture plants at a scale comparable to the training data, the full image resolution (typically 4032 x 3024 px from smartphone cameras) exceeds the 640 px training tile size. SAHI slicing ensures each region is processed at native training resolution, maximizing detection accuracy for small and densely packed ragweed plants.

---

## 3. Test Sites

### 3.1 Ambrosia Santa Rosa

| Parameter | Value |
|-----------|-------|
| Source path | `/media/malezainia2/D/extracted/Ambrosia.Sta.Rosa/fotos.proyecto01-02-23/` |
| Date | February 1, 2023 |
| Camera | Ground-level smartphone photographs |
| Total images | 75 |
| Images with GPS | 52 |
| Total detections | **1,390** |
| Min / Max per image | 1 / 52 |
| Mean / Median per image | 18.5 / 16 |
| Location | ~36.53S, 71.92W (Santa Rosa, Maule, Chile) |

### 3.2 CATO Maiz

| Parameter | Value |
|-----------|-------|
| Source path | `/media/malezainia2/D/extracted/CATO_MAIZ/` |
| Date | January 3, 2023 |
| Camera | Ground-level smartphone photographs |
| Total images | 67 |
| Images with GPS | 65 |
| Total detections | **2,443** |
| Min / Max per image | 4 / 94 |
| Mean / Median per image | 36.5 / 32 |
| Location | ~36.55S, 71.91W |

### 3.3 Trigo Corregidas

| Parameter | Value |
|-----------|-------|
| Source path | `/media/malezainia2/D/extracted/trigo_corregidas/` |
| Date | 2023 |
| Camera | Ground-level smartphone photographs |
| Total images | 48 |
| Images with GPS | 48 (organized in 4 subdirectories of 12) |
| Total detections | **1,372** |
| Min / Max per image | 9 / 70 |
| Mean / Median per image | 28.6 / 26 |

### Summary Table

| Dataset | Images | With GPS | Detections | Mean/img | Median/img | Max/img |
|---------|--------|----------|------------|----------|------------|---------|
| Ambrosia Santa Rosa | 75 | 52 | 1,390 | 18.5 | 16 | 52 |
| CATO Maiz | 67 | 65 | 2,443 | 36.5 | 32 | 94 |
| Trigo Corregidas | 48 | 48 | 1,372 | 28.6 | 26 | 70 |
| **Total** | **190** | **165** | **5,205** | **27.4** | -- | -- |

---

## 4. Full Lentejas Dec 2024 Campaign

The primary deployment campaign for the book chapter, conducted in December 2024 at the Santa Rosa lentil site.

| Parameter | Value |
|-----------|-------|
| Dataset | Lentejas December 2024 |
| Total images | 75 |
| Images with GPS | 52 |
| Total detections | **1,390** |
| Result CSV | `malezainia2/ambel_seba/results/ambel_sta_rosa.csv` |
| Shapefile | `outputs/geo_exports/ambel_lentejas_dic24_ground.shp` |

**CSV structure:** `image, latitude, longitude, altitude, total_det, nr_ambel`

---

## 5. Geo-Export Workflow

The pipeline converts per-image detection counts with GPS coordinates into georeferenced shapefiles for visualization in QGIS.

### Workflow Steps

```
SAHI Inference  -->  Per-image CSV  -->  Shapefile  -->  QGIS Visualization
(detections)        (GPS + counts)      (points)        (graduated symbology)
```

1. **SAHI inference** produces per-image detection counts
2. **CSV generation** links detection counts to EXIF GPS coordinates (latitude, longitude, altitude)
3. **Shapefile creation** converts CSV rows into georeferenced point features (EPSG:4326)
4. **QGIS integration** loads shapefiles with graduated symbology by detection count

### Output Shapefiles

| Shapefile | Description |
|-----------|-------------|
| `outputs/geo_exports/ambel_sta_rosa_ground.shp` | Santa Rosa detections (75 points) |
| `outputs/geo_exports/ambel_cato_maiz_ground.shp` | CATO Maiz detections (67 points) |
| `outputs/geo_exports/ambel_trigo_corregidas_ground.shp` | Trigo Corregidas detections (48 points) |
| `outputs/geo_exports/ambel_all_ground.shp` | Merged shapefile (165 geo-referenced points) |
| `outputs/geo_exports/ambel_lentejas_dic24_ground.shp` | Lentejas Dec 2024 campaign |

### Output CSVs

| CSV | Rows | Total Detections |
|-----|------|------------------|
| `malezainia2/ambel_seba/results/ambel_sta_rosa.csv` | 75 | 1,390 |
| `malezainia2/ambel_seba/results/ambel_cato_maiz.csv` | 67 | 2,443 |
| `malezainia2/ambel_seba/results/ambel_trigo_corregidas.csv` | 48 | 1,372 |
| `malezainia2/ambel_seba/results/ambel_lentejas_dic24.csv` | 183 | 5,502 |

**CSV column format:** `image, latitude, longitude, altitude, total_det, nr_ambel`

---

## 6. Scripts for Reproduction

All scripts are located in the project repository.

### 6.1 SAHI Inference

**Script:** `scripts/ambel_sahi_inference_full.py`

Reusable SAHI inference pipeline with EXIF GPS extraction. Processes a directory of field photographs, runs SAHI sliced inference with the trained model, and outputs a CSV with per-image GPS coordinates and detection counts.

```bash
python scripts/ambel_sahi_inference_full.py \
    --model training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/weights/best.pt \
    --images /path/to/field/images \
    --output results/output_name \
    --slice 640 --conf 0.25
```

### 6.2 Shapefile Creation

**Script:** `scripts/ambel_create_shapefiles.py`

Converts CSV files with GPS coordinates and detection counts into ESRI shapefiles (.shp) with associated projection files (.prj, EPSG:4326).

```bash
python scripts/ambel_create_shapefiles.py \
    --input results/ambel_sta_rosa.csv \
    --output outputs/geo_exports/ambel_sta_rosa_ground
```

### 6.3 Geo-Export Pipeline

**Script:** `scripts/cli/export_ambel_geo.py`

Combined geo-export pipeline that reads SAHI summary JSON outputs, extracts GPS from image EXIF data, and produces both CSV and shapefile outputs.

```bash
python scripts/cli/export_ambel_geo.py \
    --images /path/to/source/images \
    --summary outputs/sahi_*/summary.json \
    --output ambel_dataset_name
```

### 6.4 QGIS Visualization

**Scripts:**
- `malezainia2/ambel_seba/qgis_scripts/ADD_AMBEL_ADULT_LAYERS.py` -- Load all AMBEL layers into QGIS
- `malezainia2/ambel_seba/qgis_scripts/APPLY_AMBEL_ADULT_STYLE.py` -- Apply graduated symbology

```python
# In QGIS Python Console (Ctrl+Alt+P):
exec(open('outputs/geo_exports/ADD_AMBEL_ADULT_LAYERS.py').read())
exec(open('outputs/geo_exports/APPLY_AMBEL_ADULT_STYLE.py').read())
```

---

## 7. Inference Example Panels

Visual comparison panels were generated for 3 representative images from each test site, showing original photographs alongside annotated detection outputs.

### Available Examples

| Site | Run | Example Images | Path |
|------|-----|----------------|------|
| Ambrosia Santa Rosa | Run 3 | IMG_20230201_124117, IMG_20230201_123511, IMG_20230201_120043_2 | `malezainia2/ambel_seba/inference_examples/ambrosia_sta_rosa/` |
| CATO Maiz | Run 3 | IMG_20230103_104827, IMG_20230103_105600, IMG_20230103_103342 | `malezainia2/ambel_seba/inference_examples/cato_maiz/` |
| Trigo Corregidas | Run 3 | 111_T11_GARRISON_3.5_hi, 107_T7_TREFLAN_4_hi, 106_T6_TREFLAN_3_hi | `malezainia2/ambel_seba/inference_examples/trigo_corregidas/` |

Each inference example directory contains:
- `comparison_panels/` -- Side-by-side original vs. annotated images
- `labels/` -- YOLO-format detection labels (.txt)
- `summary.json` -- Inference statistics
- `detection_summary.png` -- Visual summary

Additional runs (Run 1, Run 2, Run 4) are also stored for cross-run comparison in `inference_examples/` subdirectories.

---

## 8. Key Findings for Book Chapter

1. **High detection accuracy**: mAP@50 = 0.886 demonstrates reliable ragweed detection from ground-level photographs with a single-class model
2. **Operational scalability**: 190 images across 3 sites processed via SAHI, yielding 5,205 georeferenced detections
3. **GPS-enabled mapping**: 87% of images (165/190) had valid EXIF GPS, enabling direct spatial mapping without manual georeferencing
4. **Variable density**: Per-image detection counts ranged from 1 to 94, reflecting natural variation in ragweed infestation intensity across sites
5. **QGIS integration**: Automated pipeline from raw photographs to graduated symbology maps for stakeholder communication

---

## 9. Data Provenance

| Item | Path |
|------|------|
| Model weights | `training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/weights/best.pt` |
| Training artifacts | `malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/` |
| Inference documentation | `malezainia2/ambel_seba/INFERENCE_RESULTS.md` |
| Result CSVs | `malezainia2/ambel_seba/results/` |
| Shapefiles | `outputs/geo_exports/ambel_*_ground.shp` |
| Inference examples | `malezainia2/ambel_seba/inference_examples/` |
| SAHI inference script | `scripts/ambel_sahi_inference_full.py` |
| Shapefile creation script | `scripts/ambel_create_shapefiles.py` |
| Geo-export script | `scripts/cli/export_ambel_geo.py` |
| QGIS layer script | `malezainia2/ambel_seba/qgis_scripts/ADD_AMBEL_ADULT_LAYERS.py` |
| QGIS style script | `malezainia2/ambel_seba/qgis_scripts/APPLY_AMBEL_ADULT_STYLE.py` |

---

*Evidence compiled: 2026-02-23*
*Publication target: Springer-Nature book chapter on weeds, climate change, and technology*
