# CLI Reference

After `pip install ragweed-ai-toolkit`, eight command-line tools are available. Each wraps the Python API with argument parsing and formatted output.

---

## ragweed-train

Train a YOLO model with validated hyperparameters from the chapter (Table 3).

**Module:** `ragweed_toolkit.detection.trainer`

```bash
ragweed-train --data path/to/data.yaml
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | *(required)* | Path to dataset YAML file |
| `--model` | `yolo11l.pt` | Pretrained checkpoint |
| `--epochs` | `50` | Maximum training epochs |
| `--imgsz` | `640` | Input image size in pixels |
| `--batch` | `64` | Batch size |
| `--device` | `0` | CUDA device: `0`, `0,1`, or `cpu` |
| `--optimizer` | `SGD` | Optimizer: SGD, Adam, auto |
| `--lr0` | `0.01` | Initial learning rate |
| `--patience` | `15` | Early stopping patience (epochs) |
| `--project` | `training_results` | Output directory |
| `--name` | *(empty)* | Run name prefix |
| `--seed` | `42` | Random seed |
| `--workers` | `8` | DataLoader workers |
| `--no-cache` | *(flag)* | Disable image caching |

### Example

```bash
ragweed-train \
    --data configs/ragweed_v4.yaml \
    --model yolo11l.pt \
    --epochs 100 \
    --batch 32 \
    --device 0,1 \
    --project training_results/experiment_v2
```

---

## ragweed-sahi

Run SAHI sliced inference on images with a trained YOLO model.

**Module:** `ragweed_toolkit.detection.inference`

```bash
ragweed-sahi --model weights/best.pt --images path/to/images/
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | *(required)* | Path to trained YOLO `.pt` checkpoint |
| `--images` | *(required)* | Image file(s) or directory (accepts multiple) |
| `--slice-size` | `640` | Slice size in pixels |
| `--overlap` | `0.2` | Overlap ratio between slices |
| `--conf` | `0.25` | Confidence threshold |
| `--device` | `cuda:0` | Compute device |
| `--output` | *(none)* | Save results JSON to this path |

### Example

```bash
ragweed-sahi \
    --model weights/best.pt \
    --images field_photos/ \
    --slice-size 1024 \
    --conf 0.3 \
    --output results/detections.json
```

---

## ragweed-embed

Extract CNN embeddings from a directory of images.

**Module:** `ragweed_toolkit.embeddings.extractor`

```bash
ragweed-embed --source images/ --output embeddings.npy
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--source` | *(required)* | Directory containing images |
| `--output` | *(required)* | Output `.npy` file for embeddings |
| `--backbone` | `resnet50` | CNN backbone: `resnet50`, `efficientnet_b0`, `efficientnet_b2` |
| `--batch-size` | `32` | Batch size |
| `--device` | `cuda` | Compute device |
| `--max-images` | *(all)* | Maximum images to process |
| `--workers` | `4` | DataLoader workers |

### Example

```bash
ragweed-embed \
    --source field_images/ \
    --output embeddings/field.npy \
    --backbone resnet50 \
    --batch-size 64
```

---

## ragweed-mmd

Compute MMD between two embedding sets and evaluate deployment readiness.

**Module:** `ragweed_toolkit.embeddings.mmd`

```bash
ragweed-mmd --source source.npy --target target.npy
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--source` | *(required)* | Path to source embeddings `.npy` file |
| `--target` | *(required)* | Path to target embeddings `.npy` file |
| `--permutations` | `1000` | Number of permutations for p-value |
| `--seed` | `42` | Random seed |

### Deployment Gate Tiers

| MMD Range | Tier | Recommendation |
|-----------|------|----------------|
| < 0.15 | Green | Deploy directly |
| 0.15 -- 0.30 | Yellow | Deploy with augmentation |
| 0.30 -- 0.45 | Orange | Active learning required |
| > 0.45 | Red | Collect new training data |

### Example

```bash
ragweed-mmd \
    --source embeddings/chile.npy \
    --target embeddings/spain.npy \
    --permutations 2000
```

---

## ragweed-tile

Tile a GeoTIFF orthomosaic into JPEG tiles for YOLO inference.

**Module:** `ragweed_toolkit.orthomosaic.tiling`

```bash
ragweed-tile --input mosaic.tif --output tiles/
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | *(required)* | Path to input GeoTIFF orthomosaic |
| `--output` | *(required)* | Output directory for tiles |
| `--tile-size` | `1024` | Tile size in pixels |
| `--overlap` | `0` | Overlap in pixels |
| `--min-valid` | `0.9` | Min fraction of valid pixels to keep a tile |
| `--quality` | `95` | JPEG quality (1--100) |

### Example

```bash
ragweed-tile \
    --input orthomosaics/field_20260115.tif \
    --output tiles/field_20260115/ \
    --tile-size 1024 \
    --overlap 128
```

---

## ragweed-kriging

Interpolate point observations using ordinary kriging with an exponential variogram.

**Module:** `ragweed_toolkit.geostatistics.kriging`

```bash
ragweed-kriging \
    --points detections.gpkg \
    --value-col nr_ambel \
    --boundary field.gpkg \
    --output kriged.gpkg
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--points` | *(required)* | GeoPackage/Shapefile with point observations |
| `--value-col` | *(required)* | Column name to interpolate |
| `--boundary` | *(required)* | GeoPackage/Shapefile with study area polygon |
| `--output` | *(required)* | Output GeoPackage path |
| `--resolution` | `5.0` | Grid cell size in metres |
| `--nugget` | `43066.5` | Variogram nugget |
| `--partial-sill` | `43066.5` | Variogram partial sill |
| `--range` | `81.2` | Variogram range in metres |

### Example

```bash
ragweed-kriging \
    --points outputs/detections_linked.gpkg \
    --value-col nr_ambel \
    --boundary boundaries/apalta.gpkg \
    --output outputs/kriged_apalta.gpkg \
    --resolution 5.0
```

---

## ragweed-lisa

Compute bivariate LISA clusters from a spatial dataset.

**Module:** `ragweed_toolkit.spatial.lisa`

```bash
ragweed-lisa --input data.gpkg --var-x PC1 --var-y density
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | *(required)* | GeoPackage/Shapefile with point observations |
| `--var-x` | *(required)* | Column name for the first variable |
| `--var-y` | *(required)* | Column name for the second variable |
| `--output` | *(none)* | Output GeoPackage path (optional) |
| `--threshold` | `15.0` | Distance threshold for spatial weights (metres) |
| `--permutations` | `999` | Number of permutations for p-values |
| `--alpha` | `0.05` | Significance level |

### Cluster Interpretation

| Cluster | Color | Meaning | Management Action |
|---------|-------|---------|-------------------|
| HH | Red | Hot spot | Priority treatment |
| LL | Blue | Cold spot | Monitoring only |
| HL | Orange | Spatial outlier | Investigate |
| LH | Purple | Spatial outlier | Emerging outbreak |
| NS | Gray | Not significant | No action |

### Example

```bash
ragweed-lisa \
    --input presto_weed_data.gpkg \
    --var-x PC1 \
    --var-y AMBEL_density \
    --output lisa_clusters.gpkg \
    --threshold 15 \
    --permutations 999
```

---

## ragweed-indices

Compute spectral indices from a 10-band Sentinel-2 GeoTIFF.

**Module:** `ragweed_toolkit.satellite.indices`

```bash
ragweed-indices --input sentinel2.tif --output-dir indices/
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | *(required)* | Input 10-band GeoTIFF (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12) |
| `--output-dir` | *(required)* | Output directory for index GeoTIFFs |
| `--indices` | *(all)* | Specific indices to compute |

### Available Indices

| Index | Description |
|-------|-------------|
| NDVI | Vegetation density |
| EVI | Atmosphere-corrected vegetation |
| GNDVI | Chlorophyll content |
| RENDVI | Phenological state |
| S2WI | Soil moisture |
| NBR2 | Moisture and crop residue |
| BSI | Bare soil fraction |
| Clay | Clay minerals (SWIR ratio) |
| SWIRd | Surface texture proxy |

### Example

```bash
# Compute all indices
ragweed-indices --input sentinel2_stack.tif --output-dir indices/

# Compute specific indices only
ragweed-indices --input sentinel2_stack.tif --output-dir indices/ --indices NDVI EVI BSI
```
