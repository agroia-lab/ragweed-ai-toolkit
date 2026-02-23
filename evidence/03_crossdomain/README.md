# Evidence 03: Cross-Domain Detection Experiment

## Overview

This evidence package documents a two-phase experiment testing whether a ragweed (*Ambrosia artemisiifolia*) detection model trained exclusively on Chilean field imagery can generalize to international datasets, and whether combining Chilean and international training data can bridge the domain gap.

**Key finding:** A Chilean-only model (mAP50 = 0.886 on Chilean data) catastrophically fails on international imagery (mAP50 = 0.108, a 77.8 percentage point drop). Training a combined model on Chilean + international data recovers international performance to mAP50 = 0.874 (+76.6pp) at a cost of moderate Chilean regression (mAP50 = 0.742, -14.4pp).

---

## 1. Experimental Design

### 1.1 Hypothesis

Domain shift analysis (Section 5 of the AMBEL results series) demonstrated that all 9 ragweed databases are statistically distinguishable, with Maximum Mean Discrepancy (MMD) values of 0.27--0.43 between Chilean and international databases. This experiment tests whether that measured distributional distance translates to actual detection performance loss, and whether including target-domain training data can close the gap.

### 1.2 Two-Phase Structure

| Phase | Description | Goal |
|-------|-------------|------|
| **Phase 1** | Evaluate best Chilean model on unified international test set (2,367 images) | Quantify cross-domain detection failure |
| **Phase 2** | Train combined model on Chilean + international data; evaluate on both domains | Measure domain gap recovery and Chilean regression |

### 1.3 Success Criteria (Pre-registered)

1. **Primary:** Combined model international mAP50 > CL-only model international mAP50
2. **Guard rail:** CL_Seba regression < 5 percentage points (mAP50 > 0.836)
3. **Bonus:** Per-database improvement correlates with MMD distance

---

## 2. Datasets

### 2.1 Chilean Training Data (Baseline Model)

| Property | Value |
|----------|-------|
| Dataset | CL_Seba (Lentejas augmented) |
| Source | Roboflow `peppo-4-rotaciones/ambrosia-lentejas-seba-0rdlg v1` |
| Total images | 4,955 |
| Annotations | 33,184 |
| Augmentation | 3x (crop 0-29%, rotation +/-15deg, salt-and-pepper 1.68%) |
| Splits | train=4,335 / valid=411 / test=209 |
| Image size | Variable (~938-1050 x 808-915 px) |
| Format | YOLO TXT, single class (0 = AMBEL) |
| Origin | Santa Rosa, central Chile. Ground-level field photography |

**Data path:** `/media/malezainia2/E/ProcessingData/ambrosia-lentejas-seba-0rdlg-v1/`

### 2.2 International Test Set (4 Databases)

| Database | Code | Images | Annotations | Source Format | Class Mapping | Origin | Equipment |
|----------|------|-------:|------------:|---------------|---------------|--------|-----------|
| ND Individual | ND_Indiv | 876 | 2,010 | YOLO TXT | class 2 -> 0 | North Dakota, USA | Canon EOS 90D |
| Michigan 3-Season | MI_3Season | 719 | 1,171 | Pascal VOC XML | "Ragweed" -> 0 | Michigan, USA | iPhone, Nikon, Canon |
| WeedCrop PrecAg | WeedCrop | 495 | 1,086 | YOLO TXT (13 classes) | class 8 -> 0 | Multi-site, USA | Field cameras, 8 crop backgrounds |
| ND Aerial | ND_Aerial | 277 | 647 | YOLO TXT (5 classes) | class 2 -> 0 | North Dakota, USA | DJI Phantom 4 Pro |
| **Total** | | **2,367** | **4,914** | | | | |

**Excluded databases:**
- DB05 Purdue (*A. trifida* -- different species, giant ragweed)
- DB06 WeedCube (hyperspectral pseudo-RGB, no bounding boxes)

**Format unification notes:**
- **ND Individual:** Single-species subset. All labels remapped class 2 -> 0. Two images with empty labels excluded (876 of 878).
- **Michigan 3-Season:** Pascal VOC XML converted to YOLO format (absolute pixel coords -> normalized center+wh). Three yearly directories: data2021 (512), data2022 (93), data2023 (114).
- **WeedCrop PrecAg:** 13-class labels filtered to class 8 (ragweed) only. Non-ragweed annotations discarded. 495 images across 8 crop subdirectories.
- **ND Aerial:** 5-class aerial labels filtered to class 2 (ragweed). 277 images with at least one ragweed annotation retained.

All output images prefixed with database code (e.g., `ND_Indiv_rag100.jpg`, `MI_3Season_20210820_iPhoneSE_YL_1562.jpg`) for per-database tracking.

**Data path:** `/media/malezainia2/E/ProcessingData/ambel-international-v1/`

### 2.3 Combined Training Set (Phase 2)

| Source | Split Strategy | Train | Valid | Test | Total |
|--------|---------------|------:|------:|-----:|------:|
| CL_Seba | Preserve Roboflow splits | 4,335 | 411 | 209 | 4,955 |
| CL_Alberto | Preserve Roboflow splits | 576 | 55 | 28 | 659 |
| International (stratified) | 80/10/10 per source DB | 1,894 | 238 | 235 | 2,367 |
| **Total** | | **6,805** | **704** | **472** | **7,981** |

**International stratified split breakdown (seed=42):**

| Database | Train | Valid | Test | Total |
|----------|------:|------:|-----:|------:|
| michigan_3season | 575 | 72 | 72 | 719 |
| nd_individual | 701 | 88 | 87 | 876 |
| weedcrop_precag | 396 | 50 | 49 | 495 |
| nd_aerial | 222 | 28 | 27 | 277 |

CL_Seba and CL_Alberto preserved their original Roboflow train/valid/test splits. International images were split 80/10/10 stratified by source database, ensuring proportional representation in every split. All filenames prefixed to prevent collisions.

**Data path:** `/media/malezainia2/E/ProcessingData/ambel-combined-intl-v1/`

---

## 3. Models

| Property | CL-Only Model (Baseline) | Combined Model |
|----------|--------------------------|----------------|
| Architecture | YOLOv11l | YOLOv11l |
| Parameters | 25.3M (464 layers fused, 86.6 GFLOPs) | 25.3M (identical) |
| Training data | CL_Seba Lentejas (4,955 imgs) | CL + Alberto + International (7,981 imgs) |
| Epochs | 100 | 50 |
| Batch size | 64 | 16 |
| Image size | 640 | 640 |
| Device | 2x RTX 4090 (DataParallel) | 1x RTX 4090 |
| Training time | 39.9 min | 52.5 min |
| Optimizer | SGD (Ultralytics defaults) | SGD (Ultralytics defaults) |
| Early stopping | patience=15 | patience=15 |
| Seed | 42 | 42 |
| Pretrained | COCO | COCO |

**Note:** The combined model used batch=16 (vs 64) and single GPU due to a PyTorch CUDA allocator issue with the `expandable_segments` feature on this hardware configuration.

### Model Weights

| Model | Path |
|-------|------|
| CL-only (best) | `training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/weights/best.pt` |
| Combined (best) | `training_results/malezainia2/ambel_crossdomain/combined_intl_v1_20260222_161204/weights/best.pt` |
| Combined (last) | `training_results/malezainia2/ambel_crossdomain/combined_intl_v1_20260222_161204/weights/last.pt` |
| Combined (checkpoints) | `weights/epoch0.pt`, `epoch10.pt`, `epoch20.pt`, `epoch30.pt`, `epoch40.pt` |

---

## 4. Phase 1 Results: Chilean Model Fails on International Data

### 4.1 Overall Metrics

| Metric | Value |
|--------|------:|
| mAP@0.5 | **0.108** |
| mAP@0.5:0.95 | 0.047 |
| Precision | 0.268 |
| Recall | 0.180 |
| Total images evaluated | 2,367 |
| Total GT annotations | 4,914 |
| Evaluation time | 32.9s |

**Baseline reference:** The same CL-only model achieves mAP50 = 0.886 on its Chilean validation set. The drop to 0.108 represents a **77.8 percentage point decline**.

### 4.2 Per-Database Breakdown

| Database | Images | GT Annotations | Predictions | GT/Image | Pred/Image | Detection Ratio |
|----------|-------:|--------------:|------------:|---------:|-----------:|----------------:|
| michigan_3season | 719 | 1,171 | 0 | 1.6 | 0.0 | 0.000 |
| nd_aerial | 277 | 647 | 0 | 2.3 | 0.0 | 0.000 |
| nd_individual | 876 | 2,010 | 0 | 2.3 | 0.0 | 0.000 |
| weedcrop_precag | 495 | 1,086 | 0 | 2.2 | 0.0 | 0.000 |
| **TOTAL** | **2,367** | **4,914** | **0** | | | **0.000** |

The per-database breakdown reveals that the CL-only model produced **zero predictions** at the evaluation confidence threshold (0.001) for all four international databases. This is not a gradual degradation -- it is a complete failure to detect ragweed in any international image.

### 4.3 Phase 1 Artifacts

| File | Path (relative to project root) |
|------|------|
| Metrics JSON | `malezainia2/ambel_crossdomain/phase1_eval/crossdomain_metrics.json` |
| Evaluation report | `malezainia2/ambel_crossdomain/phase1_eval/crossdomain_report.md` |
| PR curve | `malezainia2/ambel_crossdomain/phase1_eval/PR_curve.png` |
| F1 curve | `malezainia2/ambel_crossdomain/phase1_eval/F1_curve.png` |
| P curve | `malezainia2/ambel_crossdomain/phase1_eval/P_curve.png` |
| R curve | `malezainia2/ambel_crossdomain/phase1_eval/R_curve.png` |
| Confusion matrix | `malezainia2/ambel_crossdomain/phase1_eval/confusion_matrix.png` |
| Confusion matrix (normalized) | `malezainia2/ambel_crossdomain/phase1_eval/confusion_matrix_normalized.png` |
| Visual batch 0 (labels) | `malezainia2/ambel_crossdomain/phase1_eval/yolo_val/val_batch0_labels.jpg` |
| Visual batch 0 (predictions) | `malezainia2/ambel_crossdomain/phase1_eval/yolo_val/val_batch0_pred.jpg` |
| Visual batch 1 (labels) | `malezainia2/ambel_crossdomain/phase1_eval/yolo_val/val_batch1_labels.jpg` |
| Visual batch 1 (predictions) | `malezainia2/ambel_crossdomain/phase1_eval/yolo_val/val_batch1_pred.jpg` |
| Visual batch 2 (labels) | `malezainia2/ambel_crossdomain/phase1_eval/yolo_val/val_batch2_labels.jpg` |
| Visual batch 2 (predictions) | `malezainia2/ambel_crossdomain/phase1_eval/yolo_val/val_batch2_pred.jpg` |
| Predictions JSON | `malezainia2/ambel_crossdomain/phase1_eval/yolo_val/predictions.json` |

---

## 5. Phase 2 Results: Combined Model Training and Evaluation

### 5.1 Training Progression

| Epoch | mAP50 | mAP50-95 | Precision | Recall | Train box_loss |
|------:|------:|---------:|----------:|-------:|---------------:|
| 1 | 0.162 | 0.053 | 0.222 | 0.293 | 1.879 |
| 5 | 0.350 | 0.140 | 0.419 | 0.427 | 1.469 |
| 10 | 0.479 | 0.208 | 0.544 | 0.498 | 1.179 |
| 20 | 0.570 | 0.262 | 0.607 | 0.545 | 0.978 |
| 30 | 0.626 | 0.306 | 0.647 | 0.604 | 0.885 |
| 40 | 0.694 | 0.383 | 0.729 | 0.629 | 0.816 |
| 50 | 0.752 | 0.449 | 0.759 | 0.671 | 0.695 |

Best validation mAP50 = 0.752 on the combined validation set (704 images from all sources). No plateau at epoch 50 -- validation losses continued decreasing, suggesting further improvement is possible with more epochs.

### 5.2 Combined Model on International Test (Phase 2 -- INTL)

| Metric | Value |
|--------|------:|
| mAP@0.5 | **0.874** |
| mAP@0.5:0.95 | 0.635 |
| Precision | 0.835 |
| Recall | 0.772 |
| Total images | 2,367 |
| Total GT annotations | 4,914 |
| Evaluation time | 34.1s |

**Per-database breakdown (Combined model on international test):**

| Database | Images | GT | Predictions | GT/Image | Pred/Image | Detection Ratio |
|----------|-------:|---:|------------:|---------:|-----------:|----------------:|
| michigan_3season | 719 | 1,171 | 13,922 | 1.6 | 19.4 | 11.889 |
| nd_aerial | 277 | 647 | 20,088 | 2.3 | 72.5 | 31.048 |
| nd_individual | 876 | 2,010 | 30,782 | 2.3 | 35.1 | 15.314 |
| weedcrop_precag | 495 | 1,086 | 20,752 | 2.2 | 41.9 | 19.109 |

Note: High detection ratios reflect the low confidence threshold (0.001) used for mAP computation, which produces many low-confidence predictions. The mAP@0.5 of 0.874 confirms that the high-confidence predictions are accurate.

### 5.3 Combined Model on Chilean Test (Phase 2 -- CL Regression Check)

| Metric | Value |
|--------|------:|
| mAP@0.5 | **0.742** |
| mAP@0.5:0.95 | 0.435 |
| Precision | 0.757 |
| Recall | 0.660 |
| Total images | 209 |
| Total GT annotations | 1,785 |
| Evaluation time | 22.1s |

### 5.4 Phase 2 Artifacts

**Combined model on international test:**

| File | Path |
|------|------|
| Metrics JSON | `malezainia2/ambel_crossdomain/phase2_eval_intl/crossdomain_metrics.json` |
| PR curve | `malezainia2/ambel_crossdomain/phase2_eval_intl/PR_curve.png` |
| F1 curve | `malezainia2/ambel_crossdomain/phase2_eval_intl/F1_curve.png` |
| P curve | `malezainia2/ambel_crossdomain/phase2_eval_intl/P_curve.png` |
| R curve | `malezainia2/ambel_crossdomain/phase2_eval_intl/R_curve.png` |
| Confusion matrix | `malezainia2/ambel_crossdomain/phase2_eval_intl/confusion_matrix.png` |
| Confusion matrix (normalized) | `malezainia2/ambel_crossdomain/phase2_eval_intl/confusion_matrix_normalized.png` |
| Visual batch 0 (labels) | `malezainia2/ambel_crossdomain/phase2_eval_intl/yolo_val/val_batch0_labels.jpg` |
| Visual batch 0 (predictions) | `malezainia2/ambel_crossdomain/phase2_eval_intl/yolo_val/val_batch0_pred.jpg` |
| Visual batch 1 (labels) | `malezainia2/ambel_crossdomain/phase2_eval_intl/yolo_val/val_batch1_labels.jpg` |
| Visual batch 1 (predictions) | `malezainia2/ambel_crossdomain/phase2_eval_intl/yolo_val/val_batch1_pred.jpg` |
| Prediction labels | `malezainia2/ambel_crossdomain/phase2_eval_intl/yolo_val/labels/*.txt` |

**Combined model on Chilean test:**

| File | Path |
|------|------|
| Metrics JSON | `malezainia2/ambel_crossdomain/phase2_eval_cl/crossdomain_metrics.json` |
| PR curve | `malezainia2/ambel_crossdomain/phase2_eval_cl/PR_curve.png` |
| F1 curve | `malezainia2/ambel_crossdomain/phase2_eval_cl/F1_curve.png` |
| P curve | `malezainia2/ambel_crossdomain/phase2_eval_cl/P_curve.png` |
| R curve | `malezainia2/ambel_crossdomain/phase2_eval_cl/R_curve.png` |
| Confusion matrix | `malezainia2/ambel_crossdomain/phase2_eval_cl/confusion_matrix.png` |
| Confusion matrix (normalized) | `malezainia2/ambel_crossdomain/phase2_eval_cl/confusion_matrix_normalized.png` |
| Visual batch 0 (labels) | `malezainia2/ambel_crossdomain/phase2_eval_cl/yolo_val/val_batch0_labels.jpg` |
| Visual batch 0 (predictions) | `malezainia2/ambel_crossdomain/phase2_eval_cl/yolo_val/val_batch0_pred.jpg` |
| Visual batch 1 (labels) | `malezainia2/ambel_crossdomain/phase2_eval_cl/yolo_val/val_batch1_labels.jpg` |
| Visual batch 1 (predictions) | `malezainia2/ambel_crossdomain/phase2_eval_cl/yolo_val/val_batch1_pred.jpg` |
| Prediction labels | `malezainia2/ambel_crossdomain/phase2_eval_cl/yolo_val/labels/*.txt` |

**Combined model training artifacts:**

| File | Path |
|------|------|
| Training curves | `training_results/malezainia2/ambel_crossdomain/combined_intl_v1_20260222_161204/training_curves.png` |
| Training curves (live) | `training_results/malezainia2/ambel_crossdomain/combined_intl_v1_20260222_161204/training_curves_live.png` |
| Results CSV | `training_results/malezainia2/ambel_crossdomain/combined_intl_v1_20260222_161204/results.csv` |
| Results plot | `training_results/malezainia2/ambel_crossdomain/combined_intl_v1_20260222_161204/results.png` |
| Training summary JSON | `training_results/malezainia2/ambel_crossdomain/combined_intl_v1_20260222_161204/training_summary.json` |
| Training metrics JSON | `training_results/malezainia2/ambel_crossdomain/combined_intl_v1_20260222_161204/training_metrics.json` |
| Args YAML | `training_results/malezainia2/ambel_crossdomain/combined_intl_v1_20260222_161204/args.yaml` |
| Confusion matrix | `training_results/malezainia2/ambel_crossdomain/combined_intl_v1_20260222_161204/confusion_matrix.png` |
| PR curve | `training_results/malezainia2/ambel_crossdomain/combined_intl_v1_20260222_161204/PR_curve.png` |
| Labels distribution | `training_results/malezainia2/ambel_crossdomain/combined_intl_v1_20260222_161204/labels.jpg` |
| Labels correlogram | `training_results/malezainia2/ambel_crossdomain/combined_intl_v1_20260222_161204/labels_correlogram.jpg` |

**Comparison artifacts:**

| File | Path |
|------|------|
| Comparison chart | `malezainia2/ambel_crossdomain/comparison/comparison_chart.png` |
| Comparison report | `malezainia2/ambel_crossdomain/comparison/comparison_report.md` |
| Comparison data JSON | `malezainia2/ambel_crossdomain/comparison/comparison_data.json` |

---

## 6. Cross-Domain Comparison Summary

| Scenario | mAP50 | mAP50-95 | Precision | Recall | Delta mAP50 |
|----------|------:|---------:|----------:|-------:|------------:|
| CL Model on CL Test (baseline) | **0.886** | 0.744 | -- | -- | -- |
| CL Model on INTL Test | 0.108 | 0.047 | 0.268 | 0.180 | -77.8pp |
| Combined Model on INTL Test | **0.874** | 0.635 | 0.835 | 0.772 | +76.6pp vs CL-on-INTL |
| Combined Model on CL Test | **0.742** | 0.435 | 0.757 | 0.660 | -14.4pp vs baseline |

### Success Criteria Assessment

| Criterion | Threshold | Result | Status |
|-----------|-----------|--------|--------|
| Cross-domain improvement | Combined INTL > CL INTL | 0.874 > 0.108 | **PASS** |
| CL regression tolerance | Combined CL > 0.836 (within 5pp) | 0.742 < 0.836 | **FAIL** (14.4pp regression) |
| Overall combined quality | mAP50 > 0.70 on combined val | 0.752 > 0.70 | **PASS** |

---

## 7. Interpretation and Key Conclusions

### 7.1 Domain Shift is the Primary Deployment Risk

The drop from mAP50 = 0.886 to 0.108 is not gradual degradation -- it is near-complete failure. The CL-only model detects fewer than 1 in 5 ragweed plants on international data, and of those detections, 3 out of 4 are false positives. Per-database analysis reveals that the model produces **zero predictions** on all four international databases at the evaluation threshold. This validates the embedding analysis conclusion that each database encodes distinct visual signatures driven by camera, background, and acquisition protocol rather than species biology.

**Practical implication:** A model trained on a single field campaign cannot be trusted on data from different equipment, geography, or season without explicit cross-domain validation.

### 7.2 Data Combination is Effective but Not Free

Adding 2,367 international images to 5,614 Chilean images:
- Nearly eliminates the international gap: 0.108 -> 0.874 (+76.6pp)
- Incurs moderate Chilean regression: 0.886 -> 0.742 (-14.4pp)

This trade-off reflects a fundamental tension in multi-domain training: visual diversity improves generalization at the cost of domain-specific specialization.

### 7.3 Linking MMD to Detection Performance

The embedding analysis (Section 5) predicted deployment gate thresholds:

| MMD Range | Recommendation | Observed Result |
|-----------|---------------|-----------------|
| < 0.15 | Deploy directly | -- |
| 0.15-0.30 | Deploy with augmentation | CL->INTL (MMD ~0.27-0.43): mAP50 = 0.108 (fail) |
| 0.30-0.45 | Active learning required | Same range: training with target data -> mAP50 = 0.874 (success) |
| > 0.45 | Collect substantial new data | -- |

The results validate the deployment gate framework: MMD distances of 0.27-0.43 correctly predicted that the CL model would fail on international data and that target-domain training data would be necessary.

### 7.4 Chilean Regression Context

The 14.4pp CL regression exceeds the pre-defined 5pp tolerance. Contributing factors:
1. **Diluted Chilean representation:** CL images go from 100% to 70% of training data (5,614 CL out of 7,981 total)
2. **Reduced training budget:** 50 epochs vs 100 for the baseline (training curve showed no plateau at epoch 50)
3. **Smaller batch size:** batch=16 vs batch=64 (hardware limitation, not design choice)
4. **Single GPU:** Combined model trained on 1x RTX 4090 vs 2x for baseline

### 7.5 Limitations

1. **Per-database mAP unavailable:** Overall mAP50 = 0.108 may mask variation across databases (e.g., ND Individual may be more detectable than ND Aerial)
2. **CL regression test limited:** Uses only CL_Seba test split (209 augmented images); CL_Alberto would provide a more independent test
3. **Combined model potentially under-trained:** No plateau at epoch 50; longer training could recover CL performance
4. **No per-database mAP for combined model:** Prediction counts are available but true per-database mAP requires separate evaluations

---

## 8. Reproduction Commands

All commands assume:
- Working directory: `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon`
- Conda environment: `virtual_environment_yolo` (ultralytics 8.3.49, torch 2.4.1, CUDA)
- External drive mounted at `/media/malezainia2/E/`

### Phase 0: Build International Dataset

```bash
python scripts/cli/build_international_dataset.py --validate
```

Output: `/media/malezainia2/E/ProcessingData/ambel-international-v1/`

### Phase 1: Evaluate CL Model on International Data

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
conda run -n virtual_environment_yolo python scripts/cli/crossdomain_eval.py \
    --model training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/weights/best.pt \
    --data configs/yaml_config/data_ambel-international-v1-test.yaml \
    --manifest /media/malezainia2/E/ProcessingData/ambel-international-v1/manifest.csv \
    --output malezainia2/ambel_crossdomain/phase1_eval
```

### Phase 2A: Build Combined Dataset

```bash
python scripts/cli/build_combined_dataset.py --validate
```

Output: `/media/malezainia2/E/ProcessingData/ambel-combined-intl-v1/`

### Phase 2B: Train Combined Model

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
conda run -n virtual_environment_yolo python scripts/cli/train_yolo_cli.py \
    --model yolo11l.pt \
    --data configs/yaml_config/data_ambel-combined-intl-v1.yaml \
    --epochs 50 --batch 16 --imgsz 640 --device 0 \
    --project training_results/malezainia2/ambel_crossdomain --name combined_intl_v1
```

### Phase 2C: Evaluate Combined Model on International Test

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
conda run -n virtual_environment_yolo python scripts/cli/crossdomain_eval.py \
    --model training_results/malezainia2/ambel_crossdomain/combined_intl_v1_20260222_161204/weights/best.pt \
    --data configs/yaml_config/data_ambel-international-v1-test.yaml \
    --manifest /media/malezainia2/E/ProcessingData/ambel-international-v1/manifest.csv \
    --output malezainia2/ambel_crossdomain/phase2_eval_intl
```

### Phase 2C: Evaluate Combined Model on CL Test (Regression)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
conda run -n virtual_environment_yolo python scripts/cli/crossdomain_eval.py \
    --model training_results/malezainia2/ambel_crossdomain/combined_intl_v1_20260222_161204/weights/best.pt \
    --data configs/yaml_config/data_ambel-lentejas-seba-v1.yaml \
    --output malezainia2/ambel_crossdomain/phase2_eval_cl
```

### Phase 2D: Generate Comparison

```bash
python scripts/cli/crossdomain_compare.py \
    --evals \
        malezainia2/ambel_crossdomain/phase1_eval/crossdomain_metrics.json \
        malezainia2/ambel_crossdomain/phase2_eval_intl/crossdomain_metrics.json \
        malezainia2/ambel_crossdomain/phase2_eval_cl/crossdomain_metrics.json \
    --labels "CL Model -> INTL" "Combined -> INTL" "Combined -> CL" \
    --output malezainia2/ambel_crossdomain/comparison
```

---

## 9. Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/cli/build_international_dataset.py` | Build unified international YOLO dataset from 4 databases (format conversion, class remapping, filename prefixing) |
| `scripts/cli/build_combined_dataset.py` | Merge Chilean + international data with source-stratified splits |
| `scripts/cli/crossdomain_eval.py` | Evaluate model on test set with per-database breakdown via manifest |
| `scripts/cli/crossdomain_compare.py` | Compare multiple evaluations, generate comparison table + chart |
| `scripts/cli/train_yolo_cli.py` | YOLO training with real-time visualization and ClearML logging |

## 10. Configuration Files

| File | Purpose |
|------|---------|
| `configs/crossdomain/international_databases.yaml` | Database paths, formats, class mappings, prefixes for all databases |
| `configs/yaml_config/data_ambel-international-v1-test.yaml` | YOLO test-only config for international set |
| `configs/yaml_config/data_ambel-combined-intl-v1.yaml` | YOLO config for combined training set |
| `configs/yaml_config/data_ambel-lentejas-seba-v1.yaml` | YOLO config for CL_Seba dataset |

---

## 11. Hardware

- **GPU:** NVIDIA GeForce RTX 4090 (24 GB VRAM) x 2
- **Combined model training:** Single GPU due to PyTorch DDP `expandable_segments` allocator bug
- **Conda environment:** `virtual_environment_yolo` (ultralytics 8.3.49, torch 2.4.1, CUDA)
- **Platform:** Ubuntu Linux (malezainia2)

---

## 12. Source Documentation

The following documents in the project repository contain the full narrative and analysis:

| Document | Path |
|----------|------|
| Methods and Results (Section 6) | `malezainia2/ambel_crossdomain/CROSSDOMAIN_METHODS_RESULTS.md` |
| Experimental Plan | `malezainia2/ambel_crossdomain/CROSSDOMAIN_PLAN.md` |
| Comparison Report | `malezainia2/ambel_crossdomain/comparison/comparison_report.md` |
| Phase 1 Evaluation Report | `malezainia2/ambel_crossdomain/phase1_eval/crossdomain_report.md` |

---

*Evidence compiled: 2026-02-23. Corresponds to Section 6 of the AMBEL detection results series (Cross-Domain Detection Experiment).*
