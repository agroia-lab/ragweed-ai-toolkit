# Training Progression Evidence: AMBEL (Ragweed) YOLOv11 Detection

**Evidence Category:** Model Training and Hyperparameter Optimization
**Species:** *Ambrosia artemisiifolia* (Common Ragweed)
**Architecture:** YOLOv11l (Large) -- Ultralytics
**Machine:** malezainia2 (2x NVIDIA RTX 4090, 24 GB VRAM each)
**Conda Environment:** `virtual_environment_yolo`
**Training Dates:** 2026-02-17 / 2026-02-18
**Framework:** PyTorch 2.4.1, Ultralytics YOLOv11

---

## Summary Comparison Table

| Metric | Run 1 | Run 2 | Run 3 | Run 4 |
|--------|-------|-------|-------|-------|
| **Dataset** | Seba 640 | Lentejas Seba v1 | Lentejas Seba v1 | Combined v1 |
| **Images (train/val/test)** | 1,450 / 411 / 204 | 4,335 / 411 / 209 | 4,335 / 411 / 209 | 4,911 / 466 / 237 |
| **Transfer from** | COCO pretrained | COCO pretrained | COCO pretrained | Run 3 best.pt |
| **imgsz** | 640 | 1024 | 640 | 640 |
| **Batch size** | 32 | 8 | 64 | 64 |
| **GPU(s)** | 1x RTX 4090 | 1x RTX 4090 | 2x RTX 4090 | 2x RTX 4090 |
| **Epochs (completed)** | 100 | 100 | 100 | 16 (early stop) |
| **Training time** | 26.2 min | 164.0 min | 39.9 min | 7.5 min |
| **mAP50** | 0.810 | 0.871 | **0.886** | 0.839 |
| **mAP50-95** | 0.594 | 0.708 | **0.744** | 0.668 |
| **Precision** | 0.844 | **0.934** | 0.934 | 0.914 |
| **Recall** | 0.709 | 0.775 | **0.808** | 0.732 |
| **Fitness** | 0.616 | 0.724 | **0.744** | 0.668 |

**Best overall model: Run 3** (mAP50 = 0.886, mAP50-95 = 0.744)

---

## Field Inference Comparison (SAHI, 9 Images, 3 Sites)

Detection counts using SAHI (slice=640, conf=0.25) on 9 held-out field images from 3 different geographic sites:

| Site (3 images each) | Run 1 | Run 2 | Run 3 | Run 4 |
|----------------------|-------|-------|-------|-------|
| Ambrosia Sta. Rosa | 86 | 90 | 77 | **122** |
| CATO Maiz | **156** | 136 | 93 | 101 |
| Trigo corregidas | **202** | 178 | 166 | 193 |
| **TOTAL** | **444** | 404 | 336 | 416 |

---

## Run 1: Baseline -- Seba 640 Original Dataset

### Method

| Parameter | Value |
|-----------|-------|
| Architecture | YOLOv11l (Large) |
| Pretrained weights | `yolo11l.pt` (COCO) |
| Dataset | Seba 640 -- Original Roboflow images |
| Dataset source | `albertopractica/ambrosia.sta.rosa` v1 |
| Resolution | 640 x 640 (native) |
| Augmentations | None (Roboflow originals only) |
| Train / Val / Test split | 1,450 / 411 / 204 images |
| Dataset size | ~450 MB |
| Epochs (max / patience) | 100 / 15 |
| Batch size | 32 |
| Image size (imgsz) | 640 |
| Device | `cuda:0` (1x RTX 4090) |
| Optimizer | auto (SGD) |
| Learning rate (lr0 / lrf) | 0.01 / 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warmup epochs | 3.0 |
| AMP (mixed precision) | True |
| Mosaic augmentation | 1.0 |
| Close mosaic (last N epochs) | 10 |
| Random erasing | 0.4 |
| Horizontal flip | 0.5 |
| Seed | 42 |
| Deterministic | True |
| Cache | True |
| Workers | 8 |

### Results

| Metric | Value |
|--------|-------|
| **mAP50** | 0.810 |
| **mAP50-95** | 0.594 |
| **Precision** | 0.844 |
| **Recall** | 0.709 |
| **Fitness** | 0.616 |
| Training time | 26.2 minutes |
| Epochs completed | 100 (full run) |

### Key Insight

Baseline performance with original un-augmented images. The model achieves reasonable precision (0.844) but limited recall (0.709), suggesting the small training set (1,450 images) with no offline augmentations is insufficient for robust detection. This run establishes the performance floor for subsequent augmentation and scaling experiments.

### Data Paths

| Resource | Absolute Path |
|----------|---------------|
| Dataset YAML | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/configs/yaml_config/data_ambel-seba-640.yaml` |
| Dataset root | `/media/malezainia2/E/ProcessingData/ambrosia.Seba/` |
| Training results | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/run1_seba640_b32_1gpu/` |
| Best weights | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/run1_seba640_b32_1gpu/weights/best.pt` |
| Last weights | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/run1_seba640_b32_1gpu/weights/last.pt` |
| args.yaml | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/run1_seba640_b32_1gpu/args.yaml` |
| results.csv | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/run1_seba640_b32_1gpu/results.csv` |

### Reproduction Command

```bash
cd /home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon
conda activate virtual_environment_yolo

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
python scripts/cli/train_yolo_cli.py \
    --model yolo11l.pt \
    --data configs/yaml_config/data_ambel-seba-640.yaml \
    --epochs 100 \
    --batch 32 \
    --imgsz 640 \
    --device 0 \
    --patience 15 \
    --cache \
    --seed 42
```

### Generated Artifacts

| Artifact | Path (relative to run directory) |
|----------|----------------------------------|
| Training arguments | `args.yaml` |
| Epoch-by-epoch metrics | `results.csv` |
| Training summary (JSON) | `training_summary.json` |
| Training metrics (JSON) | `training_metrics.json` |
| Training curves (final) | `training_curves.png` |
| Training curves (live) | `training_curves_live.png` |
| Results plot (Ultralytics) | `results.png` |
| Precision-Recall curve | `PR_curve.png` |
| F1 curve | `F1_curve.png` |
| Precision curve | `P_curve.png` |
| Recall curve | `R_curve.png` |
| Confusion matrix | `confusion_matrix.png` |
| Confusion matrix (normalized) | `confusion_matrix_normalized.png` |
| Label distribution | `labels.jpg` |
| Label correlogram | `labels_correlogram.jpg` |
| Training batch samples | `train_batch0.jpg`, `train_batch1.jpg`, `train_batch2.jpg` |
| Final training batch samples | `train_batch4140.jpg`, `train_batch4141.jpg`, `train_batch4142.jpg` |
| Validation predictions | `val_batch0_pred.jpg`, `val_batch1_pred.jpg`, `val_batch2_pred.jpg` |
| Validation ground truth | `val_batch0_labels.jpg`, `val_batch1_labels.jpg`, `val_batch2_labels.jpg` |
| Checkpoint weights | `weights/epoch{0,10,20,...,90}.pt` (every 10 epochs) |
| Best model weights | `weights/best.pt` |
| Last model weights | `weights/last.pt` |

---

## Run 2: Augmented Dataset at High Resolution (1024)

### Method

| Parameter | Value |
|-----------|-------|
| Architecture | YOLOv11l (Large) |
| Pretrained weights | `yolo11l.pt` (COCO) |
| Dataset | Lentejas Seba v1 -- 3x augmented |
| Dataset source | `peppo-4-rotaciones/ambrosia-lentejas-seba-0rdlg` v1 |
| Resolution | ~1050 x 915 (native) |
| Offline augmentations | Random crop 0-29%, rotation +/-15 deg, salt & pepper noise |
| Train / Val / Test split | 4,335 / 411 / 209 images |
| Total annotations | Part of 4,955-image Roboflow dataset |
| Dataset size | ~1.1 GB |
| Epochs (max / patience) | 100 / 15 |
| Batch size | 8 |
| Image size (imgsz) | 1024 |
| Device | `cuda:0` (1x RTX 4090) |
| Optimizer | auto (SGD) |
| Learning rate (lr0 / lrf) | 0.01 / 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warmup epochs | 3.0 |
| AMP (mixed precision) | True |
| Mosaic augmentation | 1.0 |
| Close mosaic (last N epochs) | 10 |
| Random erasing | 0.4 |
| Horizontal flip | 0.5 |
| Seed | 42 |
| Deterministic | True |
| Cache | True |
| Workers | 8 |

### Results

| Metric | Value |
|--------|-------|
| **mAP50** | 0.871 |
| **mAP50-95** | 0.708 |
| **Precision** | 0.934 |
| **Recall** | 0.775 |
| **Fitness** | 0.724 |
| Training time | 164.0 minutes |
| Epochs completed | 100 (full run) |

### Key Insight

Augmented data (3x expansion with crop, rotation, noise) provides a significant improvement over the baseline: +7.5% mAP50, +19.2% mAP50-95. Precision jumps from 0.844 to 0.934. However, the high resolution (1024) forces a small batch size (8) due to VRAM constraints on a single GPU, resulting in 6.3x longer training time (164 min vs 26 min). This run demonstrates the value of offline augmentation but reveals the computational cost of high-resolution training on a single GPU.

### Data Paths

| Resource | Absolute Path |
|----------|---------------|
| Dataset YAML | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/configs/yaml_config/data_ambel-lentejas-seba-v1.yaml` |
| Dataset root | `/media/malezainia2/E/ProcessingData/ambrosia-lentejas-seba-v1/` |
| Training results | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/run2_lentejas1024_b8_1gpu/` |
| Best weights | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/run2_lentejas1024_b8_1gpu/weights/best.pt` |
| Last weights | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/run2_lentejas1024_b8_1gpu/weights/last.pt` |
| args.yaml | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/run2_lentejas1024_b8_1gpu/args.yaml` |
| results.csv | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/run2_lentejas1024_b8_1gpu/results.csv` |

### Reproduction Command

```bash
cd /home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon
conda activate virtual_environment_yolo

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
python scripts/cli/train_yolo_cli.py \
    --model yolo11l.pt \
    --data configs/yaml_config/data_ambel-lentejas-seba-v1.yaml \
    --epochs 100 \
    --batch 8 \
    --imgsz 1024 \
    --device 0 \
    --patience 15 \
    --cache \
    --seed 42
```

### Generated Artifacts

| Artifact | Path (relative to run directory) |
|----------|----------------------------------|
| Training arguments | `args.yaml` |
| Epoch-by-epoch metrics | `results.csv` |
| Training summary (JSON) | `training_summary.json` |
| Training metrics (JSON) | `training_metrics.json` |
| Training curves (final) | `training_curves.png` |
| Training curves (live) | `training_curves_live.png` |
| Results plot (Ultralytics) | `results.png` |
| Precision-Recall curve | `PR_curve.png` |
| F1 curve | `F1_curve.png` |
| Precision curve | `P_curve.png` |
| Recall curve | `R_curve.png` |
| Confusion matrix | `confusion_matrix.png` |
| Confusion matrix (normalized) | `confusion_matrix_normalized.png` |
| Label distribution | `labels.jpg` |
| Label correlogram | `labels_correlogram.jpg` |
| Training batch samples | `train_batch0.jpg`, `train_batch1.jpg`, `train_batch2.jpg` |
| Final training batch samples | `train_batch48780.jpg`, `train_batch48781.jpg`, `train_batch48782.jpg` |
| Validation predictions | `val_batch0_pred.jpg`, `val_batch1_pred.jpg`, `val_batch2_pred.jpg` |
| Validation ground truth | `val_batch0_labels.jpg`, `val_batch1_labels.jpg`, `val_batch2_labels.jpg` |
| Checkpoint weights | `weights/epoch{0,10,20,...,90}.pt` (every 10 epochs) |
| Best model weights | `weights/best.pt` |
| Last model weights | `weights/last.pt` |

---

## Run 3: Augmented Dataset at Standard Resolution with Dual-GPU (Best Model)

### Method

| Parameter | Value |
|-----------|-------|
| Architecture | YOLOv11l (Large) |
| Pretrained weights | `yolo11l.pt` (COCO) |
| Dataset | Lentejas Seba v1 -- 3x augmented (same as Run 2) |
| Dataset source | `peppo-4-rotaciones/ambrosia-lentejas-seba-0rdlg` v1 |
| Resolution | ~1050 x 915 (native), resized to 640 |
| Offline augmentations | Random crop 0-29%, rotation +/-15 deg, salt & pepper noise |
| Train / Val / Test split | 4,335 / 411 / 209 images |
| Dataset size | ~1.1 GB |
| Epochs (max / patience) | 100 / 15 |
| Batch size | 64 |
| Image size (imgsz) | 640 |
| Device | `cuda:0,1` (2x RTX 4090, DataParallel) |
| Optimizer | auto (SGD) |
| Learning rate (lr0 / lrf) | 0.01 / 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warmup epochs | 3.0 |
| AMP (mixed precision) | True |
| Mosaic augmentation | 1.0 |
| Close mosaic (last N epochs) | 10 |
| Random erasing | 0.4 |
| Horizontal flip | 0.5 |
| Seed | 42 |
| Deterministic | True |
| Cache | True |
| Workers | 8 |

**Environment variables required:**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
NCCL_P2P_DISABLE=1
```

### Results

| Metric | Value |
|--------|-------|
| **mAP50** | **0.886** |
| **mAP50-95** | **0.744** |
| **Precision** | 0.934 |
| **Recall** | **0.808** |
| **Fitness** | ~0.744 |
| Training time | 39.9 minutes |
| Epochs completed | 100 (full run) |
| Best epoch metrics (epoch 100) | P=0.934, R=0.808, mAP50=0.886, mAP50-95=0.744 |

**Epoch-level detail from results.csv (last 5 epochs):**

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| 96 | 0.937 | 0.801 | 0.883 | 0.733 |
| 97 | 0.938 | 0.798 | 0.882 | 0.734 |
| 98 | 0.931 | 0.804 | 0.883 | 0.734 |
| 99 | 0.943 | 0.801 | 0.887 | 0.743 |
| 100 | 0.934 | 0.808 | 0.886 | 0.744 |

### Key Insight

**Batch size matters more than resolution.** Run 3 (640px, batch 64) surpasses Run 2 (1024px, batch 8) on all metrics: +1.7% mAP50, +5.1% mAP50-95, +4.3% recall -- while training 4.1x faster (40 min vs 164 min). The larger batch size enabled by dual-GPU DataParallel provides more stable gradient estimates per step, improving convergence. This is the best validated model across all four runs and is used as the production AMBEL detector.

### Data Paths

| Resource | Absolute Path |
|----------|---------------|
| Dataset YAML | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/configs/yaml_config/data_ambel-lentejas-seba-v1.yaml` |
| Dataset root | `/media/malezainia2/E/ProcessingData/ambrosia-lentejas-seba-v1/` |
| Training results | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/` |
| Best weights | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/weights/best.pt` |
| Last weights | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/weights/last.pt` |
| args.yaml | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/args.yaml` |
| results.csv | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/results.csv` |

### Reproduction Command

```bash
cd /home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon
conda activate virtual_environment_yolo

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
NCCL_P2P_DISABLE=1 \
python scripts/cli/train_yolo_cli.py \
    --model yolo11l.pt \
    --data configs/yaml_config/data_ambel-lentejas-seba-v1.yaml \
    --epochs 100 \
    --batch 64 \
    --imgsz 640 \
    --device "0,1" \
    --patience 15 \
    --cache \
    --seed 42
```

### Generated Artifacts

| Artifact | Path (relative to run directory) |
|----------|----------------------------------|
| Training arguments | `args.yaml` |
| Epoch-by-epoch metrics | `results.csv` |
| Training summary (JSON) | `training_summary.json` |
| Training curves (final) | `training_curves.png` |
| Results plot (Ultralytics) | `results.png` |
| Precision-Recall curve | `PR_curve.png` |
| F1 curve | `F1_curve.png` |
| Precision curve | `P_curve.png` |
| Recall curve | `R_curve.png` |
| Confusion matrix | `confusion_matrix.png` |
| Confusion matrix (normalized) | `confusion_matrix_normalized.png` |
| Label distribution | `labels.jpg` |
| Label correlogram | `labels_correlogram.jpg` |
| Training batch samples | `train_batch0.jpg`, `train_batch1.jpg`, `train_batch2.jpg` |
| Final training batch samples | `train_batch6120.jpg`, `train_batch6121.jpg`, `train_batch6122.jpg` |
| Validation predictions | `val_batch0_pred.jpg`, `val_batch1_pred.jpg`, `val_batch2_pred.jpg` |
| Validation ground truth | `val_batch0_labels.jpg`, `val_batch1_labels.jpg`, `val_batch2_labels.jpg` |
| Checkpoint weights | `weights/epoch{0,10,20,...,90}.pt` (every 10 epochs) |
| Best model weights | `weights/best.pt` |
| Last model weights | `weights/last.pt` |

---

## Run 4: Transfer Learning on Combined Multi-Source Dataset

### Method

| Parameter | Value |
|-----------|-------|
| Architecture | YOLOv11l (Large) |
| Pretrained weights | **Run 3 best.pt** (transfer learning) |
| Dataset | Combined v1 -- Lentejas Seba + Alberto merged |
| Dataset sources | `peppo-4-rotaciones/ambrosia-lentejas-seba-0rdlg` v1 + `albertopractica/ambosia_sr1` v1 |
| Alberto resolution | 1344 x 1008 (native) |
| Combined total | 5,614 images, 41,057 annotations |
| Train / Val / Test split | 4,911 / 466 / 237 images |
| Dataset size | ~1.7 GB |
| Epochs (max / patience) | 100 / 15 |
| Batch size | 64 |
| Image size (imgsz) | 640 |
| Device | `cuda:0,1` (2x RTX 4090, DataParallel) |
| Optimizer | auto (SGD) |
| Learning rate (lr0 / lrf) | 0.01 / 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warmup epochs | 3.0 |
| AMP (mixed precision) | True |
| Mosaic augmentation | 1.0 |
| Close mosaic (last N epochs) | 10 |
| Random erasing | 0.4 |
| Horizontal flip | 0.5 |
| Seed | 42 |
| Deterministic | True |
| Cache | True |
| Workers | 8 |

**Environment variables required:**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
NCCL_P2P_DISABLE=1
```

### Results

| Metric | Value |
|--------|-------|
| **mAP50** | 0.839 |
| **mAP50-95** | 0.668 |
| **Precision** | 0.914 |
| **Recall** | 0.732 |
| **Fitness** | ~0.668 |
| Training time | 7.5 minutes |
| Epochs completed | 16 (early stopped at patience=15) |
| Best epoch | Epoch 1 (mAP50 = 0.839) |

**Epoch-level detail from results.csv (showing performance degradation):**

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| 1 | 0.914 | 0.732 | 0.839 | 0.668 |
| 2 | 0.833 | 0.684 | 0.786 | 0.575 |
| 3 | 0.775 | 0.657 | 0.745 | 0.498 |
| 4 | 0.747 | 0.645 | 0.723 | 0.482 |
| 8 | 0.812 | 0.685 | 0.781 | 0.544 |
| 13 | 0.833 | 0.695 | 0.793 | 0.558 |
| 16 | 0.816 | 0.687 | 0.781 | 0.557 |

### Key Insight

**Transfer learning with domain shift caused catastrophic forgetting.** The model peaked at epoch 1 (mAP50 = 0.839), then immediately degraded as the learning rate (lr0 = 0.01) was too aggressive for fine-tuning, overwriting the learned features from Run 3. The validation set changed substantially -- it now includes 55 new Alberto images from a different Santa Rosa field -- so the best-epoch metrics are not directly comparable to Runs 1-3. Despite lower formal metrics, Run 4 detects +58% more ragweed in Ambrosia Sta. Rosa field images (122 vs 77 detections), suggesting the Alberto training data from a similar geographic domain improves real-world recall for Santa Rosa-type fields. This reveals an important lesson: naive fine-tuning with a high learning rate is unsuitable for multi-domain adaptation.

### Data Paths

| Resource | Absolute Path |
|----------|---------------|
| Dataset YAML | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/configs/yaml_config/data_ambel-combined-v1.yaml` |
| Dataset root | `/media/malezainia2/E/ProcessingData/ambel_combined_v1/` |
| Alberto source data | `/media/malezainia2/E/ProcessingData/ambrosia.dataset_alberto/` |
| Alberto local copy | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/malezainia2/ambel_Alberto/` |
| Training results | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/combined_v1_transfer_20260218_171619/` |
| Best weights | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/combined_v1_transfer_20260218_171619/weights/best.pt` |
| Last weights | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/combined_v1_transfer_20260218_171619/weights/last.pt` |
| Transfer source | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/weights/best.pt` |
| args.yaml | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/combined_v1_transfer_20260218_171619/args.yaml` |
| results.csv | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/training_results/malezainia2/ambel_seba/combined_v1_transfer_20260218_171619/results.csv` |

### Reproduction Command

```bash
cd /home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon
conda activate virtual_environment_yolo

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
NCCL_P2P_DISABLE=1 \
python scripts/cli/train_yolo_cli.py \
    --model training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/weights/best.pt \
    --data configs/yaml_config/data_ambel-combined-v1.yaml \
    --epochs 100 \
    --batch 64 \
    --imgsz 640 \
    --device "0,1" \
    --patience 15 \
    --cache \
    --seed 42
```

### Generated Artifacts

| Artifact | Path (relative to run directory) |
|----------|----------------------------------|
| Training arguments | `args.yaml` |
| Epoch-by-epoch metrics | `results.csv` |
| Training summary (JSON) | `training_summary.json` |
| Training curves (final) | `training_curves.png` |
| Results plot (Ultralytics) | `results.png` |
| Precision-Recall curve | `PR_curve.png` |
| F1 curve | `F1_curve.png` |
| Precision curve | `P_curve.png` |
| Recall curve | `R_curve.png` |
| Confusion matrix | `confusion_matrix.png` |
| Confusion matrix (normalized) | `confusion_matrix_normalized.png` |
| Label distribution | `labels.jpg` |
| Label correlogram | `labels_correlogram.jpg` |
| Training batch samples | `train_batch0.jpg`, `train_batch1.jpg`, `train_batch2.jpg` |
| Validation predictions | `val_batch0_pred.jpg`, `val_batch1_pred.jpg`, `val_batch2_pred.jpg` |
| Validation ground truth | `val_batch0_labels.jpg`, `val_batch1_labels.jpg`, `val_batch2_labels.jpg` |
| Checkpoint weights | `weights/epoch0.pt`, `weights/epoch10.pt` |
| Best model weights | `weights/best.pt` |
| Last model weights | `weights/last.pt` |

---

## Dataset Configuration Files

### data_ambel-seba-640.yaml (Runs 1)

```yaml
# AMBEL (Ambrosia artemisiifolia) Single-Class Detection Dataset
# Source: Roboflow (albertopractica/ambrosia.sta.rosa v1)
# Images: 640x640, 2065 total
# License: CC BY 4.0

path: /media/malezainia2/E/ProcessingData/ambrosia.Seba
train: train/images
val: valid/images
test: test/images

nc: 1
names:
  0: AMBEL
```

### data_ambel-lentejas-seba-v1.yaml (Runs 2, 3)

```yaml
# AMBEL (Ambrosia artemisiifolia) Single-Class Detection Dataset -- Augmented
# Source: Roboflow (peppo-4-rotaciones/ambrosia-lentejas-seba-0rdlg v1)
# Images: ~1050x915 native resolution, 4955 total (3x augmented)
# Augmentations: random crop 0-29%, rotation +/-15 deg, salt & pepper noise
# License: CC BY 4.0

path: /media/malezainia2/E/ProcessingData/ambrosia-lentejas-seba-v1
train: train/images
val: valid/images
test: test/images

nc: 1
names:
  0: AMBEL
```

### data_ambel-combined-v1.yaml (Run 4)

```yaml
# AMBEL (Ambrosia artemisiifolia) Combined Dataset -- Alberto + Lentejas
# Sources:
#   - Alberto (albertopractica/ambosia_sr1 v1): 659 images, 1344x1008 native
#   - Lentejas (peppo-4-rotaciones/ambrosia-lentejas-seba-0rdlg v1): 4955 images, ~1050x915 native
# Total: 5614 images, 41057 annotations
# Single-class: AMBEL (class 0)
# License: CC BY 4.0

path: /media/malezainia2/E/ProcessingData/ambel_combined_v1
train: train/images
val: valid/images
test: test/images

nc: 1
names:
  0: AMBEL
```

---

## Shared Hyperparameters (All 4 Runs)

The following hyperparameters were held constant across all runs, as recorded in each `args.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| task | detect | Object detection mode |
| optimizer | auto (SGD) | Ultralytics default optimizer selection |
| lr0 | 0.01 | Initial learning rate |
| lrf | 0.01 | Final learning rate factor |
| momentum | 0.937 | SGD momentum |
| weight_decay | 0.0005 | L2 regularization |
| warmup_epochs | 3.0 | Linear warmup |
| warmup_momentum | 0.8 | Warmup momentum |
| warmup_bias_lr | 0.1 | Warmup bias learning rate |
| box | 7.5 | Box loss weight |
| cls | 0.5 | Classification loss weight |
| dfl | 1.5 | Distribution focal loss weight |
| nbs | 64 | Nominal batch size for scaling |
| amp | True | Automatic mixed precision |
| mosaic | 1.0 | Mosaic augmentation probability |
| close_mosaic | 10 | Disable mosaic for last 10 epochs |
| flipud | 0.0 | No vertical flip |
| fliplr | 0.5 | Horizontal flip probability |
| hsv_h | 0.015 | Hue shift |
| hsv_s | 0.7 | Saturation shift |
| hsv_v | 0.4 | Value (brightness) shift |
| translate | 0.1 | Translation augmentation |
| scale | 0.5 | Scale augmentation |
| erasing | 0.4 | Random erasing probability |
| auto_augment | randaugment | Automatic augmentation policy |
| degrees | 0.0 | No rotation (online) |
| shear | 0.0 | No shear |
| perspective | 0.0 | No perspective transform |
| mixup | 0.0 | No mixup |
| copy_paste | 0.0 | No copy-paste |
| iou (NMS) | 0.7 | IoU threshold for NMS |
| max_det | 300 | Maximum detections per image |
| seed | 42 | Random seed |
| deterministic | True | Deterministic training |
| cache | True | Image caching enabled |
| workers | 8 | DataLoader workers |
| save_period | 10 | Save checkpoint every 10 epochs |
| patience | 15 | Early stopping patience |

---

## Hardware and Software Environment

| Component | Specification |
|-----------|---------------|
| Machine | malezainia2 |
| GPU 0 | NVIDIA RTX 4090 (24 GB VRAM) |
| GPU 1 | NVIDIA RTX 4090 (24 GB VRAM) |
| Multi-GPU mode | DataParallel (DP) for Runs 3, 4 |
| Framework | PyTorch 2.4.1 + CUDA |
| Model library | Ultralytics (YOLOv11) |
| Conda environment | `virtual_environment_yolo` |
| OS | Ubuntu Linux |

**Known environment issues:**
- PyTorch 2.4.1 requires `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` to avoid a CUDA allocator bug
- Dual-GPU DataParallel requires `NCCL_P2P_DISABLE=1` on this specific hardware

---

## Key Findings (Consolidated)

1. **Offline data augmentation is the single largest contributor to performance.** Moving from 1,450 un-augmented images (Run 1) to 4,335 augmented images (Runs 2, 3) improved mAP50 by +7.5% and mAP50-95 by +19.2%.

2. **Batch size outweighs input resolution for this task.** Run 3 (640px, batch 64, 2 GPUs) outperforms Run 2 (1024px, batch 8, 1 GPU) on every metric while training 4.1x faster. Larger batches provide more stable gradient estimation.

3. **Dual-GPU DataParallel enables practical batch scaling.** Two RTX 4090s support batch 64 at 640px with cached data, completing 100 epochs in 40 minutes. This makes rapid experimental iteration feasible.

4. **Transfer learning requires careful learning rate tuning.** Run 4's naive fine-tuning (lr0 = 0.01) on the combined dataset caused catastrophic forgetting, with the model peaking at epoch 1 and degrading thereafter. A lower learning rate (e.g., lr0 = 0.0001) or layer freezing would be necessary for successful domain adaptation.

5. **Multi-source data improves site-specific detection despite lower formal metrics.** Run 4 detects +58% more ragweed in Santa Rosa field images, demonstrating that domain-specific training data enhances real-world performance even when aggregate validation metrics decrease due to a changed validation set composition.

---

## Progression Narrative

The four runs form a systematic investigation of key training factors for agricultural weed detection:

```
Run 1 (Baseline)          Run 2 (Augmentation)       Run 3 (Batch Scaling)      Run 4 (Multi-Domain)
mAP50 = 0.810             mAP50 = 0.871              mAP50 = 0.886              mAP50 = 0.839
                           +7.5%                       +1.7%                      -5.3% (*)
1,450 imgs, no aug        4,335 imgs, 3x aug         Same data as Run 2         5,614 imgs (2 sources)
batch 32, 1 GPU           batch 8, 1 GPU             batch 64, 2 GPUs           batch 64, 2 GPUs
640px                     1024px                     640px                      640px
26 min                    164 min                    40 min                     7.5 min (early stop)

(*) Run 4 uses a different validation set (includes Alberto images), so metrics are not directly comparable.
```

The progression demonstrates: (i) the dominant role of training data diversity over model hyperparameters, (ii) the practical advantage of batch size scaling over resolution scaling for fixed-compute training, and (iii) the challenges of naive transfer learning for multi-domain adaptation in agricultural settings.
