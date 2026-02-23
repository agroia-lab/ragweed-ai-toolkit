# EVIDENCE CATALOG — LENCU Book Chapter (Springer-Nature)

**Chapter title:** *AI-Driven Spatial Decision Support System for Sustainable Weed Management under Climate Variability*
**Book editor:** Prof. N. Benkeblia, UWI Jamaica; Springer-Nature, 2026
**Target species:** *Ambrosia artemisiifolia* (Common Ragweed / AMBEL)
**Compiled:** 2026-02-23 | **Machine:** malezainia2

---

## Evidence Folder Structure

```
evidence/
├── 00_master_index/          ← THIS FILE (catalog and navigation)
├── 01_training_progression/  ← 4 YOLO training runs (679 lines)
├── 02_embedding_analysis/    ← 9-database feature space (506 lines)
├── 03_cross_domain_experiment/ ← Phase 1 & 2 evaluations (479 lines)
├── 04_orthomosaic_pipeline/  ← Ortho tiling + domain shift (524 lines)
├── 05_field_inference/       ← Ground-level SAHI detections (265 lines)
├── 06_satellite_weed_analysis/ ← Presto + Sentinel-2 correlations (491 lines)
└── 07_drone_scale_detection/ ← Class inversion discovery (346 lines)
```

**Total documentation:** 3,290 lines across 7 evidence packages

---

## Summary of All Exercises

| # | Exercise | Method | Key Result | Status | Chapter Section |
|---|----------|--------|------------|--------|-----------------|
| 1 | **Training Progression** | YOLOv11l, 4 runs, hyperparameter sweep (batch, imgsz, data) | Best mAP50 = **0.886** (Run 3: 640px, batch 64, dual GPU) | Complete | 3.1 Methods |
| 2 | **Embedding Space Analysis** | ResNet50 → 2048-dim → UMAP/t-SNE/PCA + MMD/PAD | Database-specific clustering dominates; MMD 0.27-0.43 predicts deployment failure | Complete | 3.3 Domain Analysis |
| 3 | **Cross-Domain Experiment** | CL-only model vs Combined model on international test set | CL→INTL: mAP50=0.108 (fail); Combined→INTL: mAP50=0.874 (recovery) | Complete | 3.4 Generalization |
| 4 | **Orthomosaic Pipeline** | 9-phase: tile → embed → domain shift → select → label → train → infer → map | 2,151 tiles, MMD=0.4422, 80 tiles selected for labeling | Phases 0-3 done | 3.5 Orthomosaic |
| 5 | **Field Inference** | SAHI (slice=640, conf=0.25) on smartphone ground images | 3 sites, 336 detections; full campaign 1,390 detections over 75 images | Complete | 3.2 Field Deployment |
| 6 | **Satellite-Weed Analysis** | Presto embeddings + Sentinel-2 NDVI + GWR + LISA | NDVI×LENCU r=0.890; GWR R²=0.910; "spectral bridge works" | Complete | 5.3 Satellite Scaling |
| 7 | **Drone-Scale Detection** | 3 YOLOv11 variants × 2 SAHI resolutions on tomato drone imagery | AMBEL/LENCU class inversion; morphological convergence at altitude | Complete | 6.1 Species Discrimination |

---

## Chronological Timeline

| Date | Exercise | Milestone |
|------|----------|-----------|
| 2026-02-17 | Training (Run 1-2) | First AMBEL model trained; scaling to 1024px |
| 2026-02-18 | Training (Run 3-4) | Best model achieved (Run 3 mAP50=0.886); combined transfer tested |
| 2026-02-18 | Alberto Dataset | CL_Alberto zip obtained from Roboflow |
| 2026-02-18 | Satellite-Weed | Presto + Sentinel-2 analysis completed for Santa Rosa |
| 2026-02-21 | Embedding Analysis | 9-database embedding space + MMD/PAD framework |
| 2026-02-22 | Cross-Domain | Phase 1 (catastrophic failure) + Phase 2 (combined recovery) |
| 2026-02-22 | Orthomosaic | Sessions 1-2: tiling, embeddings, domain shift, tile selection |
| Pre-2026 | Drone Detection | Multi-species comparison on tomato (class inversion discovery) |
| Pre-2026 | Field Inference | SAHI deployment on 3 ground sites + full campaign |

---

## Mapping to Book Chapter Sections

(from `argument_map.md` three-act structure)

### Act I: Field-Level Detection
| Chapter Section | Evidence Folder | Status |
|-----------------|-----------------|--------|
| 3.1 Model Development | `01_training_progression/` | Complete |
| 3.2 Field Deployment | `05_field_inference/` | Complete |

### Act II: Domain Generalization
| Chapter Section | Evidence Folder | Status |
|-----------------|-----------------|--------|
| 3.3 Distribution Analysis | `02_embedding_analysis/` | Complete |
| 3.4 Cross-Domain Experiment | `03_cross_domain_experiment/` | Complete |
| 3.5 Orthomosaic Scale-Up | `04_orthomosaic_pipeline/` | Phases 0-3 of 9 |

### Act III: Satellite Integration
| Chapter Section | Evidence Folder | Status |
|-----------------|-----------------|--------|
| 5.3 Satellite Scaling | `06_satellite_weed_analysis/` | Complete |
| 6.1 Species Discrimination | `07_drone_scale_detection/` | Complete |

---

## Data Inventory (malezainia2)

### Images Processed
| Category | Count |
|----------|-------|
| Training images (across all datasets) | ~12,936 |
| Embedding analysis images | 5,338 |
| Orthomosaic tiles generated | 2,151 |
| International test images | 2,367 |
| Field inference images | 75 + 9 test |
| Drone comparison images | 591 (10 detailed) |
| **Total unique images involved** | **~20,000+** |

### Models Trained/Evaluated
| Model | mAP50 | Use |
|-------|-------|-----|
| Run 1: Seba 640 baseline | 0.810 | Baseline |
| Run 2: Lentejas 1024 | 0.871 | Resolution test |
| Run 3: Lentejas 640 dual GPU | **0.886** | **Production / best** |
| Run 4: Combined transfer | 0.839 | Transfer learning test |
| Combined CL+INTL | 0.752 val / 0.874 intl test | Cross-domain model |
| Original (multi-species 1024) | 0.846 | Drone-scale comparison |
| Maaferna (multi-species 2048) | 0.868 | Drone-scale comparison |
| Retrained (multi-species 1024) | 0.861 | Drone-scale comparison |

### Databases Used
| Database | Images | Origin | Type |
|----------|--------|--------|------|
| CL_Seba (Lentejas augmented) | 4,955 | Chile | Ground |
| CL_Alberto | 659 | Chile | Ground |
| CL_StaRosa (unlabelled) | 75 | Chile | Ground |
| ND Individual | 876 | North Dakota, USA | Ground |
| Michigan 3-Season | 719 | Michigan, USA | Ground |
| WeedCrop PrecAg | 495 | Multi-site USA | Ground |
| ND Aerial | 277 | North Dakota, USA | Drone |
| Purdue 4Weed | 150 | Indiana, USA | Vehicle |
| WeedCube USDA | 20 | USA | Hyperspectral |

### Figures Generated
| Category | Approximate Count |
|----------|-------------------|
| Training curves & confusion matrices | ~20 |
| Embedding visualizations (HTML + PNG) | ~15 |
| Cross-domain evaluation plots | ~50 |
| Orthomosaic visualizations | ~8 |
| Field inference shapefiles/maps | ~5 |
| Satellite analysis outputs | ~10 |
| Drone-scale comparison panels | ~7 |
| **Total figures/artifacts** | **~115** |

---

## External Drive Paths (malezainia2)

| Data | Path |
|------|------|
| Orthomosaic (2.4 GB) | `/media/malezainia2/E/Proyecto_lentejas_v2/Fotos lentejas 20 dic 24/lencu 13 dic 24/Ortomosaico.rgb.tif` |
| Ortho tiles (2,151) | `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_tiles_1024/` |
| Ortho embeddings | `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_embeddings/` |
| Ortho domain shift | `/media/malezainia2/E/Proyecto_lentejas_v2/ortho_domain_shift/` |
| Tiles for labeling (80) | `/media/malezainia2/E/Proyecto_lentejas_v2/tiles_for_labeling/` |
| International dataset | `/media/malezainia2/E/ProcessingData/ambel-international-v1/` |
| Combined dataset | `/media/malezainia2/E/ProcessingData/ambel-combined-intl-v1/` |
| Seba 640 dataset | `/media/malezainia2/E/ProcessingData/ambrosia.Seba/` |
| Lentejas Seba v1 | `/media/malezainia2/E/ProcessingData/ambrosia-lentejas-seba-v1/` |
| Alberto dataset | `/media/malezainia2/E/ProcessingData/ambrosia.dataset_alberto/` |
| Combined v1 | `/media/malezainia2/E/ProcessingData/ambel_combined_v1/` |
| Satellite (malezainia1 drive) | `/media/malezainia1/DRONES/ambel-data/` |

---

## Local Repository Paths (Key Files)

| Component | Path (relative to project root) |
|-----------|------|
| Best model weights | `training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/weights/best.pt` |
| Combined model | `training_results/malezainia2/ambel_crossdomain/combined_intl_v1_20260222_161204/weights/best.pt` |
| Training comparison | `malezainia2/ambel_seba/COMPARISON.md` |
| Inference results | `malezainia2/ambel_seba/INFERENCE_RESULTS.md` |
| Embedding report | `malezainia2/ragweed_embeddings/RAGWEED_EMBEDDINGS_REPORT.md` |
| Cross-domain report | `malezainia2/ambel_crossdomain/CROSSDOMAIN_METHODS_RESULTS.md` |
| Ortho plan | `malezainia2/ambel_ortho/AMBEL_ORTHO_PLAN.md` |
| Session checkpoints | `malezainia2/ambel_ortho/sessions/session_{1,2}_checkpoint.md` |
| Drone detection methods | `research_docs/lencu_book_chapter/lencu_methods_results.md` |
| Satellite analysis | `research_docs/lencu_book_chapter/satellite_weed_analysis/ANALYSIS_REPORT.md` |

---

## Machine & Environment

| Property | Value |
|----------|-------|
| Machine | malezainia2 |
| GPU | 2x NVIDIA RTX 4090 (24 GB VRAM each) |
| Preferred GPU | cuda:1 |
| Conda env | `virtual_environment_yolo` |
| Python | `/home/malezainia2/anaconda3/envs/virtual_environment_yolo/bin/python` |
| PyTorch | 2.4.1 + CUDA |
| Ultralytics | YOLOv11 (8.3.49+) |
| SAHI | Slicing Aided Hyper Inference |
| Embedding model | ResNet50 (ImageNet V2, torchvision) |

---

## Reproducibility Checklist

| Exercise | Data Available | Code Available | Can Reproduce | Notes |
|----------|---------------|----------------|---------------|-------|
| Training Runs 1-4 | Yes (ext drive) | Yes (CLI scripts) | Yes | Need ext drive E mounted |
| Embedding Analysis | Yes (ext drive) | Yes (generate_ragweed_embeddings.py) | Yes | ~30 min on RTX 4090 |
| Cross-Domain Eval | Yes (ext drive) | Yes (4 scripts) | Yes | Dataset build + eval + compare |
| Orthomosaic Phases 0-3 | Yes (ext drive) | Yes (3 scripts) | Yes | Need 2.4 GB orthomosaic |
| Orthomosaic Phases 4-8 | Partial | Yes (scripts exist) | No | Awaiting manual labeling of 80 tiles |
| Field Inference | Partial | Yes | Partial | Need original campo images on ext drive |
| Satellite-Weed | On malezainia1 | Yes | No (wrong machine) | Data paths reference malezainia1 drives |
| Drone Detection | Pre-existing | Pre-existing | Partial | Original model training on malezainia1 |

---

## What Remains on malezainia1 (Future Compilation)

The following evidence exists on malezainia1 but has NOT been compiled yet:

1. **Original multi-species training** (AMBEL, LENCU, POLAV, POLPE) — the 3 models used in drone-scale comparison
2. **Satellite embeddings raw data** — Sentinel-2 imagery, Presto model outputs
3. **INIA drone datasets** — sectors 1-4, 106/107/108MEDIA
4. **Sugal campo processing** — orobanche detection pipeline results
5. **QGIS projects** — master projects with all layers loaded
6. **SmartMap outputs** — kriging rasters, fuzzy zone classification
7. **EM38 electrical conductivity** — soil characterization data
8. **CENIA multi-date inferences** — 5 flights at 640 & 1024 resolutions

**Action needed:** Replicate this evidence compilation on malezainia1 to capture the full picture.

---

## Quick Navigation

Each folder's `README.md` contains:
- Detailed method description
- Complete results with tables and metrics
- All data/code paths (absolute)
- Exact CLI commands to reproduce
- Generated artifact inventory

**To browse:** Open any `evidence/0X_*/README.md` for full details on that exercise.
