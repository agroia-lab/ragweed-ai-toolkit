# Data Access

This repository does not include raw data or model weights due to size constraints.
All data is available on the project machine (malezainia2) at the following locations.

## Model Weights

| Model | Path |
|-------|------|
| CL-only (Run 3, mAP50=0.886) | `training_results/malezainia2/ambel_seba/run3_lentejas640_b64_2gpu/weights/best.pt` |
| Combined international (mAP50=0.874) | `training_results/malezainia2/ambel_crossdomain/combined_intl_v1_20260222_161204/weights/best.pt` |

## Datasets

| Dataset | Path |
|---------|------|
| CL_Seba (augmented) | `/media/malezainia2/E/ProcessingData/ambrosia-lentejas-seba-v1/` |
| International unified | `/media/malezainia2/E/ProcessingData/ambel-international-v1/` |
| Combined | `/media/malezainia2/E/ProcessingData/ambel-combined-intl-v1/` |

## Embeddings

| Resource | Path |
|----------|------|
| ResNet50 embeddings (5,338 images, 9 databases) | `/media/malezainia2/E/rageweed_international_databases/ragweed_embeddings/embeddings.npz` |
| PRESTO outputs | `/home/malezainia2/dev/INIA_DeepLearning_Ubuntu_mod_lleon/outputs/lencu_presto/` |

## Orthomosaic

| Resource | Path |
|----------|------|
| Santa Rosa orthomosaic (2.4 GB) | See `configs/` for exact path |

## Shapefiles and Ancillary

| Resource | Path |
|----------|------|
| Detection points | `puntos lenteja v2.shp` (1,685 points) |
| EM38 conductivity | 1,899 points at 75 cm and 150 cm |
| Kriging rasters (5 m) | `1_Krig_AMBEL_Grid_Map.tiff`, etc. |
