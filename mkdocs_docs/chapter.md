# Chapter Results

Key results from the Springer Nature book chapter:

> **AI-Driven Spatial Decision-Support for Sustainable Weed Management under Climate Variability**
>
> Lorenzo A. Leon-Aguilar et al.
>
> In: *Smart-Agriculture and Technology-Innovation: Facing the Dynamic of Climate Change*
> (Springer Nature, 2026)

## Summary of Results

### Act I -- Field Detection

| Metric | Value |
|--------|-------|
| Best mAP50 (AMBEL detection) | 0.886 |
| Model | YOLOv11-L, 640 px, SGD, 50 epochs |
| SAHI slice size | 640 px, overlap 0.2, conf 0.25 |

### Act II -- Cross-Domain Generalization

| Metric | Value |
|--------|-------|
| Cross-domain collapse (Chile -> International) | mAP50 = 0.108 |
| Multi-domain recovery | mAP50 = 0.874 |
| Deployment gate thresholds | Green < 0.15, Yellow < 0.30, Orange < 0.45, Red >= 0.45 |

### Act III -- Satellite Integration

| Metric | Value |
|--------|-------|
| NDVI-AMBEL correlation (Sept 2024) | r = 0.837 |
| NDVI-LENCU correlation (Sept 2024) | r = 0.890 |
| PRESTO PC1 x AMBEL correlation | r = 0.739 |
| PRESTO PC1 x LENCU correlation | r = 0.717 |
| Bivariate Moran's I (PC1 x AMBEL) | 0.706 (p = 0.001) |
| GWR R-squared AMBEL (PC1+PC2+PC3) | 0.882 |
| GWR R-squared LENCU (PC1+PC2+PC3) | 0.910 |
| LISA significant pixels | 68% |

## Multi-Scale Integration

The central finding: no single scale solves the problem. Together, the three scales form a decision-support toolkit:

```
SATELLITE (10 m)  -->  DRONE (1-5 cm)  -->  DETECTION (mm)  -->  ADVISORY
  zone delineation      orthomosaic         species ID           management
  NDVI, PRESTO          ODM                 YOLO + SAHI          risk zones
```

## Citation

```bibtex
@incollection{leon2026ragweed,
  title     = {AI-Driven Spatial Decision-Support for Sustainable Weed
               Management under Climate Variability},
  author    = {Le{\'o}n-Aguilar, Lorenzo A.},
  booktitle = {Smart-Agriculture and Technology-Innovation: Facing the
               Dynamic of Climate Change},
  publisher = {Springer Nature},
  year      = {2026},
}
```
