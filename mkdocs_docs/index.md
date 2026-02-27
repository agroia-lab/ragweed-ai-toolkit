# Ragweed AI Toolkit

**Multi-scale AI toolkit for weed detection, mapping, and spatial analysis.**

Companion software to the Springer Nature book chapter:

> **AI-Driven Spatial Decision-Support for Sustainable Weed Management under Climate Variability**
>
> Lorenzo A. Leon-Aguilar et al.
>
> In: *Smart-Agriculture and Technology-Innovation: Facing the Dynamic of Climate Change*
> (Springer Nature, 2026)

---

## Three Scales of Analysis

```
FIELD (cm)              CROSS-DOMAIN            SATELLITE (10 m)
YOLOv11 + SAHI    -->   ResNet-50 + MMD    -->   PRESTO + GWR
detect plants           assess deployment        map risk zones
                        readiness
         |                     |                       |
         v                     v                       v
    Kriging maps          Deployment gates         LISA clusters
    (5 m density)         (green/yellow/           (hot spots /
                           orange/red)              cold spots)
```

| Scale | Module | What it does |
|-------|--------|-------------|
| Field | `detection` | YOLOv11 training + SAHI sliced inference + GPS extraction |
| Field | `geostatistics` | Ordinary kriging density surfaces from point detections |
| Field | `orthomosaic` | Drone mosaic tiling, georeferenced export, active learning |
| Cross-domain | `embeddings` | ResNet-50 features, MMD distance, UMAP/t-SNE |
| Cross-domain | `crossdomain` | Multi-source dataset building, transfer evaluation |
| Satellite | `satellite` | PRESTO embeddings, Sentinel-2 composites, spectral indices |
| Satellite | `spatial` | Bivariate Moran's I, LISA clusters, GWR |
| All | `viz` | Publication-quality figures at 300 DPI |

## Quick Links

- [Installation](installation.md) -- Get up and running in minutes
- [CLI Reference](cli.md) -- 8 command-line tools for common workflows
- [Tutorials](tutorials/detection.md) -- Step-by-step guides with code examples
- [API Reference](api/detection.md) -- Full Python API documentation
- [Chapter Results](chapter.md) -- Key metrics from the Springer chapter

## Key Results

| Metric | Value |
|--------|-------|
| Best field detection (mAP50) | 0.886 |
| Cross-domain collapse (CL -> Int'l) | 0.886 -> 0.108 |
| Multi-domain recovery | 0.874 |
| PRESTO PC1 x AMBEL correlation | r = 0.739 |
| GWR R-squared (AMBEL, 3 PCs) | 0.882 |
| GWR R-squared (LENCU, 3 PCs) | 0.910 |
| LISA significant pixels | 68% |

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

## License

MIT License. See [LICENSE](https://github.com/agroia-lab/ragweed-ai-toolkit/blob/master/LICENSE) for details.
