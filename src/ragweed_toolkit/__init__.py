"""
Ragweed AI Toolkit
==================

Multi-scale AI toolkit for weed detection, mapping, and spatial analysis.
Companion software to the Springer Nature chapter:

    "AI-Driven Spatial Decision-Support for Sustainable Weed Management
     under Climate Variability"

Modules
-------
detection     : YOLOv11 training + SAHI sliced inference + GPS extraction
embeddings    : ResNet-50 feature extraction, MMD deployment gates, UMAP/t-SNE
crossdomain   : Cross-domain dataset building and transfer evaluation
orthomosaic   : Orthomosaic tiling, detection georeferencing, active learning
geostatistics : Ordinary kriging, semivariogram fitting, density surfaces
spatial       : Bivariate Moran's I, LISA clusters, GWR, cross-variograms
satellite     : PRESTO embeddings, Sentinel-2 composites, spectral indices
viz           : Publication-quality maps, scatter plots, heatmaps
"""

__version__ = "0.1.0"
__author__ = "Lorenzo León"
