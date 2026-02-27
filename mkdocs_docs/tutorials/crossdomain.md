# Tutorial: Cross-Domain Analysis

This tutorial demonstrates how to evaluate whether a detection model trained on one domain (e.g., Chilean field images) will work when deployed to a different domain (e.g., European datasets). The pipeline uses ResNet-50 embeddings, Maximum Mean Discrepancy (MMD), and a deployment gate with four risk tiers.

**Corresponding notebook:** `notebooks/02_crossdomain_analysis.ipynb`

---

## Background

In the chapter, a YOLOv11 model trained on Chilean data (mAP50 = 0.886) collapsed to mAP50 = 0.108 when evaluated on international datasets. Multi-domain training recovered performance to 0.874. The MMD-based deployment gate helps predict this collapse *before* it happens in the field.

## Step 1: Extract Embeddings

Extract 2048-D ResNet-50 feature vectors from each image in your source (training) and target (deployment) domains.

```python
from ragweed_toolkit.embeddings import FeatureExtractor, extract_embeddings

extractor = FeatureExtractor(backbone="resnet50")

embeddings_source = extract_embeddings(source_image_paths, extractor)
embeddings_target = extract_embeddings(target_image_paths, extractor)

print(f"Source: {embeddings_source.shape}")  # (N, 2048)
print(f"Target: {embeddings_target.shape}")  # (M, 2048)
```

Or use the CLI:

```bash
ragweed-embed --source source_images/ --output source.npy
ragweed-embed --source target_images/ --output target.npy
```

## Step 2: Compute MMD

The MMD measures distributional distance in a reproducing kernel Hilbert space. Higher MMD means more domain shift.

```python
from ragweed_toolkit.embeddings import compute_mmd

mmd_value = compute_mmd(embeddings_source, embeddings_target)
print(f"MMD = {mmd_value:.3f}")
```

## Step 3: Permutation Test

Test whether the observed MMD is statistically significant. Under the null hypothesis (identical distributions), domain labels are randomly shuffled.

```python
from ragweed_toolkit.embeddings import permutation_test

mmd_value, p_value = permutation_test(
    embeddings_source, embeddings_target,
    n_permutations=1000,
    random_state=42,
)
print(f"MMD = {mmd_value:.4f}, p = {p_value:.4f}")
```

## Step 4: Deployment Gate

The deployment gate maps the MMD value to an actionable risk tier:

| MMD Range | Tier | Recommendation |
|-----------|------|----------------|
| < 0.15 | Green | Deploy directly |
| 0.15 -- 0.30 | Yellow | Deploy with augmentation |
| 0.30 -- 0.45 | Orange | Active learning required |
| > 0.45 | Red | Collect new training data |

```python
from ragweed_toolkit.embeddings import deployment_gate

result = deployment_gate(mmd_value)
print(f"Tier: {result.tier}")
print(f"Recommendation: {result.recommendation}")
```

Or use the CLI for the full pipeline:

```bash
ragweed-mmd --source source.npy --target target.npy --permutations 1000
```

## Step 5: Visualize Domain Separation

Reduce embeddings to 2D with UMAP or PCA and colour by domain to visualize overlap.

```python
import numpy as np
from ragweed_toolkit.embeddings import reduce_embeddings

all_embeddings = np.vstack([embeddings_source, embeddings_target])
labels = ["source"] * len(embeddings_source) + ["target"] * len(embeddings_target)

df = reduce_embeddings(
    all_embeddings,
    method="umap",
    n_components=2,
    metadata={"domain": labels},
)
# df has columns: umap_x, umap_y, domain
```

Visualize with plotly or matplotlib:

```python
import plotly.express as px

fig = px.scatter(df, x="umap_x", y="umap_y", color="domain",
                 title="Cross-Domain Embedding Space")
fig.write_html("domain_separation.html")
```

## MMD Heatmap for Multiple Domains

When comparing many domains, compute a pairwise MMD matrix:

```python
domain_names = ["Chile", "Spain", "Germany", "USA"]
domain_embeddings = [X_chile, X_spain, X_germany, X_usa]

n = len(domain_names)
mmd_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        mmd_matrix[i, j] = compute_mmd(domain_embeddings[i], domain_embeddings[j])
        mmd_matrix[j, i] = mmd_matrix[i, j]
```

## What's Next

- **[Spatial Mapping](spatial.md)** -- Use detection outputs for geostatistical analysis
- **[Satellite Integration](satellite.md)** -- Correlate satellite features with weed density
