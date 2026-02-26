"""
Example 03: Embeddings, MMD, and Deployment Gate
==================================================

Demonstrates the cross-domain analysis pipeline from Chapter 6:

1. Generate synthetic "embeddings" (stand-in for real ResNet-50 features)
2. Compute MMD between training and target distributions
3. Run permutation test for statistical significance
4. Apply the deployment gate to get a risk tier

In production, replace the synthetic data with real embeddings:
    extractor = FeatureExtractor("resnet50")
    embeddings = extract_embeddings(image_paths, extractor, device="cuda:0")
"""

import numpy as np

from ragweed_toolkit.embeddings.mmd import (
    compute_mmd,
    deployment_gate,
    permutation_test,
)

np.random.seed(42)

# ------------------------------------------------------------------ #
# 1.  Synthetic embeddings (simulating 2048-D ResNet-50 features)
# ------------------------------------------------------------------ #
d = 128  # lower dim for speed; real pipeline uses 2048
n_train, n_target = 200, 150

# Training domain: centered at zero
X_train = np.random.randn(n_train, d)

# Scenario A: similar domain (small shift)
X_similar = X_train[:n_target] + np.random.randn(n_target, d) * 0.1

# Scenario B: moderately shifted domain
X_shifted = np.random.randn(n_target, d) + 0.5

# Scenario C: very different domain
X_different = np.random.randn(n_target, d) * 3 + 2.0

# ------------------------------------------------------------------ #
# 2.  Compute MMD for each scenario
# ------------------------------------------------------------------ #
print("=== MMD Domain Shift Analysis ===\n")
print(f"{'Scenario':<25} {'MMD':>8} {'Tier':<10} {'Recommendation'}")
print("-" * 80)

for name, X_target in [
    ("A: Similar domain", X_similar),
    ("B: Moderate shift", X_shifted),
    ("C: Very different", X_different),
]:
    mmd_val = compute_mmd(X_train, X_target)
    result = deployment_gate(mmd_val)
    print(f"{name:<25} {mmd_val:>8.3f} {result.tier:<10} {result.recommendation}")

# ------------------------------------------------------------------ #
# 3.  Permutation test on the shifted domain
# ------------------------------------------------------------------ #
print("\n=== Permutation Test (Scenario B) ===")
mmd_val, p_value = permutation_test(
    X_train, X_shifted,
    n_permutations=200,  # use 1000 in production
    random_state=42,
)
print(f"  MMD       = {mmd_val:.4f}")
print(f"  p-value   = {p_value:.4f}")
print(f"  Reject H0 = {p_value < 0.05} (alpha=0.05)")

# ------------------------------------------------------------------ #
# 4.  Deployment gate thresholds
# ------------------------------------------------------------------ #
print("\n=== Deployment Gate Thresholds ===")
print("  MMD < 0.15  -> green  : Deploy directly")
print("  0.15 - 0.30 -> yellow : Deploy with augmentation")
print("  0.30 - 0.45 -> orange : Active learning required")
print("  > 0.45      -> red    : Collect new training data")
