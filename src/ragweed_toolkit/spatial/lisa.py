"""
Local Indicators of Spatial Association (LISA)
===============================================

Decompose bivariate Moran's I into local cluster types:

    HH (High-High) : Hot spot -- high x surrounded by high y
    LL (Low-Low)    : Cold spot
    HL (High-Low)   : Spatial outlier -- high x surrounded by low y
    LH (Low-High)   : Spatial outlier -- low x surrounded by high y
    NS              : Not significant at the chosen alpha level

Uses ``esda.moran.Moran_Local_BV`` with row-standardised DistanceBand
weights (Chapter 5, Section 5.4.1).

Dependencies: esda, libpysal, geopandas, numpy
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
from esda.moran import Moran_Local_BV
from libpysal.weights import DistanceBand

from .morans import build_distance_weights

# Quadrant labels (esda convention: q attribute is 1-based)
QUADRANT_MAP = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}

LISA_COLORS = {
    "HH": "#e74c3c",  # red
    "HL": "#e67e22",  # orange
    "LH": "#9b59b6",  # purple
    "LL": "#3498db",  # blue
    "NS": "#d5d5d5",  # light gray
}


def compute_bivariate_lisa(
    gdf: gpd.GeoDataFrame,
    var_x: str,
    var_y: str,
    threshold: float = 15.0,
    permutations: int = 999,
    alpha: float = 0.05,
    w: Optional[DistanceBand] = None,
) -> gpd.GeoDataFrame:
    """
    Compute bivariate Local Moran's I and classify into LISA clusters.

    Parameters
    ----------
    gdf : GeoDataFrame
        Point observations in a projected CRS with columns *var_x* and *var_y*.
    var_x : str
        Column name for the first variable.
    var_y : str
        Column name for the second variable.
    threshold : float
        Distance threshold for the spatial weight matrix (metres).
    permutations : int
        Number of permutations for pseudo p-values.
    alpha : float
        Significance level for cluster classification.
    w : DistanceBand, optional
        Pre-computed weight matrix.

    Returns
    -------
    GeoDataFrame
        Copy of *gdf* with added columns:

        - ``lisa_cluster`` : str -- one of HH, HL, LH, LL, NS
        - ``lisa_Ii`` : float -- local Moran's I statistic
        - ``lisa_p`` : float -- pseudo p-value
        - ``lisa_q`` : int -- quadrant code (1=HH, 2=LH, 3=LL, 4=HL)
    """
    mask = gdf[var_x].notna() & gdf[var_y].notna()
    gdf_valid = gdf[mask].copy().reset_index(drop=True)

    if len(gdf_valid) < 10:
        raise ValueError(f"Too few valid observations ({len(gdf_valid)}) for LISA.")

    if w is None:
        w = build_distance_weights(gdf_valid, threshold=threshold)

    x = gdf_valid[var_x].values
    y = gdf_valid[var_y].values

    lm = Moran_Local_BV(x, y, w, permutations=permutations)

    # Classify into quadrants
    labels = []
    for i in range(len(gdf_valid)):
        if lm.p_sim[i] < alpha:
            labels.append(QUADRANT_MAP.get(lm.q[i], "NS"))
        else:
            labels.append("NS")

    gdf_result = gdf_valid.copy()
    gdf_result["lisa_cluster"] = labels
    gdf_result["lisa_Ii"] = lm.Is
    gdf_result["lisa_p"] = lm.p_sim
    gdf_result["lisa_q"] = lm.q

    return gdf_result


def lisa_summary(gdf: gpd.GeoDataFrame, cluster_col: str = "lisa_cluster") -> dict:
    """
    Summarise LISA cluster counts and percentages.

    Parameters
    ----------
    gdf : GeoDataFrame
        Output from :func:`compute_bivariate_lisa`.
    cluster_col : str
        Column containing cluster labels.

    Returns
    -------
    dict
        Keys: HH, HL, LH, LL, NS -- each a dict with ``count`` and ``pct``.
    """
    n = len(gdf)
    summary = {}
    for cat in ["HH", "HL", "LH", "LL", "NS"]:
        count = int((gdf[cluster_col] == cat).sum())
        summary[cat] = {"count": count, "pct": round(100 * count / n, 1) if n > 0 else 0.0}
    return summary


def main():
    """CLI entry point for bivariate LISA cluster analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute bivariate LISA clusters from a spatial dataset."
    )
    parser.add_argument(
        "--input", required=True,
        help="GeoPackage/Shapefile with point observations",
    )
    parser.add_argument("--var-x", required=True, help="Column name for first variable")
    parser.add_argument("--var-y", required=True, help="Column name for second variable")
    parser.add_argument("--output", default=None, help="Output GeoPackage path (optional)")
    parser.add_argument("--threshold", type=float, default=15.0,
                        help="Distance threshold for spatial weights in metres (default: 15)")
    parser.add_argument("--permutations", type=int, default=999,
                        help="Number of permutations for p-values (default: 999)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level (default: 0.05)")
    args = parser.parse_args()

    gdf = gpd.read_file(args.input)
    print(f"Input: {len(gdf)} observations")
    print(f"Variables: {args.var_x} x {args.var_y}")
    print(f"Threshold: {args.threshold}m, permutations: {args.permutations}, alpha: {args.alpha}")

    result = compute_bivariate_lisa(
        gdf,
        args.var_x,
        args.var_y,
        threshold=args.threshold,
        permutations=args.permutations,
        alpha=args.alpha,
    )

    summary = lisa_summary(result)
    print("\nLISA Cluster Summary:")
    for cat in ["HH", "HL", "LH", "LL", "NS"]:
        info = summary[cat]
        print(f"  {cat}: {info['count']} ({info['pct']}%)")

    if args.output:
        from pathlib import Path
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_file(str(out_path), driver="GPKG")
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
