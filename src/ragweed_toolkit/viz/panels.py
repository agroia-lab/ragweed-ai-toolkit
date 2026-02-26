"""
Detection Comparison Panels
===========================

Side-by-side ground-truth vs prediction panels and detection grids for
visual inspection of YOLO+SAHI results.

- :func:`plot_detection_panel` -- Single image with GT and/or prediction boxes
- :func:`plot_detection_grid` -- Grid of detection panels

All functions return a :class:`matplotlib.figure.Figure` and optionally
save to file at 300 DPI.

Usage::

    from ragweed_toolkit.viz.panels import plot_detection_panel, plot_detection_grid
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .style import DPI, FONT_SIZE, PANEL_LABEL_SIZE, set_publication_style


# Type alias for bounding boxes: list of (x_min, y_min, x_max, y_max[, label])
BBoxList = Sequence[Union[Tuple[float, float, float, float],
                          Tuple[float, float, float, float, str]]]


def plot_detection_panel(
    image: Union[str, Path, np.ndarray],
    *,
    gt_boxes: Optional[BBoxList] = None,
    pred_boxes: Optional[BBoxList] = None,
    title: Optional[str] = None,
    gt_color: str = "#2ecc71",
    pred_color: str = "#e74c3c",
    linewidth: float = 2.0,
    figsize: tuple = (12, 5),
    output_path: Optional[Union[str, Path]] = None,
) -> matplotlib.figure.Figure:
    """Plot a detection panel: image with ground truth and/or prediction boxes.

    If both *gt_boxes* and *pred_boxes* are given, creates a two-panel
    side-by-side view (GT left, Pred right).  If only one is provided,
    creates a single panel.

    Parameters
    ----------
    image : path or ndarray
        Image file path or a numpy array (H, W, 3).
    gt_boxes : sequence of tuples, optional
        Ground-truth boxes as ``(x_min, y_min, x_max, y_max)`` or
        ``(x_min, y_min, x_max, y_max, label)``.
    pred_boxes : sequence of tuples, optional
        Prediction boxes (same format).
    title : str, optional
        Figure super-title.
    gt_color, pred_color : str
        Box colours for GT and predictions.
    linewidth : float
        Box line width.
    figsize : tuple
        Figure size.
    output_path : path-like, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    set_publication_style()
    img = _load_image(image)

    both = gt_boxes is not None and pred_boxes is not None

    if both:
        fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=figsize)
        ax_gt.imshow(img)
        _draw_boxes(ax_gt, gt_boxes, gt_color, linewidth)
        ax_gt.set_title("Ground Truth", fontsize=FONT_SIZE + 1)
        ax_gt.text(
            0.03, 0.97, "(A)", transform=ax_gt.transAxes,
            fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="top",
            ha="left", color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                      alpha=0.6, edgecolor="none"),
        )
        ax_gt.axis("off")

        ax_pred.imshow(img)
        _draw_boxes(ax_pred, pred_boxes, pred_color, linewidth)
        ax_pred.set_title("Predictions", fontsize=FONT_SIZE + 1)
        ax_pred.text(
            0.03, 0.97, "(B)", transform=ax_pred.transAxes,
            fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="top",
            ha="left", color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                      alpha=0.6, edgecolor="none"),
        )
        ax_pred.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(figsize[0] // 2, figsize[1]))
        ax.imshow(img)
        boxes = gt_boxes if gt_boxes is not None else pred_boxes
        color = gt_color if gt_boxes is not None else pred_color
        label = "Ground Truth" if gt_boxes is not None else "Predictions"
        if boxes is not None:
            _draw_boxes(ax, boxes, color, linewidth)
        ax.set_title(label, fontsize=FONT_SIZE + 1)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=FONT_SIZE + 3, fontweight="bold", y=1.02)

    plt.tight_layout()
    _maybe_save(fig, output_path)
    return fig


def plot_detection_grid(
    images: Sequence[Union[str, Path, np.ndarray]],
    *,
    ncols: int = 3,
    gt_boxes_list: Optional[Sequence[Optional[BBoxList]]] = None,
    pred_boxes_list: Optional[Sequence[Optional[BBoxList]]] = None,
    titles: Optional[Sequence[str]] = None,
    gt_color: str = "#2ecc71",
    pred_color: str = "#e74c3c",
    linewidth: float = 1.5,
    cell_size: float = 4.0,
    output_path: Optional[Union[str, Path]] = None,
) -> matplotlib.figure.Figure:
    """Grid of detection panels.

    Each cell shows one image with optional GT (green) and prediction (red)
    boxes overlaid.

    Parameters
    ----------
    images : sequence
        List of image paths or arrays.
    ncols : int
        Number of columns in the grid.
    gt_boxes_list : sequence, optional
        List of GT box sequences, one per image (None entries are skipped).
    pred_boxes_list : sequence, optional
        List of prediction box sequences.
    titles : sequence of str, optional
        Per-cell titles.
    gt_color, pred_color : str
        Box colours.
    linewidth : float
        Box line width.
    cell_size : float
        Approximate width/height per cell in inches.
    output_path : path-like, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    set_publication_style()
    n = len(images)
    nrows = max(1, (n + ncols - 1) // ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(cell_size * ncols, cell_size * nrows),
    )
    axes = np.atleast_2d(axes)

    for idx in range(nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        if idx < n:
            img = _load_image(images[idx])
            ax.imshow(img)

            if gt_boxes_list is not None and idx < len(gt_boxes_list):
                gt = gt_boxes_list[idx]
                if gt is not None:
                    _draw_boxes(ax, gt, gt_color, linewidth)

            if pred_boxes_list is not None and idx < len(pred_boxes_list):
                pred = pred_boxes_list[idx]
                if pred is not None:
                    _draw_boxes(ax, pred, pred_color, linewidth)

            if titles is not None and idx < len(titles):
                ax.set_title(titles[idx], fontsize=FONT_SIZE - 1)
        ax.axis("off")

    plt.tight_layout()
    _maybe_save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load an image from path or pass through an array."""
    if isinstance(image, np.ndarray):
        return image
    from PIL import Image
    return np.array(Image.open(str(image)))


def _draw_boxes(
    ax: matplotlib.axes.Axes,
    boxes: BBoxList,
    color: str,
    linewidth: float,
) -> None:
    """Draw bounding boxes on an axes."""
    for box in boxes:
        x_min, y_min, x_max, y_max = box[:4]
        w = x_max - x_min
        h = y_max - y_min
        rect = Rectangle(
            (x_min, y_min), w, h,
            linewidth=linewidth, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)

        if len(box) > 4:
            label = str(box[4])
            ax.text(
                x_min, y_min - 2, label,
                fontsize=max(6, FONT_SIZE - 3), color=color,
                fontweight="bold", va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.15", facecolor="black",
                    alpha=0.5, edgecolor="none",
                ),
            )


def _maybe_save(
    fig: matplotlib.figure.Figure,
    output_path: Optional[Union[str, Path]],
) -> None:
    """Save figure if output_path is provided."""
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=DPI, bbox_inches="tight",
                    facecolor="white")
        print(f"  Saved: {output_path}")
