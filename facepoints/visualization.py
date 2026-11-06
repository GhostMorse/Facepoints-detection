"""Matplotlib visualisation helpers for ground-truth and predicted landmarks."""

import numpy as np
import matplotlib.pyplot as plt


def plot_keypoints(image, keypoints=None, ground_truth=None, prediction=None, title=None, show_indices=False, ax=None):
    """Render an RGB image with optional ground-truth and predicted landmarks."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image)
    if keypoints is not None:
        points = np.asarray(keypoints).reshape(-1, 2)
        ax.scatter(points[:, 0], points[:, 1], s=36, c="tab:red", marker="o", label="Keypoints")
        if show_indices:
            for index, (x_coord, y_coord) in enumerate(points):
                ax.annotate(str(index), (x_coord, y_coord), xytext=(3, 3), textcoords="offset points", color="yellow")
    if ground_truth is not None:
        points = np.asarray(ground_truth).reshape(-1, 2)
        ax.scatter(points[:, 0], points[:, 1], s=42, c="tab:green", marker="o", label="Ground truth")
    if prediction is not None:
        points = np.asarray(prediction).reshape(-1, 2)
        ax.scatter(points[:, 0], points[:, 1], s=42, c="tab:red", marker="x", label="Prediction")
    if ground_truth is not None or prediction is not None or keypoints is not None:
        ax.legend(loc="best")
    if title:
        ax.set_title(title)
    ax.set_axis_off()
    return ax


def visualize_keypoints(image, ground_truth=None, prediction=None, title=None, show_indices=False, output_path=None):
    """Create, display or save a landmark overlay for one image."""
    figure, axis = plt.subplots(figsize=(7, 7))
    plot_keypoints(
        image,
        ground_truth=ground_truth,
        prediction=prediction,
        title=title,
        show_indices=show_indices,
        ax=axis,
    )
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, bbox_inches="tight", dpi=160)
    return figure, axis
