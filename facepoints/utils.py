"""Shared utilities for reproducibility, configuration and prediction files."""

import csv
import json
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    """Set random seeds for NumPy and PyTorch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device):
    """Resolve an explicit CPU/CUDA request or select an available device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")
    return torch.device(device)


def load_normalization(path):
    """Load channel statistics and preprocessing metadata from a JSON file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Normalization file was not found: {path}. Run notebooks/exploration.ipynb first."
        )
    with path.open(encoding="utf-8") as file:
        normalization = json.load(file)
    mean = normalization.get("mean")
    std = normalization.get("std")
    image_size = normalization.get("image_size")
    if not isinstance(mean, list) or len(mean) != 3:
        raise ValueError(f"Invalid 'mean' field in {path}.")
    if not isinstance(std, list) or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError(f"Invalid 'std' field in {path}.")
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise ValueError(f"Invalid 'image_size' field in {path}.")
    return normalization


def save_normalization(path, mean, std, image_size, num_images, seed, train_fraction):
    """Persist channel statistics and the dataset settings used to calculate them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "image_size": [int(value) for value in image_size],
        "num_images": int(num_images),
        "seed": int(seed),
        "train_fraction": float(train_fraction),
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)
    return payload


def save_predictions(path, predictions):
    """Write image names and flattened keypoint coordinates to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", *[f"coord_{index}" for index in range(28)]])
        for name, coordinates in predictions.items():
            writer.writerow([name, *np.asarray(coordinates, dtype=np.float64).tolist()])
