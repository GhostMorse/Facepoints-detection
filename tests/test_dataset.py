"""Tests for landmark CSV parsing and image-coordinate resizing."""

import csv

import numpy as np
import torch

from facepoints.data.dataset import MyDataset, read_csv


def test_dataset_resizes_image_and_coordinates(sample_images):
    """Verify image tensor shape and half-pixel landmark scaling."""
    target = {"00001.jpg": np.tile(np.asarray([0.0, 0.0, 39.0, 19.0]), 7)}
    dataset = MyDataset(
        sample_images,
        target=target,
        resize_size=(10, 20),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    image, coordinates, name = dataset[0]
    assert name == "00001.jpg"
    assert image.shape == (3, 10, 20)
    assert coordinates.shape == (28,)
    assert torch.allclose(coordinates[:4], torch.tensor([-0.25, -0.25, 19.25, 9.25]))
    assert dataset.image_sizes[name] == (20, 40)


def test_read_csv(tmp_path):
    """Read a header-based CSV containing one image and 28 coordinates."""
    path = tmp_path / "gt.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", *[f"p{index}" for index in range(28)]])
        writer.writerow(["00001.jpg", *range(28)])
    result = read_csv(path)
    assert result["00001.jpg"].shape == (28,)
    assert result["00001.jpg"][0] == 0
    assert result["00001.jpg"][-1] == 27
