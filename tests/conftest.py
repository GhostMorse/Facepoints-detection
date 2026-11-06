"""Shared pytest fixtures."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def sample_images(tmp_path):
    """Create two small RGB images for dataset tests."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.fromarray(np.full((20, 40, 3), 64, dtype=np.uint8)).save(image_dir / "00001.jpg")
    Image.fromarray(np.full((30, 50, 3), 128, dtype=np.uint8)).save(image_dir / "00002.jpg")
    return image_dir
