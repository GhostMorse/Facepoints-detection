"""Tests for checkpoint loading and original-space inference coordinates."""

import numpy as np
import torch
from PIL import Image

from facepoints.evaluate import FacepointsEvaluator
from facepoints.models.detection import MyDetectionModel
from facepoints.utils import save_normalization


def test_evaluator_predicts_original_space_coordinates(tmp_path):
    """Load a local checkpoint and verify the single-image prediction contract."""
    image_path = tmp_path / "sample.jpg"
    Image.fromarray(np.full((80, 120, 3), 128, dtype=np.uint8)).save(image_path)
    model = MyDetectionModel((100, 100))
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    normalization_path = tmp_path / "normalization.json"
    save_normalization(normalization_path, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25), (100, 100), 1, 42, 0.8)
    evaluator = FacepointsEvaluator(checkpoint_path, normalization_path, device="cpu")
    prediction = evaluator.predict_one(image_path)
    assert prediction.shape == (28,)
