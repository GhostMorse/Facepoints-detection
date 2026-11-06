"""CPU tests for the face landmark regression model."""

import torch

from facepoints.models.detection import MyDetectionModel


def test_model_output_shape():
    """Verify that the model returns 28 coordinates for each input image."""
    model = MyDetectionModel((100, 100)).eval()
    with torch.inference_mode():
        output = model(torch.zeros(2, 3, 100, 100))
    assert output.shape == (2, 28)
