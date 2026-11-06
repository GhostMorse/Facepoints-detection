"""Tests for normalisation metadata persistence."""

from facepoints.utils import load_normalization, save_normalization


def test_normalization_round_trip(tmp_path):
    """Verify that normalisation values and image metadata survive JSON persistence."""
    path = tmp_path / "outputs" / "normalization.json"
    save_normalization(path, (0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (100, 100), 12, 42, 0.8)
    result = load_normalization(path)
    assert result["mean"] == [0.1, 0.2, 0.3]
    assert result["std"] == [0.4, 0.5, 0.6]
    assert result["image_size"] == [100, 100]
