"""Project configuration and repository paths."""

from pathlib import Path

SEED = 42
IMAGE_SIZE = (100, 100)
TRAIN_FRACTION = 0.8
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/facepoints_model.pt")
DEFAULT_NORMALIZATION_PATH = Path("outputs/normalization.json")
DEFAULT_PREDICTIONS_PATH = Path("outputs/predictions.csv")
