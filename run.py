"""Run batch inference and save predicted facial landmarks to CSV."""

import argparse
from pathlib import Path

from facepoints.config import DEFAULT_CHECKPOINT_PATH, DEFAULT_NORMALIZATION_PATH, DEFAULT_PREDICTIONS_PATH
from facepoints.evaluate import FacepointsEvaluator
from facepoints.utils import save_predictions


def parse_args():
    """Parse paths and inference options."""
    parser = argparse.ArgumentParser(description="Predict 14 facial landmarks for an image directory.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--normalization", type=Path, default=DEFAULT_NORMALIZATION_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main():
    """Generate and persist predictions for the requested image directory."""
    args = parse_args()
    evaluator = FacepointsEvaluator(args.checkpoint, args.normalization, device=args.device)
    predictions = evaluator.predict(args.images_dir, batch_size=args.batch_size)
    save_predictions(args.output, predictions)
    print(f"Predicted {len(predictions)} images. Results saved to {args.output}.")


if __name__ == "__main__":
    main()
