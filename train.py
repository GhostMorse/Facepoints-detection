"""Command-line entry point for training the face landmark regressor."""

import argparse
from pathlib import Path

from facepoints.config import DEFAULT_CHECKPOINT_PATH, DEFAULT_NORMALIZATION_PATH, IMAGE_SIZE, SEED
from facepoints.data.dataset import read_csv
from facepoints.engine import train_detector
from facepoints.utils import load_normalization, resolve_device


def parse_args():
    """Parse dataset paths and training options."""
    parser = argparse.ArgumentParser(description="Train the 14-point face landmark detector.")
    parser.add_argument("--gt-path", type=Path, required=True, help="Path to the training gt.csv file.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing training images.")
    parser.add_argument("--normalization", type=Path, default=DEFAULT_NORMALIZATION_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fast", action="store_true", help="Run the original one-epoch CPU smoke-training mode.")
    return parser.parse_args()


def main():
    """Load dataset metadata and start model training."""
    args = parse_args()
    normalization = load_normalization(args.normalization)
    image_size = (args.image_size, args.image_size)
    if not args.fast and tuple(normalization["image_size"]) != image_size:
        raise ValueError(
            f"Training image size {image_size} differs from the normalization size "
            f"{tuple(normalization['image_size'])}. Recalculate statistics in exploration.ipynb."
        )
    device = resolve_device(args.device)
    if args.fast and device.type != "cpu":
        raise ValueError("The --fast mode uses CPU. Pass --device cpu.")
    train_detector(
        train_gt=read_csv(args.gt_path),
        train_img_dir=args.images_dir,
        mean=normalization["mean"],
        std=normalization["std"],
        checkpoint_path=args.checkpoint,
        device=device,
        fast_train=args.fast,
        batch_size=args.batch_size,
        image_size=image_size,
        epochs=args.epochs,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )


if __name__ == "__main__":
    main()
