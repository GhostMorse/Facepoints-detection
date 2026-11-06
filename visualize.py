"""Visualise ground-truth and/or predicted landmarks for one image."""

import argparse
from pathlib import Path

from PIL import Image

from facepoints.config import DEFAULT_NORMALIZATION_PATH
from facepoints.data.dataset import read_csv
from facepoints.evaluate import FacepointsEvaluator
from facepoints.visualization import visualize_keypoints


def parse_args():
    """Parse the image, annotation and optional inference inputs."""
    parser = argparse.ArgumentParser(description="Overlay facial landmarks on an image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--gt-csv", type=Path)
    parser.add_argument("--predictions-csv", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--normalization", type=Path, default=DEFAULT_NORMALIZATION_PATH)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show-indices", action="store_true")
    return parser.parse_args()


def main():
    """Load available landmarks and render them over the selected image."""
    args = parse_args()
    if args.predictions_csv is not None and args.checkpoint is not None:
        raise ValueError("Choose either --predictions-csv or --checkpoint, not both.")
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    ground_truth = read_csv(args.gt_csv).get(args.image.name) if args.gt_csv else None
    prediction = None
    if args.predictions_csv:
        prediction = read_csv(args.predictions_csv).get(args.image.name)
    elif args.checkpoint:
        prediction = FacepointsEvaluator(
            args.checkpoint, args.normalization, device=args.device
        ).predict_one(args.image)
    figure, _ = visualize_keypoints(
        image,
        ground_truth=ground_truth,
        prediction=prediction,
        title=args.image.name,
        show_indices=args.show_indices,
        output_path=args.output,
    )
    if args.output:
        print(f"Visualisation saved to {args.output}.")
    else:
        import matplotlib.pyplot as plt
        plt.show()
    return figure


if __name__ == "__main__":
    main()
