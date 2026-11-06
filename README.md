# Facepoints Detection

Facial landmark detection in PyTorch. Given an RGB face image, the model regresses **14 facial keypoints** (28 pixel coordinates). The implementation preserves the original four-stage CNN, its parallel skip branch, coordinate scaling, geometric augmentation strategy and MSE-based training objective.

## Overview

The network predicts fourteen ordered facial landmarks directly from a resized image. Each coordinate pair represents an `(x, y)` position in the original image coordinate system. During preprocessing, images are resized with bicubic interpolation and normalized using channel statistics calculated from the training split. Inference maps the predicted coordinates back to the original image dimensions.

## Model

`MyDetectionModel` contains four convolutional blocks with channel widths 64, 128, 256 and 512, followed by a 128-unit fully connected layer and a 28-value regression head. A parallel convolutional branch maps the RGB input to a 512-channel feature map; its output is added to the main branch before flattening. The model uses batch normalization, ReLU activations and dropout.

Training uses AdamW, MSELoss and `CosineAnnealingWarmRestarts`. The best checkpoint is selected by validation loss. The geometric augmentation pipeline uses horizontal flipping with landmark reordering, safe rotation and bounding-box-safe crop operations from Albumentations 1.3.1.

## Project structure

```text
Facepoints-detection/
├── facepoints/
│   ├── data/
│   │   ├── dataset.py        # CSV parsing and FacePointsDataset
│   │   ├── augment.py        # Joint image/keypoint augmentation
│   │   └── transforms.py     # Tensor conversion and normalisation
│   ├── models/
│   │   └── detection.py      # MyDetectionModel
│   ├── engine.py             # Training and validation loops
│   ├── evaluate.py           # Batch and single-image inference
│   ├── visualization.py      # Ground-truth and prediction overlays
│   ├── config.py             # Shared project paths and constants
│   └── utils.py              # Seeding, device and artifact utilities
├── notebooks/
│   ├── exploration.ipynb     # Dataset exploration and MEAN/STD calculation
│   └── inference.ipynb       # Standalone model inference notebook
├── tests/                    # CPU pytest suite
├── checkpoints/              # Model checkpoints (not tracked by Git)
├── outputs/                  # Normalisation metadata and predictions
├── train.py                  # Training entry point
├── run.py                    # Batch inference entry point
├── visualize.py              # Single-image visualisation entry point
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <repository-url>
cd Facepoints-detection
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10–3.12 is recommended. The project uses `albumentations==1.3.1` to retain compatibility with the augmentation API used in the original implementation. A CUDA-capable GPU is recommended for training and for calculating channel statistics. Inference and the test suite can run on CPU.

## Data

Place the dataset under the repository, or point the notebooks and command-line tools to another location:

```text
data/
└── public_tests/
    └── 00_test_img_input/
        ├── train/
        │   ├── gt.csv
        │   └── images/
        │       ├── 00001.jpg
        │       └── ...
        └── test/
            └── images/
                ├── ...
```

The first column of `gt.csv` contains the image filename. The remaining 28 columns contain fourteen `(x, y)` coordinate pairs in the original image pixel space. The filename in the CSV must match the corresponding image filename.

## Exploration and normalisation statistics

Open `notebooks/exploration.ipynb` and edit the dataset paths in the first configuration cell. The notebook displays image dimensions and ground-truth landmarks, reproduces the seeded 80/20 training/validation split, and calculates channel statistics using only the training indices. The statistics are calculated at **100 × 100**, matching the model input resolution.

The notebook writes the resulting metadata to `outputs/normalization.json`. Both training and inference load this file; they do not silently substitute hard-coded channel statistics. Run the exploration notebook before training or inference. The statistics cell follows the original CUDA-only calculation procedure and stops with an explicit error when CUDA is unavailable.

## Usage

### Training

```bash
python train.py \
    --gt-path data/public_tests/00_test_img_input/train/gt.csv \
    --images-dir data/public_tests/00_test_img_input/train/images \
    --normalization outputs/normalization.json \
    --checkpoint checkpoints/facepoints_model.pt \
    --device cuda
```

The original training settings are retained: 200 epochs, batch size 128, 100 × 100 input images, AdamW learning rate `3e-4`, weight decay `0.1`, and a seeded 80/20 split. Epoch count, batch size, image size, learning rate and weight decay are exposed as command-line arguments; the split fraction and scheduler configuration remain in `facepoints/config.py` and `facepoints/engine.py`. Use `--device cpu`, `--device cuda` or `--device auto` to select the compute device.

The optional `--fast` mode reproduces the original one-epoch CPU smoke-training configuration with 60 × 60 inputs and no augmentation. It does not write a checkpoint:

```bash
python train.py \
    --gt-path data/public_tests/00_test_img_input/train/gt.csv \
    --images-dir data/public_tests/00_test_img_input/train/images \
    --device cpu --fast
```

### Batch inference

```bash
python run.py \
    --images-dir data/public_tests/00_test_img_input/test/images \
    --checkpoint checkpoints/facepoints_model.pt \
    --normalization outputs/normalization.json \
    --device auto \
    --output outputs/predictions.csv
```

The output CSV contains the filename and 28 predicted coordinates in the original image coordinate system.

### Visualisation

Show ground-truth landmarks:

```bash
python visualize.py \
    --image data/public_tests/00_test_img_input/train/images/00001.jpg \
    --gt-csv data/public_tests/00_test_img_input/train/gt.csv
```

Compare ground truth with a saved prediction:

```bash
python visualize.py \
    --image data/public_tests/00_test_img_input/train/images/00001.jpg \
    --gt-csv data/public_tests/00_test_img_input/train/gt.csv \
    --predictions-csv outputs/predictions.csv \
    --output outputs/00001_visualization.png
```

Run model inference directly for one image by passing `--checkpoint` instead of `--predictions-csv`.

### Notebooks

- `notebooks/exploration.ipynb` — dataset inspection, ground-truth visualisation, train/validation split and channel-statistics calculation.
- `notebooks/inference.ipynb` — independent checkpoint loading, batch inference, coordinate output and prediction visualisation.

Edit the path variables in the first configuration cell of either notebook. No machine-specific absolute paths are embedded in the project.

## Implementation notes

- The model class and layer attribute names are preserved to maintain compatibility with state dictionaries produced by the original architecture.
- Landmark coordinates are resized using the half-pixel mapping `(x + 0.5) * scale - 0.5`; inference applies the inverse mapping to recover original-image coordinates.
- Horizontal flipping reorders landmarks with the original index mapping: `[3, 2, 1, 0, 9, 8, 7, 6, 5, 4, 10, 13, 12, 11]`.
- The normalisation artifact records the mean, standard deviation, image size, number of images, seed and training fraction used for its calculation.
- The fast mode is intended for a short CPU smoke run, not for producing the final checkpoint.

## Tests

```bash
pytest
```

The tests cover CSV parsing, coordinate resizing, the landmark permutation and the model output shape. They run on CPU and do not require a dataset or a trained checkpoint.
