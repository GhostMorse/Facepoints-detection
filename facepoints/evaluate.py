"""Batch and single-image inference for the facial landmark model."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from facepoints.data.dataset import MyDataset
from facepoints.data.transforms import build_image_transform
from facepoints.models.detection import MyDetectionModel
from facepoints.utils import load_normalization, resolve_device


class FacepointsEvaluator:
    """Load a trained checkpoint and return landmarks in original image coordinates."""

    def __init__(self, checkpoint_path, normalization_path, device="auto"):
        self.device = resolve_device(device)
        self.normalization = load_normalization(normalization_path)
        self.image_size = tuple(self.normalization["image_size"])
        self.mean = self.normalization["mean"]
        self.std = self.normalization["std"]
        self.model = MyDetectionModel(self.image_size).to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.transform = build_image_transform(self.mean, self.std)

    def predict(self, images_path, batch_size=256):
        """Predict keypoints for every supported image in a directory."""
        dataset = MyDataset(images_path, resize_size=self.image_size, mean=self.mean, std=self.std)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        predictions = {}
        with torch.inference_mode():
            for image_batch, _, names in tqdm(loader, desc="Inference", unit="batch"):
                batch_predictions = self.model(image_batch.to(self.device)).cpu().numpy()
                for name, prediction in zip(names, batch_predictions):
                    old_h, old_w = dataset.image_sizes[name]
                    prediction[::2] = (prediction[::2] + 0.5) * (old_w / self.image_size[1]) - 0.5
                    prediction[1::2] = (prediction[1::2] + 0.5) * (old_h / self.image_size[0]) - 0.5
                    predictions[name] = prediction
        return predictions

    def predict_one(self, image_path):
        """Predict keypoints for one image and return coordinates in its original pixel space."""
        image_path = Path(image_path)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            old_w, old_h = image.size
            image = image.resize((self.image_size[1], self.image_size[0]), resample=Image.Resampling.BICUBIC)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            prediction = self.model(image_tensor)[0].cpu().numpy()
        prediction[::2] = (prediction[::2] + 0.5) * (old_w / self.image_size[1]) - 0.5
        prediction[1::2] = (prediction[1::2] + 0.5) * (old_h / self.image_size[0]) - 0.5
        return prediction
