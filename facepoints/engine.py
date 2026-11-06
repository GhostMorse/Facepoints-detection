"""Training and validation loops for facial landmark regression."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from facepoints.config import IMAGE_SIZE, SEED, TRAIN_FRACTION
from facepoints.data.augment import augment_batch, build_train_transform
from facepoints.data.dataset import MyDataset
from facepoints.models.detection import MyDetectionModel
from facepoints.utils import set_seed


def train_detector(
    train_gt,
    train_img_dir,
    mean,
    std,
    checkpoint_path,
    device,
    fast_train=False,
    batch_size=128,
    image_size=IMAGE_SIZE,
    epochs=200,
    seed=SEED,
    train_fraction=TRAIN_FRACTION,
    learning_rate=3e-4,
    weight_decay=0.1,
):
    """Train the landmark regressor and save the checkpoint with the lowest validation loss."""
    if fast_train:
        batch_size, image_size, epochs = 256, (60, 60), 1
        device = torch.device("cpu")
    else:
        device = torch.device(device)
    set_seed(seed)
    print(f"Type of training: fast_train = {fast_train}")
    print(f"Number of epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Input image size: {image_size}")
    print(f"Device: {device}")

    dataset = MyDataset(train_img_dir, train_gt, resize_size=image_size, mean=mean, std=std)
    if len(dataset) < 2:
        raise ValueError("At least two labelled images are required for a train/validation split.")
    indices = np.random.RandomState(seed).permutation(len(dataset))
    split_index = round(len(dataset) * train_fraction)
    if split_index == 0 or split_index == len(dataset):
        raise ValueError("The selected train/validation fraction produces an empty split.")
    train_indices, validation_indices = indices[:split_index], indices[split_index:]
    train_dataset = Subset(dataset, train_indices.tolist())
    validation_dataset = Subset(dataset, validation_indices.tolist())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size * 4, shuffle=False, drop_last=False, num_workers=0
    )
    if len(train_loader) == 0:
        raise ValueError(
            f"The training split contains {len(train_dataset)} images, fewer than batch_size={batch_size}. "
            "Reduce --batch-size."
        )
    train_transform = None if fast_train else build_train_transform(image_size)

    model = MyDetectionModel(image_size).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, eta_min=5e-5, T_mult=1
    )
    best_validation_loss = torch.inf
    checkpoint_path = Path(checkpoint_path)
    if not fast_train:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_samples = 0
        progress = tqdm(train_loader, desc=f"Train epoch {epoch + 1}", unit="batch", dynamic_ncols=True)
        for image_batch, target_batch, _ in progress:
            if train_transform is not None:
                image_batch, target_batch = augment_batch(
                    image_batch, target_batch, train_transform, image_size, device
                )
            else:
                image_batch = image_batch.to(device)
                target_batch = target_batch.to(device)
            optimizer.zero_grad()
            predictions = model(image_batch)
            loss = criterion(predictions, target_batch)
            loss.backward()
            optimizer.step()
            batch_count = image_batch.shape[0]
            total_loss += loss.item() * batch_count
            num_samples += batch_count
            progress.set_postfix(loss=f"{loss.item():.4f}")
        print(f"Training loss: {total_loss / num_samples:.6f}")

        model.eval()
        total_loss = 0.0
        num_samples = 0
        with torch.no_grad():
            progress = tqdm(validation_loader, desc="Validation", unit="batch", dynamic_ncols=True)
            for image_batch, target_batch, _ in progress:
                image_batch = image_batch.to(device)
                target_batch = target_batch.to(device)
                loss = criterion(model(image_batch), target_batch)
                batch_count = image_batch.shape[0]
                total_loss += loss.item() * batch_count
                num_samples += batch_count
                progress.set_postfix(loss=f"{loss.item():.4f}")
        validation_loss = total_loss / num_samples
        print(f"Validation loss: {validation_loss:.6f}")
        scheduler.step()

        if not fast_train and validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Updated checkpoint: {checkpoint_path}")

    if not fast_train:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return model
