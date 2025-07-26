import os
from typing import Tuple

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from config import Config


def prepare_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, int]:
    """Create train/validation dataloaders.

    Args:
        config: configuration object.

    Returns:
        train_loader, val_loader, num_classes
    """
    transform = transforms.Compose(
        [
            transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(config.DATA_DIR, transform=transform)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    return train_loader, val_loader, len(dataset.classes)
