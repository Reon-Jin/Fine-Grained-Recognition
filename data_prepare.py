import os
from typing import Tuple

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

from config import Config


def get_train_transform(input_size: int, use_strong_aug: bool) -> transforms.Compose:
    """Return data transform with or without strong augmentation."""
    if use_strong_aug:
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def get_val_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def prepare_dataloaders(config: Config, current_epoch: int = 0) -> Tuple[DataLoader, DataLoader, int]:
    """Create train/validation dataloaders with warm-up aware augmentation."""
    use_strong_aug = current_epoch >= config.WARMUP_EPOCHS
    train_transform = get_train_transform(config.INPUT_SIZE, use_strong_aug)
    val_transform = get_val_transform(config.INPUT_SIZE)

    # 其余部分保持一致...

    dataset = datasets.ImageFolder(config.DATA_DIR)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED),
    )

    # Apply transforms dynamically
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

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
