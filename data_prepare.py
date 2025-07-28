# data_prepare.py

import os
from typing import Tuple

import torch
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandAugment
from torch.utils.data import DataLoader, random_split

from config import Config


def get_train_transform(input_size: int, use_strong_aug: bool) -> transforms.Compose:
    """Return data transform with or without strong augmentation."""
    if use_strong_aug:
        cfg = Config()
        # 基础几何和颜色增强
        aug_list = [
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        ]
        # 根据 config 决定是否加入 RandAugment
        if getattr(cfg, 'USE_RANDAUGMENT', False):
            aug_list.append(
                RandAugment(num_ops=cfg.RANDAUG_N, magnitude=cfg.RANDAUG_M)
            )
        # 最后统一做 ToTensor 和 Normalize
        aug_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.NORMALIZE_MEAN,
                                 std=cfg.NORMALIZE_STD)
        ]
        return transforms.Compose(aug_list)
    else:
        # 简单增强流程
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN,
                                 std=Config.NORMALIZE_STD)
        ])


def get_val_transform(input_size: int) -> transforms.Compose:
    """Validation transform (no randomness)."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORMALIZE_MEAN,
                             std=Config.NORMALIZE_STD)
    ])


def prepare_dataloaders(config: Config, current_epoch: int = 0) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create train/validation dataloaders with warm-up aware augmentation.
    :param config: 全局配置
    :param current_epoch: 用于判断是否跳过 warm-up 阶段
    :return: train_loader, val_loader, num_classes
    """
    use_strong_aug   = current_epoch >= config.WARMUP_EPOCHS
    train_transform  = get_train_transform(config.INPUT_SIZE, use_strong_aug)
    val_transform    = get_val_transform(config.INPUT_SIZE)

    # 加载数据集并切分
    dataset   = datasets.ImageFolder(config.DATA_DIR)
    val_size  = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED),
    )

    # 动态应用 transform
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform   = val_transform

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, len(dataset.classes)
