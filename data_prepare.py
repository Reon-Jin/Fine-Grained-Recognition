# data_prepare.py

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models.efficientnet import EfficientNet_B0_Weights
from config import Config


def prepare_dataloaders():
    """
    构建训练和验证 DataLoader。
    使用 Config 中定义的：
      - DATA_DIR: 数据根目录，子文件夹为类别
      - VAL_SPLIT: 验证集比例
      - SEED: 随机种子
      - BATCH_SIZE: 批大小
      - NUM_WORKERS: DataLoader 并行进程数
    返回：
      train_loader, val_loader, num_classes
    """

    # 1. 预训练权重自带的标准预处理
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    # 2. 加载整个数据集
    dataset = ImageFolder(root=Config.DATA_DIR, transform=transform)
    num_classes = len(dataset.classes)

    # 3. 划分训练/验证集
    total = len(dataset)
    val_size = int(Config.VAL_SPLIT * total)
    train_size = total - val_size
    generator = torch.Generator().manual_seed(Config.SEED)
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    # 4. 构造 DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader, num_classes
