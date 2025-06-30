# model.py
from efficientnet_pytorch import EfficientNet
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block used as a lightweight attention module."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class AIModel(nn.Module):
    def __init__(self, arch: str = 'efficientnet-b0', num_classes: int = 400) -> None:
        super().__init__()
        # ImageNet-1k 预训练
        self.backbone = EfficientNet.from_pretrained(arch)
        # Attention module after feature extraction
        self.attention = SEBlock(self.backbone._conv_head.out_channels)
        in_features = self.backbone._fc.in_features
        # 输出改为多个类别
        self.backbone._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Copy of EfficientNet.forward with attention inserted
        x = self.backbone.extract_features(x)
        x = self.attention(x)
        x = self.backbone._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.backbone._dropout(x)
        x = self.backbone._fc(x)
        return x
