# model2.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ROIModel(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1280):
        super().__init__()
        # 轻量分类头：先用 1x1把通道压到512，再一个3x3细化，最后GAP+FC
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):  # x: [B, 1280, 7, 7]
        x = self.head(x)   # -> [B, 128, 1, 1]
        x = x.flatten(1)   # -> [B, 128]
        x = self.fc(x)     # -> [B, num_classes]
        return x

