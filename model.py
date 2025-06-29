# model.py
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

class AIModel(nn.Module):
    def __init__(self, arch='efficientnet-b0', num_classes=400):
        super().__init__()
        # ImageNet-1k 预训练
        self.backbone = EfficientNet.from_pretrained(arch)
        in_features = self.backbone._fc.in_features
        # 输出改为多个类别
        self.backbone._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)