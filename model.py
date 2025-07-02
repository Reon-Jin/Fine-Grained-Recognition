import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class AIModel(nn.Module):
    """Single EfficientNet model for multi-class classification."""
    def __init__(self, num_classes, efficientnet_type="efficientnet-b0"):
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained(
            efficientnet_type,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.efficientnet(x)
