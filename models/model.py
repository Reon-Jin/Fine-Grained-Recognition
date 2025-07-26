import torch.nn as nn
from torchvision import models


class FineGrainedModel(nn.Module):
    """Simple classifier using EfficientNet-B0 backbone."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
