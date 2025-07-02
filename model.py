import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class AIModel(nn.Module):
    """EfficientNet model with added Dropout for regularization."""
    def __init__(self, num_classes, efficientnet_type="efficientnet-b0", dropout_rate=0.5):
        super().__init__()
        # Load pretrained EfficientNet backbone without classifier
        self.backbone = EfficientNet.from_pretrained(efficientnet_type)

        # Replace the classifier with Dropout + new FC layer
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()  # Remove the original FC layer
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone.extract_features(x)
        x = self.backbone._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
