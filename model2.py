import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ROIModel(nn.Module):
    """Second‑stage classifier that focuses on high‑response regions."""
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        if pretrained:
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            backbone = models.resnet50()

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.avgpool = backbone.avgpool
        self.fc      = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x, attn_mask=None):
        """Forward pass.

        Args:
            x (Tensor): B×3×H×W image
            attn_mask (Tensor|None): B×1×H×W mask, values in [0,1]
        """
        if attn_mask is not None:
            # upsample mask to input size then apply residual emphasis
            attn_mask = F.interpolate(attn_mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = x * attn_mask + x  # residual so unmasked pixels still contribute

        feats = self.features(x)
        pooled = self.avgpool(feats)
        logits = self.fc(torch.flatten(pooled, 1))
        return logits
