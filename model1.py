import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BaseModel(nn.Module):
    """First‑stage model that also exports a simple attention map."""
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        # torchvision >= 0.15 uses weights enums
        if pretrained:
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            backbone = models.resnet50(weights=None)

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

    def forward(self, x, return_attention: bool = False):
        feats = self.features(x)                  # B × C × H × W
        if return_attention:
            # channel‑wise mean as a surrogate CAM
            attn = torch.mean(feats, dim=1, keepdim=True)  # B×1×H×W
            attn = F.relu(attn)
            attn = F.interpolate(attn, size=x.shape[-2:], mode='bilinear', align_corners=False)
        pooled = self.avgpool(feats)
        logits = self.fc(torch.flatten(pooled, 1))
        if return_attention:
            return logits, attn
        return logits

class GCELoss(nn.Module):
    """Generalized Cross‑Entropy Loss (q‑parameter)."""
    def __init__(self, q: float = 0.7):
        super().__init__()
        self.q = q

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1) + 1e-6
        if self.q == 0:
            loss = -torch.log(probs)
        else:
            loss = (1. - probs ** self.q) / self.q
        return torch.mean(loss)
