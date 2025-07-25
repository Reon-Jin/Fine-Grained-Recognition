import torch
from torch import nn
import torch.nn.functional as F
import timm

from cbam import CBAM




class CBAMResNet(nn.Module):
    """ResNet backbone with CBAM attention."""

    def __init__(self, backbone='resnet50', nb_classes=1000):
        super().__init__()
        self.base = timm.create_model(backbone, pretrained=True, features_only=True)
        self.in_ch = self.base.feature_info[-1]['num_chs']
        self.cbam = CBAM(self.in_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.in_ch, nb_classes)

    def forward(self, x):
        feat = self.base(x)[-1]  # [B, C, H, W]
        feat = self.cbam(feat)
        pooled = self.pool(feat).view(x.size(0), -1)
        return self.fc(pooled)


def construct_model(name, **kwargs):
    if name == 'cbam_resnet':
        backbone = kwargs.pop('backbone', 'resnet50')
        return CBAMResNet(backbone=backbone, **kwargs)
    raise ValueError('Model %s not found' % name)
