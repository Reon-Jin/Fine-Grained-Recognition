import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import timm

from RoiPoolingConvTF2 import RoiPooling
from SelfAttention import SelfAttention
from cbam import CBAM
from utils import getROIS, crop, squeezefunc, stackfunc


class SR_GNN(nn.Module):
    """Simplified SR-GNN model implemented in PyTorch."""

    def __init__(self, pool_size=7, ROIS_resolution=42, ROIS_grid_size=3, minSize=2, nb_classes=1000):
        super().__init__()
        self.base = timm.create_model('xception', pretrained=True, features_only=True)
        self.base_channels = self.base.feature_info[-1]['num_chs']
        self.rois = getROIS(resolution=ROIS_resolution, gridSize=ROIS_grid_size, minSize=minSize)
        self.roi_pool = RoiPooling(pool_size, self.rois)
        feat_dim = self.base_channels  # ⚠️ 修复点：最终特征维度为 C 而不是 C*H*W
        self.attention = SelfAttention(self.base_channels)
        self.fc = nn.Linear(feat_dim, nb_classes)

    def forward(self, x):
        features = self.base(x)[-1]  # [B, C, H, W]
        resized = F.interpolate(
            features,
            size=(self.roi_pool.pool_size * 2, self.roi_pool.pool_size * 2),
            mode='bilinear', align_corners=False
        )
        rois = self.roi_pool(resized)  # [B, N_rois, C, H, W]

        attn_outputs = []
        for i in range(rois.size(1)):
            roi = rois[:, i, :, :, :]  # [B, C, H, W]
            attn_roi = self.attention(roi)  # [B, C, H, W]
            pooled = F.adaptive_avg_pool2d(attn_roi, 1).squeeze(-1).squeeze(-1)  # [B, C]
            attn_outputs.append(pooled)

        attn = torch.stack(attn_outputs, dim=1).mean(dim=1)  # [B, C]
        out = self.fc(attn)  # [B, nb_classes]
        return out


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
    if name == 'srgnn':
        return SR_GNN(**kwargs)
    if name == 'cbam_resnet':
        backbone = kwargs.pop('backbone', 'resnet50')
        return CBAMResNet(backbone=backbone, **kwargs)
    raise ValueError('Model %s not found' % name)
