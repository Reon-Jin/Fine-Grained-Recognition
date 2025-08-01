"""Classification heads for Co-Teaching + SVM (Route B).

This replaces the original ROIModel → FC → CE/GCE pipeline fileciteturn0file0.
We keep the same feature extractor on 7×7 maps but return a flattened vector,
then apply a linear multi-class SVM layer trained with hinge (MultiMargin) loss.

Usage (training loop snippet) – two heads A / B for Co-Teaching:

    headA = CoTeachHead(num_classes)
    headB = CoTeachHead(num_classes)
    criterion = nn.MultiMarginLoss(p=1, margin=1.0, reduction="none")

    logitsA, featA = headA(feat_map.detach())
    logitsB, featB = headB(feat_map.detach())
    lossesA = criterion(logitsA, labels)
    lossesB = criterion(logitsB, labels)
    ...  # same sorting / mutual update logic as before

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ROIModelFeat",
    "LinearSVM",
    "CoTeachHead",
]

# -----------------------------------------------------------------------------
# Feature extractor: identical to old ROIModel.head but **no FC**, returns vector
# -----------------------------------------------------------------------------
class ROIModelFeat(nn.Module):
    """Lightweight head that turns (B, C_in, 7, 7) → (B, feat_dim)."""

    def __init__(self, in_channels: int = 1280, feat_dim: int = 128):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):  # x: [B, C_in, 7, 7]
        x = self.block(x)       # [B, feat_dim, 1, 1]
        return x.flatten(1)     # [B, feat_dim]

# -----------------------------------------------------------------------------
# Linear multi-class SVM layer (one-vs-rest hinge margins)
# -----------------------------------------------------------------------------
class LinearSVM(nn.Module):
    """y = W·x  (no bias – absorbed in extra column if wanted)"""

    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim))
        # Kaiming uniform init similar to nn.Linear default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):  # x: [B, feat_dim]
        return F.linear(x, self.weight)  # logits ≈ margins, shape [B, num_classes]

# -----------------------------------------------------------------------------
# Convenience wrapper: feature extractor + SVM, drop-in replacement for ROIModel
# -----------------------------------------------------------------------------
class CoTeachHead(nn.Module):
    """Feature extractor + linear SVM. Returns (logits, feats)."""

    def __init__(self, num_classes: int, in_channels: int = 1280, feat_dim: int = 128):
        super().__init__()
        self.feat = ROIModelFeat(in_channels, feat_dim)
        self.svm = LinearSVM(feat_dim, num_classes)

    def forward(self, x):
        f = self.feat(x)
        logits = self.svm(f)
        return logits, f

