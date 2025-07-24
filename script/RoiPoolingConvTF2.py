import torch
from torch import nn
import torch.nn.functional as F


class RoiPooling(nn.Module):
    """Simple ROI pooling using bilinear resize."""

    def __init__(self, pool_size, rois):
        super().__init__()
        self.pool_size = pool_size
        self.register_buffer('rois', torch.tensor(rois, dtype=torch.int))

    def forward(self, x):
        outputs = []
        for roi in self.rois:
            x1, y1, w, h = roi.tolist()
            region = x[:, y1:y1 + h, x1:x1 + w, :]
            pooled = F.interpolate(region, size=(self.pool_size, self.pool_size), mode='bilinear', align_corners=False)
            outputs.append(pooled)
        return torch.stack(outputs, dim=1)

