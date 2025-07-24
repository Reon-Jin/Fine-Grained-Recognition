import torch
from torch import nn
import torch.nn.functional as F

class RoiPooling(nn.Module):
    def __init__(self, pool_size, rois, downsample_ratio=16):
        super().__init__()
        self.pool_size = pool_size
        self.downsample_ratio = downsample_ratio
        self.register_buffer('rois', torch.tensor(rois, dtype=torch.int))

    def forward(self, x):  # x: [B, C, H, W]
        outputs = []
        B, C, H, W = x.shape
        scale = 1.0 / self.downsample_ratio

        for idx, roi in enumerate(self.rois):
            x0, y0, w, h = roi.tolist()
            x1 = int(round(x0 * scale))
            y1 = int(round(y0 * scale))
            x2 = int(round((x0 + w) * scale))
            y2 = int(round((y0 + h) * scale))

            if x1 < 0 or y1 < 0 or x2 > W or y2 > H or x2 - x1 < 1 or y2 - y1 < 1:
                print(f"[Warning] ROI {idx} skipped: invalid (x1={x1}, y1={y1}, x2={x2}, y2={y2}), feature map: ({H}, {W})")
                continue

            region = x[:, :, y1:y2, x1:x2]
            pooled = F.interpolate(region, size=(self.pool_size, self.pool_size), mode='bilinear', align_corners=False)
            outputs.append(pooled)

        if len(outputs) == 0:
            raise RuntimeError("No valid ROIs found for pooling.")

        return torch.stack(outputs, dim=1)  # [B, N_rois, C, ps, ps]

