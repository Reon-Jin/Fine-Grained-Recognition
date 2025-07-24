import torch
from torch import nn
import torch.nn.functional as F

def hw_flatten(x):
    # x: [B, C, H, W] → [B, H*W, C]
    return x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)

class SelfAttention(nn.Module):
    """Self-attention layer compatible with PyTorch."""

    def __init__(self, filters):
        super().__init__()
        self.filters = filters
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, img):
        # img: [B, C, H, W]
        B, C, H, W = img.shape
        f_flat = hw_flatten(img)
        g_flat = hw_flatten(img)
        h_flat = hw_flatten(img)

        s = torch.matmul(g_flat, f_flat.transpose(1, 2))  # [B, N, N]
        beta = F.softmax(s, dim=-1)
        o = torch.matmul(beta, h_flat)  # [B, N, C]
        o = o.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return self.gamma * o + img


