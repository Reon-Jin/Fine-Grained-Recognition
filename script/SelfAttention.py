import torch
from torch import nn
import torch.nn.functional as F


def hw_flatten(x):
    return x.view(x.size(0), -1, x.size(-1))


class SelfAttention(nn.Module):
    """Self-attention layer compatible with PyTorch."""

    def __init__(self, filters):
        super().__init__()
        self.filters = filters
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, img, f, g, h):
        s = torch.matmul(hw_flatten(g), hw_flatten(f).transpose(1, 2))
        beta = F.softmax(s, dim=-1)
        o = torch.matmul(beta, hw_flatten(h))
        o = o.view(img.size(0), img.size(1), img.size(2), self.filters)
        return self.gamma * o + img

