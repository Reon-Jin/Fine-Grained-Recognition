# model1.py

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models.efficientnet import SqueezeExcitation
from config import Config

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 从 Config 读取 Top-K、类别数
        self.k = Config.K
        num_classes = Config.NUM_CLASSES

        # 1) 加载预训练骨架，并替换分类头
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=self.backbone.classifier[0].p),
            nn.Linear(in_feats, num_classes)
        )

        # 2) 注册钩子，自动提取所有 SE 模块的通道注意力
        self.att_weights = []
        for m in self.backbone.features.modules():
            if isinstance(m, SqueezeExcitation):
                m.register_forward_hook(self._hook_se)

    def _hook_se(self, module, input, output):
        # output: [B, C] 通道重标度权重
        self.att_weights.append(output.detach())

    def forward(self, x, labels=None):
        # 清空上次 attention
        self.att_weights = []
        logits = self.backbone(x)  # [B, NUM_CLASSES]

        if labels is not None:
            # Top-K 判断
            topk_inds = logits.topk(self.k, dim=1).indices       # [B, K]
            correct   = topk_inds.eq(labels.view(-1,1)).any(dim=1)  # [B]
            return logits, correct.float(), self.att_weights
        else:
            return logits, self.att_weights

def compute_topk_accuracy(logits, labels, k):
    topk  = logits.topk(k, dim=1).indices       # [B, k]
    corr  = topk.eq(labels.view(-1,1)).any(dim=1).float()
    return corr.mean().item()
