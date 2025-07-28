# models/model.py

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForImageClassification
from config import Config

class TransFGDeiT(nn.Module):
    """
    TransFG 基于 HuggingFace 的 DeiT-base-patch16-224（ImageNet‑1k）：
    - 用 AutoModelForImageClassification 加载预训练模型
    - 返回 attentions 和 hidden_states
    - 选 top-k patch embeddings 融合 [CLS] 进行分类
    """
    def __init__(self, num_classes: int):
        super().__init__()
        cfg = Config()

        # 1) 加载配置，开启 attentions 与 hidden_states
        model_cfg = AutoConfig.from_pretrained(
            'facebook/deit-base-patch16-224',
            output_attentions=True,
            output_hidden_states=True,
            num_labels=num_classes
        )
        # 2) 加载模型，忽略分类头尺寸不匹配
        self.deit = AutoModelForImageClassification.from_pretrained(
            'facebook/deit-base-patch16-224',
            config=model_cfg,
            ignore_mismatched_sizes=True
        )
        D = model_cfg.hidden_size  # 768

        # 3) 部件融合 MLP
        self.num_parts = cfg.NUM_PARTS
        self.part_mlp  = nn.Sequential(
            nn.Linear(D * self.num_parts, D),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.PART_DROPOUT)
        )

        # 4) 最终融合分类头
        self.classifier = nn.Linear(2 * D, num_classes)

    def forward(self, x: torch.Tensor):
        """
        Args:
          x: [B,3,224,224], 已由 DataLoader 归一化
        Returns:
          logits: [B, num_classes]
          aux: dict {
            'cls': [B, D],
            'part': [B, D],
            'attn_scores': [B, N]
          }
        """
        # 1) 前向：得到 logits、attentions、hidden_states
        outputs = self.deit(
            pixel_values=x,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = outputs.hidden_states[-1]   # [B, 1+N, D]
        attns         = outputs.attentions         # tuple len L, each [B, heads, 1+N,1+N]

        # 2) 拆分 CLS 与 patches
        cls_embed    = hidden_states[:, 0]         # [B, D]
        patch_embeds = hidden_states[:, 1:]        # [B, N, D]

        # 3) 计算 cls→patch 平均注意力
        a_last       = attns[-1]                   # [B, heads, L, L]
        cls2patch    = a_last[:, :, 0, 1:]         # [B, heads, N]
        attn_scores  = cls2patch.mean(dim=1)       # [B, N]

        # 4) 选 top-k patches
        k            = self.num_parts
        topk_idx     = torch.topk(attn_scores, k, dim=1).indices  # [B, k]

        # 5) 收集对应 patch embeddings
        B, N, D      = patch_embeds.shape
        idx_exp      = topk_idx.unsqueeze(-1).expand(-1, -1, D)  # [B, k, D]
        parts        = patch_embeds.gather(1, idx_exp)           # [B, k, D]

        # 6) 融合 part 特征
        parts_flat   = parts.reshape(B, k * D)                   # [B, k*D]
        part_feat    = self.part_mlp(parts_flat)                 # [B, D]

        # 7) 最终分类
        fusion       = torch.cat([cls_embed, part_feat], dim=1)  # [B, 2D]
        logits       = self.classifier(fusion)                   # [B, num_classes]

        return logits, {
            'cls': cls_embed,
            'part': part_feat,
            'attn_scores': attn_scores
        }
