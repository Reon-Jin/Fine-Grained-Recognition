# models/model.py

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from config import Config
from models.cbam import CBAM

class FineGrainedModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        cfg = Config()

        # 1) 共享骨干网络
        self.base        = models.efficientnet_b0(weights='DEFAULT')

        # 2) 全局分支 CBAM 注意力
        self.global_cbam = CBAM(1280)

        # 3) 池化到 [B,1280]
        self.pool        = nn.AdaptiveAvgPool2d(1)

        # 4) 划分 M×M 网格 → M^2 个块
        self.grid_size   = cfg.GRID_SIZE
        self.num_blocks  = self.grid_size * self.grid_size

        # 5) 多头门控 MLP：每个块的 1280 特征 输出 H 个 score
        self.heads       = cfg.BLOCK_HEADS
        self.block_gate  = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.heads)
        )

        # 6) 头融合瓶颈：将 H×1280 拼接 → 1280
        self.head_fuse   = nn.Sequential(
            nn.Linear(self.heads * 1280, 1280),
            nn.ReLU(inplace=True)
        )

        # 7) 融合瓶颈：全局(1280)+局部(1280) → 1280
        self.fuse_bottleneck = nn.Sequential(
            nn.Linear(1280 * 2, 1280),
            nn.ReLU(inplace=True),
        )

        # 8) 辅助分类头（仅用局部特征）
        self.aux_fc      = nn.Linear(1280, num_classes)
        # 9) 最终分类头
        self.fc          = nn.Linear(1280, num_classes)

        # 温度系数，用于调节块级 softmax
        self.temp        = cfg.ATTN_TEMPERATURE

        # 存储最后一次的各头权重 [B, heads, num_blocks]
        self.last_head_weights = None

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B,3,H,W]
        Returns:
            main_logits: [B,num_classes]
            aux_logits:  [B,num_classes]
            head_wts:    [B, heads, num_blocks]
        """
        B, C, H, W = x.shape

        # ———— 1) 骨干 + 全局 CBAM ————
        feat = self.base.features(x)                  # [B,1280,hf,wf]
        feat = self.global_cbam(feat)                 # 通道 + 空间 注意力
        feat_global = self.pool(feat).view(B, -1)     # [B,1280]

        # ———— 2) 4×4 网格切分 & 块内池化 ————
        _, _, hf, wf = feat.shape
        gh, gw = hf // self.grid_size, wf // self.grid_size

        blocks = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y1, y2 = i*gh, (i+1)*gh
                x1, x2 = j*gw, (j+1)*gw
                p = feat[:, :, y1:y2, x1:x2]                        # [B,1280,gh,gw]
                p = F.adaptive_avg_pool2d(p, 1).view(B, -1)         # [B,1280]
                blocks.append(p)
        # → [B, num_blocks, 1280]
        feats = torch.stack(blocks, dim=1)

        # ———— 3) 多头门控 + 带温度的 Softmax ————
        flat_scores = self.block_gate(feats.view(-1, 1280))   # [B*num_blocks, heads]
        scores      = flat_scores.view(B, self.num_blocks, self.heads)  # [B, num_blocks, heads]
        scores      = scores.permute(0,2,1)                   # [B, heads, num_blocks]
        scores      = scores / (self.temp + 1e-8)
        wts         = F.softmax(scores, dim=2)                # [B, heads, num_blocks]
        self.last_head_weights = wts.detach()

        # ———— 4) 按头加权聚合 → 各头局部特征 ————
        # feats: [B, num_blocks, 1280]  → expand heads axis
        feats_exp   = feats.unsqueeze(1).expand(-1, self.heads, -1, -1)  # [B, heads, num_blocks, 1280]
        local_heads = (wts.unsqueeze(-1) * feats_exp).sum(dim=2)         # [B, heads, 1280]

        # ———— 5) 融合各头局部特征 ————
        # 拼接所有头 → [B, heads*1280]
        concat_heads = local_heads.view(B, -1)
        feat_local   = self.head_fuse(concat_heads)                      # [B,1280]

        # ———— 6) 辅助分类头 ————
        aux_logits   = self.aux_fc(feat_local)                           # [B,num_classes]

        # ———— 7) 全局+局部 融合 ————
        fused        = self.fuse_bottleneck(torch.cat([feat_global, feat_local], dim=1))  # [B,1280]

        # ———— 8) 最终分类头 ————
        main_logits  = self.fc(fused)                                    # [B,num_classes]

        return main_logits, aux_logits, wts
