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

        # 1) 共享骨干网络 (EfficientNet‑B0)
        self.base = models.efficientnet_b0(weights='DEFAULT')

        # 2) 全局 CBAM 注意力
        self.global_cbam = CBAM(1280)

        # 3) 轻量级 Transformer 自注意力，参数从 Config 中读取
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1280,
            nhead=cfg.TRANSFORMER_NHEADS,
            dim_feedforward=cfg.TRANSFORMER_FF_DIM,
            dropout=cfg.HEAD_DROPOUT,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.TRANSFORMER_LAYERS
        )

        # 4) 全局池化
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 5) 划分 M×M 网格 → M^2 个块
        self.grid_size = cfg.GRID_SIZE
        self.num_blocks = self.grid_size * self.grid_size

        # 6) 多头门控 MLP
        self.heads = cfg.BLOCK_HEADS
        self.block_gate = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.HEAD_DROPOUT),
            nn.Linear(512, self.heads)
        )

        # 7) 头融合 & 融合瓶颈
        self.head_fuse = nn.Sequential(
            nn.Linear(self.heads * 1280, 1280),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.HEAD_DROPOUT)
        )
        self.fuse_bottleneck = nn.Sequential(
            nn.Linear(1280 * 2, 1280),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.HEAD_DROPOUT)
        )

        # 8) 分类头
        self.aux_dropout = nn.Dropout(cfg.AUX_DROPOUT)
        self.aux_fc      = nn.Linear(1280, num_classes)
        self.fc          = nn.Linear(1280, num_classes)

        # 温度系数
        self.temp = cfg.ATTN_TEMPERATURE
        self.last_head_weights = None

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        # Backbone + CBAM
        feat = self.base.features(x)
        feat = self.global_cbam(feat)

        # Transformer 自注意力
        B, Cf, Hf, Wf = feat.shape
        seq = feat.view(B, Cf, -1).permute(2, 0, 1)
        seq = self.transformer(seq)
        feat = seq.permute(1, 2, 0).view(B, Cf, Hf, Wf)

        # 全局池化
        feat_global = self.pool(feat).view(B, -1)

        # 网格切分 & 块内池化
        gh, gw = Hf // self.grid_size, Wf // self.grid_size
        blocks = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y1, y2 = i*gh, (i+1)*gh
                x1, x2 = j*gw, (j+1)*gw
                p = feat[:, :, y1:y2, x1:x2]
                p = F.adaptive_avg_pool2d(p, 1).view(B, -1)
                blocks.append(p)
        feats = torch.stack(blocks, dim=1)

        # 多头门控 + Softmax
        flat_scores = self.block_gate(feats.view(-1, 1280))
        scores = flat_scores.view(B, self.num_blocks, self.heads).permute(0,2,1)
        scores = scores / (self.temp + 1e-8)
        wts = F.softmax(scores, dim=2)
        self.last_head_weights = wts.detach()

        # 聚合局部特征
        feats_exp   = feats.unsqueeze(1).expand(-1, self.heads, -1, -1)
        local_heads = (wts.unsqueeze(-1) * feats_exp).sum(dim=2)
        feat_local  = self.head_fuse(local_heads.view(B, -1))

        # 辅助分类头
        aux_logits = self.aux_fc(self.aux_dropout(feat_local))

        # 融合全局 + 局部
        fused      = torch.cat([feat_global, feat_local], dim=1)
        fused      = self.fuse_bottleneck(fused)

        # 最终分类
        main_logits = self.fc(fused)
        return main_logits, aux_logits, wts
