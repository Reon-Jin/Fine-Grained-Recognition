# model.py
import torch
import torch.nn as nn
from torchvision import models


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError("Sequence length exceeds maximum length")
        return x + self.pe[:, :x.size(1), :]


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes: int) -> None:
        super().__init__()
        self.channel = ChannelAttention(in_planes)
        self.spatial = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x = x * self.channel(x)
        x = x * self.spatial(x)
        return x


class AIModel(nn.Module):
    def __init__(self, num_classes: int = 400, embed_dim: int = 256,
                 num_layers: int = 4, num_heads: int = 8) -> None:
        super().__init__()

        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        self.backbone = models.convnext_tiny(weights=weights)
        out_channels = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity()

        self.cbam_backbone = CBAM(in_planes=out_channels)
        self.embed_conv = nn.Conv2d(out_channels, embed_dim, kernel_size=1)
        self.cbam_transformer = CBAM(in_planes=embed_dim)
        self.pos_embed = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.cbam_backbone(x)  # CBAM after backbone
        x = self.embed_conv(x)
        x = self.cbam_transformer(x)  # CBAM before transformer
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (b, h*w, c)
        x = self.pos_embed(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)  # (b, c, h*w)
        global_feat = self.pool(x).squeeze(-1)
        global_feat = self.dropout(global_feat)
        return self.classifier(global_feat)
