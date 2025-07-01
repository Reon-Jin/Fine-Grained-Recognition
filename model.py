# model.py
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """1D Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 10000) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, n, d)
        if x.size(1) > self.pe.size(1):
            raise ValueError("Sequence length exceeds maximum length")
        return x + self.pe[:, :x.size(1), :]




class SpatialAttention(nn.Module):
    """Spatial attention used in CBAM."""

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
    """CBAM with only spatial attention."""

    def __init__(self) -> None:
        super().__init__()
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.spatial(x)



class AIModel(nn.Module):
    def __init__(self, arch: str = 'efficientnet-b0', num_classes: int = 400,
                 embed_dim: int = 256, num_layers: int = 4, num_heads: int = 8) -> None:
        super().__init__()
        # ImageNet-1k 预训练
        self.backbone = EfficientNet.from_pretrained(arch)
        # Spatial attention
        self.cbam = CBAM()

        self.embed_conv = nn.Conv2d(
            self.backbone._conv_head.out_channels, embed_dim, kernel_size=1
        )
        self.pos_embed = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = self.backbone._dropout
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Backbone feature extraction with spatial attention
        x = self.backbone.extract_features(x)
        x = self.cbam(x)

        # Flatten spatial dims and add positional information
        x = self.embed_conv(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (b, h*w, c)
        x = self.pos_embed(x)

        # Transformer encoder
        x = self.transformer(x)

        # Pool global features
        x = x.transpose(1, 2)  # (b, c, h*w)
        global_feat = self.pool(x).squeeze(-1)
        global_feat = self.dropout(global_feat)
        return self.classifier(global_feat)
