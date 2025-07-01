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



class SEBlock(nn.Module):
    """Squeeze-and-Excitation block used as a lightweight attention module."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class AIModel(nn.Module):
    def __init__(self, arch: str = 'efficientnet-b2', num_classes: int = 400,
                 embed_dim: int = 256, num_layers: int = 2, num_heads: int = 8) -> None:
        super().__init__()
        # ImageNet-1k 预训练
        self.backbone = EfficientNet.from_pretrained(arch)
        # Attention module after feature extraction
        self.attention = SEBlock(self.backbone._conv_head.out_channels)

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
        # Backbone feature extraction with attention
        x = self.backbone.extract_features(x)
        x = self.attention(x)

        # Flatten spatial dims and add positional information
        x = self.embed_conv(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (b, h*w, c)
        x = self.pos_embed(x)

        # Transformer encoder
        x = self.transformer(x)

        # Pool and classify
        x = x.transpose(1, 2)  # (b, c, h*w)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
