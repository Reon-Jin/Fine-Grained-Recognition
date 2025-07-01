# model.py
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
from torchvision.ops import roi_align


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


class ChannelAttention(nn.Module):
    """Channel attention used in CBAM."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


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
    """Convolutional Block Attention Module."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * self.channel(x)
        out = out * self.spatial(out)
        return out


class LightweightRPN(nn.Module):
    """Simplified RPN generating rough region proposals."""

    def __init__(self, channels: int, num_proposals: int = 4, roi_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.box_pred = nn.Conv2d(channels // 2, num_proposals * 4, kernel_size=1)
        self.part_head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.num_proposals = num_proposals
        self.roi_size = roi_size

    def forward(self, feat: torch.Tensor):
        b, c, h, w = feat.shape
        x = self.conv(feat)
        box = self.box_pred(x)
        box = torch.sigmoid(nn.functional.adaptive_avg_pool2d(box, 1))
        box = box.view(b, self.num_proposals, 4)

        # scale to feature map size
        boxes = box.clone()
        boxes[..., [0, 2]] *= w
        boxes[..., [1, 3]] *= h

        # format boxes for roi_align
        boxes_formatted = []
        for i in range(b):
            for j in range(self.num_proposals):
                coord = torch.tensor([i, boxes[i, j, 0], boxes[i, j, 1], boxes[i, j, 2], boxes[i, j, 3]], device=feat.device)
                boxes_formatted.append(coord)
        boxes_formatted = torch.stack(boxes_formatted)

        roi_feats = roi_align(feat, boxes_formatted, output_size=self.roi_size)
        roi_feats = self.part_head(roi_feats)
        roi_feats = roi_feats.view(b, self.num_proposals, c)
        # aggregate proposals
        local_feat = roi_feats.mean(dim=1)
        return local_feat, boxes

class AIModel(nn.Module):
    def __init__(self, arch: str = 'efficientnet-b2', num_classes: int = 400,
                 embed_dim: int = 256, num_layers: int = 4, num_heads: int = 8,
                 num_proposals: int = 4) -> None:
        super().__init__()
        # ImageNet-1k 预训练
        self.backbone = EfficientNet.from_pretrained(arch)
        # Attention modules
        self.attention = SEBlock(self.backbone._conv_head.out_channels)
        self.cbam = CBAM(self.backbone._conv_head.out_channels)
        self.rpn = LightweightRPN(self.backbone._conv_head.out_channels, num_proposals=num_proposals)

        self.embed_conv = nn.Conv2d(
            self.backbone._conv_head.out_channels, embed_dim, kernel_size=1
        )
        self.pos_embed = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.local_fc = nn.Linear(self.backbone._conv_head.out_channels, embed_dim)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = self.backbone._dropout
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_boxes: bool = False):
        # Backbone feature extraction with attention modules
        x = self.backbone.extract_features(x)
        x = self.attention(x)
        x = self.cbam(x)

        # Local region proposals and features
        local_feat, boxes = self.rpn(x)
        local_feat = self.local_fc(local_feat)

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

        # Fuse global and local features
        feat = global_feat + local_feat
        out = self.classifier(feat)
        if return_boxes:
            return out, boxes
        return out
