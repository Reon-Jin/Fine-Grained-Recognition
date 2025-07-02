import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def create_swin_features(unfreeze_last_stage=False):
    """
    Create a Swin-Tiny feature extractor with features_only.
    If unfreeze_last_stage is True, unfreeze the final transformer stage and its norm layer.
    """
    model = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=True,
        features_only=True,
        out_indices=(3,)
    )
    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    if unfreeze_last_stage:
        # Unfreeze the last transformer stage and normalization
        for name, p in model.named_parameters():
            if name.startswith('layers.3') or 'norm' in name:
                p.requires_grad = True
    return model


class BCNN(nn.Module):
    def __init__(self, num_classes=400, drop_prob=0.5, unfreeze_last_stage=False):
        super().__init__()
        # Two feature extractors: one fine-tuned, one frozen
        self.f1 = create_swin_features(unfreeze_last_stage=unfreeze_last_stage)
        self.f2 = create_swin_features(unfreeze_last_stage=False)

        # Use LazyLinear to adapt to actual bilinear feature size
        self.fc = nn.Sequential(
            nn.Dropout(p=drop_prob),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x):
        # branch 1
        x1 = self.f1(x)[0]
        # branch 2
        x2 = self.f2(x)[0]

        # reshape to [B, C, N]
        B, C, H, W = x1.size()
        x1 = x1.view(B, C, H * W)
        x2 = x2.view(B, C, H * W)

        # bilinear pooling: compute outer product at each spatial location and average
        phi = torch.bmm(x1, x2.transpose(1, 2)) / (H * W)  # [B, C, C]

        # flatten
        phi = phi.view(B, C * C)

        # signed square-root normalization and L2 normalization
        phi = torch.sign(phi) * torch.sqrt(torch.abs(phi) + 1e-12)
        phi = F.normalize(phi, dim=1)

        # final classification
        out = self.fc(phi)
        return out