# model.py -- Vision Transformer ViT-B/16 fine‑tuning model
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class AIModel(nn.Module):
    """Vision Transformer (ViT‑B/16) with ImageNet‑1k pre‑training."""

    def __init__(self, arch: str = "vit-b/16", num_classes: int = 400):
        super().__init__()

        valid_aliases = {"vit-b/16", "vit_b_16", "vit_base_patch16_224.augreg_in1k"}
        if arch.lower() not in valid_aliases:
            raise ValueError(f"Unsupported ViT variant: {arch}. "
                             f"Supported aliases: {valid_aliases}")

        # Backbone with pre‑trained weights
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.backbone = vit_b_16(weights=weights)

        # Replace classification head
        in_features = self.backbone.heads.head.in_features  # type: ignore[attr-defined]
        self.backbone.heads.head = nn.Linear(in_features, num_classes)  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor):
        return self.backbone(x)
