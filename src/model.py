import torch
import torch.nn as nn
from torchvision import models
from .config import NUM_CLASSES

class ResNet50Feat(nn.Module):
    """
    预训练 ResNet50 去掉最后 FC，只输出 2048-d 特征
    """
    def __init__(self):
        super().__init__()
        net = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x):
        f = self.backbone(x)          # [B,2048,1,1]
        return f.view(f.size(0), -1)  # [B,2048]

class PartialResNet(nn.Module):
    """
    完整分类模型: ResNet50 + 新 fc 层
    """
    def __init__(self):
        super().__init__()
        net = models.resnet50(pretrained=False)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        in_feats = net.fc.in_features
        self.fc = nn.Linear(in_feats, NUM_CLASSES)

    def forward(self, x):
        f = self.backbone(x).view(x.size(0), -1)
        return self.fc(f)
