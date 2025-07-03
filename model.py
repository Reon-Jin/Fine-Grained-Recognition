import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import torchvision.models as models

class BilinearCNN(nn.Module):
    """
    双线性 CNN：Stream1 用 EfficientNet-B3，Stream2 用 ResNet-50，
    在空间维度做双线性池化，降维后分类。
    """
    def __init__(
        self,
        num_classes,
        reduction_dim=512,
        dropout_rate=0.5,
        freeze_stream2=False
    ):
        super().__init__()
        # — Stream1: EfficientNet-B3 —
        self.stream1 = EfficientNet.from_pretrained('efficientnet-b0')
        self.stream1._fc = nn.Identity()

        # — Stream2: ResNet-50 —
        resnet50 = models.resnet50(pretrained=True)
        modules = list(resnet50.children())[:-2]
        self.stream2 = nn.Sequential(*modules)
        if freeze_stream2:
            for p in self.stream2.parameters():
                p.requires_grad = False

        # 通道数
        C1 = self.stream1._bn1.num_features  # 1536
        C2 = 2048                             # ResNet-50 输出通道

        # 降维 1x1 Conv
        self.reduce1 = nn.Conv2d(C1, reduction_dim, kernel_size=1, bias=False)
        self.reduce2 = nn.Conv2d(C2, reduction_dim, kernel_size=1, bias=False)

        # Dropout + 分类头
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(reduction_dim * reduction_dim, num_classes)

    def forward(self, x):
        # 提取特征 maps
        f1 = self.stream1.extract_features(x)  # B x C1 x H1 x W1
        f2 = self.stream2(x)                   # B x C2 x H2 x W2

        # 空间对齐
        if f2.shape[2:] != f1.shape[2:]:
            f2 = F.adaptive_avg_pool2d(f2, f1.shape[2:])

        # 降维
        f1 = self.reduce1(f1)  # B x D x H x W
        f2 = self.reduce2(f2)  # B x D x H x W

        B, D, H, W = f1.shape

        # 双线性池化
        f1 = f1.view(B, D, -1)
        f2 = f2.view(B, D, -1)
        bilinear = torch.bmm(f1, f2.transpose(1, 2)) / (H * W)

        # flatten
        bilinear = bilinear.view(B, D * D)
        # signed-sqrt + L2 归一化
        bilinear = torch.sign(bilinear) * torch.sqrt(torch.abs(bilinear) + 1e-12)
        bilinear = F.normalize(bilinear, dim=1)

        # Dropout + 分类
        out = self.dropout(bilinear)
        return self.classifier(out)
