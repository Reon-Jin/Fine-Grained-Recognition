import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import torchvision.models as models

class MultiStreamFeatureExtractor(nn.Module):
    """
    精简版多流特征提取网络：
    保留3条流：
      0: EfficientNet-B1（仅解冻最后3个block）
      1: SqueezeNet1_0.features（冻结）
      2: EfficientNet-B0（解冻最后unfreeze_blocks_stream4个block）
    """
    def __init__(
        self,
        num_classes,
        reduction_dim=512,
        dropout_rate=0.5,
        unfreeze_blocks_stream4: int = 3  # 解冻EfficientNet-B0的最后几个block
    ):
        super().__init__()

        # 定义保留的3条流
        self.streams = nn.ModuleList([
            EfficientNet.from_pretrained('efficientnet-b1'),  # stream[0]
            models.squeezenet1_0(pretrained=True).features,    # stream[1]
            EfficientNet.from_pretrained('efficientnet-b0')   # stream[2]
        ])

        # 移除分类头
        self.streams[0]._fc = nn.Identity()
        self.streams[2]._fc = nn.Identity()

        # ========================
        # 1. EfficientNet-B1 (stream0) 只解冻最后3个block
        # ========================
        # 先冻结所有参数
        for p in self.streams[0].parameters():
            p.requires_grad = False
        # 解冻最后3个block
        total_blocks0 = len(self.streams[0]._blocks)
        for i in range(total_blocks0 - 3, total_blocks0):
            for p in self.streams[0]._blocks[i].parameters():
                p.requires_grad = True

        # ========================
        # 2. SqueezeNet (stream1) 完全冻结
        # ========================
        for p in self.streams[1].parameters():
            p.requires_grad = False

        # ========================
        # 3. EfficientNet-B0 (stream2) 解冻最后unfreeze_blocks_stream4个block
        # ========================
        # 默认EfficientNet-B0 全部可训练，先冻结所有
        for p in self.streams[2].parameters():
            p.requires_grad = False
        total_blocks2 = len(self.streams[2]._blocks)
        for i in range(total_blocks2 - unfreeze_blocks_stream4, total_blocks2):
            for p in self.streams[2]._blocks[i].parameters():
                p.requires_grad = True

        # 通道数列表
        C = [
            self.streams[0]._bn1.num_features,  # EfficientNet-B1 → 1280
            512,                               # SqueezeNet features → 512
            self.streams[2]._bn1.num_features  # EfficientNet-B0 → 1280
        ]

        # 降维 1x1 Conv
        self.reduces = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(C[i], reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ) for i in range(3)
        ])

        # Dropout + 分类层
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(reduction_dim * 3, num_classes)

    def forward(self, x):
        feats = []
        for i, stream in enumerate(self.streams):
            # EfficientNet 使用 extract_features
            if isinstance(stream, EfficientNet):
                f = stream.extract_features(x)
            else:
                f = stream(x)
            # 自适应池化到 1x1
            if f.dim() == 4:
                f = F.adaptive_avg_pool2d(f, (1, 1))
            else:
                raise ValueError(f"Unexpected feature map dims: {f.dim()}")
            f = self.reduces[i](f)
            feats.append(f.view(f.size(0), -1))

        out = torch.cat(feats, dim=1)
        out = self.dropout(out)
        return self.classifier(out)
