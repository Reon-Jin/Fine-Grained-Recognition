import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import torchvision.models as models

class MultiStreamFeatureExtractor(nn.Module):
    """
    精简版多流特征提取网络：
    保留3条流：
      0: EfficientNet-B1（微调最后3个block）
      1: SqueezeNet1_0.features（冻结）
      2: EfficientNet-B0（部分解冻）
    """
    def __init__(
        self,
        num_classes,
        reduction_dim=512,
        dropout_rate=0.5,
        unfreeze_blocks_stream0: int = 3,  # 解冻EfficientNet-B1的最后几个block
        unfreeze_blocks_stream2: int = 3   # 解冻EfficientNet-B0的最后几个block
    ):
        super().__init__()

        # 定义保留的3条流
        self.streams = nn.ModuleList([
            EfficientNet.from_pretrained('efficientnet-b1'),  # stream[0]
            models.squeezenet1_0(pretrained=True).features,   # stream[1]
            EfficientNet.from_pretrained('efficientnet-b0')  # stream[2]
        ])

        # 去掉EfficientNet的分类头
        self.streams[0]._fc = nn.Identity()
        self.streams[2]._fc = nn.Identity()

        # 冻结EfficientNet-B1除最后unfreeze_blocks_stream0个block外的所有参数
        total_blocks_b1 = len(self.streams[0]._blocks)
        # 全部先冻结
        for p in self.streams[0].parameters():
            p.requires_grad = False
        # 然后只解冻最后几个block
        for i in range(total_blocks_b1 - unfreeze_blocks_stream0, total_blocks_b1):
            for p in self.streams[0]._blocks[i].parameters():
                p.requires_grad = True
        # （可根据需要，也可以解冻conv_head和_bn1）

        # 冻结SqueezeNet
        for p in self.streams[1].parameters():
            p.requires_grad = False

        # 解冻EfficientNet-B0最后unfreeze_blocks_stream2个block
        total_blocks_b0 = len(self.streams[2]._blocks)
        for p in self.streams[2].parameters():
            p.requires_grad = False
        for i in range(total_blocks_b0 - unfreeze_blocks_stream2, total_blocks_b0):
            for p in self.streams[2]._blocks[i].parameters():
                p.requires_grad = True

        # 通道数列表
        C = [
            self.streams[0]._bn1.num_features,  # EfficientNet-B1 → 1280
            512,                                # SqueezeNet features → 512
            self.streams[2]._bn1.num_features   # EfficientNet-B0 → 1280
        ]

        # 1x1 Conv降维
        self.reduces = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(C[i], reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ) for i in range(3)
        ])

        # Dropout + 分类器
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
            # 自适应池化
            if f.dim() == 4:
                f = F.adaptive_avg_pool2d(f, (1, 1))
            else:
                raise ValueError(f"Unexpected feature map dims: {f.dim()}")
            f = self.reduces[i](f)
            feats.append(f.view(f.size(0), -1))

        out = torch.cat(feats, dim=1)
        out = self.dropout(out)
        return self.classifier(out)
