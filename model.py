import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import torchvision.models as models

class MultiStreamFeatureExtractor(nn.Module):
    """
    多流特征提取网络：包含5条流，后4条冻结，仅做特征提取。
    流配置（均基于 ImageNet-1K 预训练）：
      0: EfficientNet-B1 (微调)
      1: MobileNetV2.features (冻结)
      2: ShuffleNetV2_x1_0 (冻结)
      3: SqueezeNet1_0.features (冻结)
      4: EfficientNet-B0 (冻结)
    """
    def __init__(
        self,
        num_classes,
        reduction_dim=512,
        dropout_rate=0.5,
        freeze_streams:list = [1,2,3,4]
    ):
        super().__init__()
        # 定义5条流
        self.streams = nn.ModuleList([
            EfficientNet.from_pretrained('efficientnet-b1'),                # 1280 ch
            models.mobilenet_v2(pretrained=True).features,                    # 1280 ch
            nn.Sequential(*list(models.shufflenet_v2_x1_0(pretrained=True).children())[:-2]),  # stage4 output 464 ch
            models.squeezenet1_0(pretrained=True).features,                  # 512 ch
            EfficientNet.from_pretrained('efficientnet-b0')                 # 1280 ch
        ])
        # 去掉 EfficientNet 分类头
        self.streams[0]._fc = nn.Identity()
        self.streams[4]._fc = nn.Identity()

        # 冻结指定流
        for idx in freeze_streams:
            for p in self.streams[idx].parameters():
                p.requires_grad = False

        # 各流输出通道
        C = [
            self.streams[0]._bn1.num_features,  # EfficientNet-B1 ->1280
            1280,                              # MobileNetV2.features ->1280
            464,                               # ShuffleNetV2_x1_0 stage4 ->464
            512,                               # SqueezeNet1_0.features ->512
            self.streams[4]._bn1.num_features  # EfficientNet-B0 ->1280
        ]

        # 1x1 Conv 降维
        self.reduces = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(C[i], reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ) for i in range(5)
        ])

        # Dropout + 分类头
        self.dropout = nn.Dropout(dropout_rate)
        # 特征拼接后维度 = reduction_dim * 5
        self.classifier = nn.Linear(reduction_dim * 5, num_classes)

    def forward(self, x):
        feats = []
        for i, stream in enumerate(self.streams):
            # EfficientNet 用 extract_features
            if isinstance(stream, EfficientNet):
                f = stream.extract_features(x)
            else:
                f = stream(x)
            # 自适应池化到 1x1
            if f.dim() == 4:
                f = F.adaptive_avg_pool2d(f, (1, 1))
            else:
                raise ValueError(f"Unexpected feature map dims: {f.dim()}")
            # 降维
            f = self.reduces[i](f)
            feats.append(f.view(f.size(0), -1))

        # 拼接所有流特征
        out = torch.cat(feats, dim=1)
        out = self.dropout(out)
        return self.classifier(out)
