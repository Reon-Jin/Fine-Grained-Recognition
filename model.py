# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
from config import PROJ_DIM

class MultiStreamFeatureExtractor(nn.Module):
    """
    3 流特征提取 + SNSCL 模块：
      0: EfficientNet-B1 (只解冻最后3个block)
      1: SqueezeNet1_0.features (完全冻结)
      2: EfficientNet-B0 (解冻末尾 unfreeze_blocks_stream4 blocks)
    """
    def __init__(self, num_classes, reduction_dim=512, dropout_rate=0.5, unfreeze_blocks_stream4=3):
        super().__init__()

        # --- Streams 初始化 ---
        self.streams = nn.ModuleList([
            EfficientNet.from_pretrained('efficientnet-b1'),
            models.squeezenet1_0(pretrained=True).features,
            EfficientNet.from_pretrained('efficientnet-b0')
        ])
        # 移除原 fc
        self.streams[0]._fc = nn.Identity()
        self.streams[2]._fc = nn.Identity()

        # --- 冻结 / 解冻 ---
        # stream0: 先全冻结，再解冻最后3个block
        for p in self.streams[0].parameters(): p.requires_grad=False
        total0 = len(self.streams[0]._blocks)
        for i in range(total0-3, total0):
            for p in self.streams[0]._blocks[i].parameters(): p.requires_grad=True

        # stream1: 完全冻结
        for p in self.streams[1].parameters(): p.requires_grad=False

        # stream2: 先全冻结，再解冻最后 unfreeze_blocks_stream4 个 block
        for p in self.streams[2].parameters(): p.requires_grad=False
        total2 = len(self.streams[2]._blocks)
        for i in range(total2-unfreeze_blocks_stream4, total2):
            for p in self.streams[2]._blocks[i].parameters(): p.requires_grad=True

        # --- 降维 & 分类头 ---
        C0 = self.streams[0]._bn1.num_features
        C1 = 512
        C2 = self.streams[2]._bn1.num_features
        self.reduction_dim = reduction_dim
        self.reduces = nn.ModuleList([
            nn.Sequential(nn.Conv2d(C0,reduction_dim,1,bias=False), nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(C1,reduction_dim,1,bias=False), nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(C2,reduction_dim,1,bias=False), nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True)),
        ])
        self.dropout    = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(reduction_dim*3, num_classes)

        # --- SNSCL 模块 ---
        self._build_snscl_modules(PROJ_DIM)

    def _build_snscl_modules(self, proj_dim):
        self.proj_head = nn.Sequential(
            nn.Linear(self.reduction_dim*3, self.reduction_dim*3),
            nn.ReLU(inplace=True),
            nn.Linear(self.reduction_dim*3, proj_dim)
        )
        self.stoch_mlp = nn.Sequential(
            nn.Linear(self.reduction_dim*3, self.reduction_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.reduction_dim, self.reduction_dim*2)
        )

    def forward(self, x):
        feats=[]
        for i,stream in enumerate(self.streams):
            f = stream.extract_features(x) if isinstance(stream,EfficientNet) else stream(x)
            f = F.adaptive_avg_pool2d(f,(1,1))
            feats.append(self.reduces[i](f).view(f.size(0),-1))
        out = self.dropout(torch.cat(feats,1))
        return self.classifier(out)

    def forward_contrastive(self, x):
        """返回 z, z_proj, z_stoch, mu, logvar"""
        feats=[]
        for i,stream in enumerate(self.streams):
            f = stream.extract_features(x) if isinstance(stream,EfficientNet) else stream(x)
            f = F.adaptive_avg_pool2d(f,(1,1))
            feats.append(self.reduces[i](f).view(f.size(0),-1))
        z = torch.cat(feats,1)
        z_proj = self.proj_head(z)
        stats  = self.stoch_mlp(z)
        D      = stats.size(1)//2
        mu, logvar = stats[:,:D], stats[:,D:]
        std    = (0.5*logvar).exp()
        eps    = torch.randn_like(std)
        z_stoch= mu + eps*std
        return z, z_proj, z_stoch, mu, logvar
