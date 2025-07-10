from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from core import resnet
import numpy as np
from core.anchors import generate_default_anchor_maps, hard_nms
from config import CAT_NUM, PROPOSAL_NUM

class ProposalNet(nn.Module):
    """Navigator: 对每个位置打分，输出 (B, #anchors)"""
    def __init__(self):
        super(ProposalNet, self).__init__()
        # 输入通道 2048，对应 ResNet-50 最后一个 conv 层
        self.down1 = nn.Conv2d(2048, 128, kernel_size=3, stride=1, padding=1)
        self.down2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.down3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        # p3/p4/p5 层上分别有 6, 6, 9 个 anchors
        self.tidy1 = nn.Conv2d(128, 6, kernel_size=1)
        self.tidy2 = nn.Conv2d(128, 6, kernel_size=1)
        self.tidy3 = nn.Conv2d(128, 9, kernel_size=1)

    def forward(self, x):
        batch = x.size(0)
        d1 = self.relu(self.down1(x))    # (B,128,H/32,W/32)
        d2 = self.relu(self.down2(d1))   # (B,128,H/64,W/64)
        d3 = self.relu(self.down3(d2))   # (B,128,H/128,W/128)
        t1 = self.tidy1(d1).view(batch, -1)
        t2 = self.tidy2(d2).view(batch, -1)
        t3 = self.tidy3(d3).view(batch, -1)
        return torch.cat((t1, t2, t3), dim=1)  # 总长度 = 6*H1*W1 + 6*H2*W2 + 9*H3*W3

class attention_net(nn.Module):
    """NTS-Net 主体：Backbone + Navigator + Teacher + Scrutinizer"""
    def __init__(self, topN=4):
        super(attention_net, self).__init__()
        # 1) Backbone: ResNet-50
        self.pretrained_model = resnet.resnet50(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512*4, CAT_NUM)

        # 2) Navigator
        self.proposal_net = ProposalNet()
        self.topN = topN

        # 3) Scrutinizer（拼接所有 part 特征 + 全图特征）
        self.concat_net = nn.Linear(2048*(self.topN + 1), CAT_NUM)
        # 4) Teacher（各 part 单独分类）
        self.partcls_net = nn.Linear(512*4, CAT_NUM)

    def forward(self, x):
        batch, _, H, W = x.shape

        # Backbone 提取：
        # raw_logits: (B, CAT_NUM)
        # rpn_feature: conv 最后那层的 feature map (B,2048,H/32,W/32)
        # global_feat: (B,2048)
        raw_logits, rpn_feature, global_feat = self.pretrained_model(x)

        # 动态生成 anchors，基于当前输入分辨率 (H,W)
        # center_anchors: (N,4), edge_anchors: (N,4)
        _, edge_anchors, _ = generate_default_anchor_maps(
            input_shape=(H, W)
        )
        # pad_side = H//2，用于从 padded image 上裁剪出 parts
        pad_side = H // 2

        # 给原图边缘补 0，以便 anchors 带出的 box 不出界
        x_pad = F.pad(x,
                      (pad_side, pad_side, pad_side, pad_side),
                      mode='constant', value=0)

        # Navigator 得分
        rpn_score = self.proposal_net(rpn_feature.detach())  # (B, N_anchors)
        # numpy 处理：拼接 [score, y0, x0, y1, x1, idx]
        all_cdds = []
        for b in range(batch):
            scores = rpn_score[b].detach().cpu().numpy().reshape(-1, 1)  # (N,1)
            boxes  = edge_anchors + pad_side                              # (N,4)
            idxs   = np.arange(len(boxes)).reshape(-1, 1)                # (N,1)
            c      = np.concatenate((scores, boxes.astype(int), idxs), axis=1)  # (N,6)
            all_cdds.append(c)
        # NMS -> topN 候选框
        top_n_cdds = [hard_nms(c, topn=self.topN, iou_thresh=0.25)
                      for c in all_cdds]  # list of (topN,6)
        top_n_cdds = np.stack(top_n_cdds, axis=0)  # (B, topN, 6)

        # 提取索引与得分
        top_n_idx  = torch.from_numpy(top_n_cdds[:,:, -1].astype(int))\
                         .long().to(x.device)           # (B, topN)
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_idx)     # (B, topN)

        # 从 padded 图上裁剪每块 region，resize 到 224×224
        part_imgs = torch.zeros((batch, self.topN, 3, H, W), device=x.device)
        for i in range(batch):
            for j in range(self.topN):
                y0, x0, y1, x1 = top_n_cdds[i, j, 1:5].astype(int)
                if y1 <= y0:
                    y1 = y0 + 1
                if x1 <= x0:
                    x1 = x0 + 1
                crop = x_pad[i:i+1, :, y0:y1, x0:x1]
                part_imgs[i, j] = F.interpolate(
                    crop,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=True,
                )[0]
        part_imgs = part_imgs.view(batch*self.topN, 3, H, W)

        # Teacher 分支：对每块 region 分类
        _, _, part_feat = self.pretrained_model(part_imgs.detach())
        part_feat = part_feat.view(batch, self.topN, -1)  # (B, topN, D)

        # Scrutinizer：拼接所有 part + 全图特征
        flat_parts = part_feat.view(batch, -1)           # (B, topN*D)
        concat_in  = torch.cat([flat_parts, global_feat], dim=1)
        concat_logits = self.concat_net(concat_in)       # (B, CAT_NUM)

        # raw_logits, part_logits 同返回
        part_logits = self.partcls_net(
            part_feat.reshape(-1, part_feat.size(-1))
        ).view(batch, self.topN, -1)

        return raw_logits, concat_logits, part_logits, top_n_idx, top_n_prob

def list_loss(logits, targets):
    """Teacher 分支的分类损失"""
    logp = F.log_softmax(logits, dim=-1)
    return -logp[torch.arange(logits.size(0)), targets]

def ranking_loss(score, targets, proposal_num=PROPOSAL_NUM):
    """Navigator 排序损失 (hinge-style)"""
    batch = score.size(0)
    loss = 0.0
    for i in range(proposal_num):
        pivot = score[:, i].unsqueeze(1)
        mask  = (targets > targets[:, i].unsqueeze(1)).float().to(score.device)
        loss += torch.sum(F.relu(1 - pivot + score) * mask)
    return loss / batch
