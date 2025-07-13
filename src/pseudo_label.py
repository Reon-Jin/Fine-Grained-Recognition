#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.model import ResNet50Feat  # 只提取特征
from src.dataset import PartialLabelDataset
from src.config import BATCH_SIZE, NUM_CLASSES, TOP_K

class FeatureDataset(Dataset):
    """
    读取 clean_train.txt，通过预训练 ResNet50 提取 2048-d 特征，并带上候选集 C(x)
    """
    def __init__(self, list_file):
        ds = PartialLabelDataset(list_file)
        self.paths = ds.paths
        self.C      = ds.C
        self.transform = ds.transform
        # 用于一次性加载所有特征——后面我们会填充 self.features
        self.features = [None] * len(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # 返回占位，真正特征会在 extract_features 阶段填入
        return idx, self.C[idx]

def extract_features(model_feat, loader, device):
    """
    将所有图像的特征预先提取并保存到 dataset.features
    """
    dataset = loader.dataset
    with torch.no_grad():
        for x, idxs, _ in tqdm(loader, desc="Extracting feats", ncols=100):
            x = x.to(device)
            f = model_feat(x).cpu()
            for i, idx in enumerate(idxs.tolist()):
                dataset.features[idx] = f[i]

def train_ecoc_classifiers(dataset, M, device, epochs=5, lr=1e-2):
    """
    训练 L 个二分类器 g_t: simple linear layer + BCE loss
    只选用那些候选集 C(x) 完全属于正类或负类的样本去训练
    返回 list of nn.Module
    """
    L = M.shape[1]
    classifiers = []
    for t in range(L):
        net = nn.Linear(dataset.features[0].shape[0], 1).to(device)
        opt = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        crit = nn.BCEWithLogitsLoss()
        # 收集训练样本索引
        pos_idx, neg_idx = [], []
        for i, C in enumerate(dataset.C):
            # 如果 C(x) 中所有类在 M[:,t] 上都 = +1，则它是正样本
            bits = M[list(C), t]
            if np.all(bits == 1):
                pos_idx.append(i)
            elif np.all(bits == -1):
                neg_idx.append(i)
        train_idx = pos_idx + neg_idx
        if not train_idx:
            classifiers.append(net)  # 无样本，跳过训练
            continue
        labels = torch.tensor(
            [1.0]*len(pos_idx) + [0.0]*len(neg_idx),
            dtype=torch.float32, device=device
        )
        # 构建 DataLoader
        feats = torch.stack([dataset.features[i] for i in train_idx]).to(device)
        loader = DataLoader(
            list(zip(feats, labels)),
            batch_size=64, shuffle=True
        )
        # 训练
        for _ in range(epochs):
            net.train()
            for x, y in loader:
                opt.zero_grad()
                out = net(x).squeeze(1)
                loss = crit(out, y)
                loss.backward()
                opt.step()
        classifiers.append(net)
    return classifiers

def ecoc_decode(feature, classifiers, M, device):
    """
    对单样本特征 feature 解码：
    1) 通过 L 个分类器得到 bit_t = sign(sigmoid(net_t(f))-0.5)
    2) 计算与每个类编码 M[j] 的汉明距离（或指数距离）
    3) 返回最小距离的类索引
    """
    logits = torch.stack([net(feature.to(device)).squeeze()
                          for net in classifiers])  # [L]
    bits = (torch.sigmoid(logits) > 0.5).int() * 2 - 1  # {+1,-1}
    bits = bits.cpu().numpy()  # [L]
    # 汉明距离：count of mismatches between bits and each M[j]
    # M 是 np.array shape (NUM_CLASSES, L)
    mismatches = (M != bits).astype(int)  # [NUM_CLASSES, L]
    dists = mismatches.sum(axis=1)       # [NUM_CLASSES]
    return int(np.argmin(dists))

def main():
    parser = argparse.ArgumentParser(
        description="ECOC 伪标签生成"
    )
    parser.add_argument(
        "--clean_txt",
        default="../data/WebFG-400/clean_train.txt",
        help="clean_train.txt"
    )
    parser.add_argument(
        "--model_out",
        default="pseudo_ecoc.txt",
        help="输出伪标签文件"
    )
    parser.add_argument(
        "--code_len", type=int, default=128,
        help="编码长度 L（论文选 128）"
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="每个二分类器训练轮数"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 提取所有训练样本特征
    feat_model = torch.nn.DataParallel(ResNet50Feat().to(device))
    ds = PartialLabelDataset(args.clean_txt)
    # 用 ImageList 加载索引和标签，但我们只要 index & C(x)
    from src.dataset import ImageList
    li = ImageList(args.clean_txt, train=False)
    loader = DataLoader(
        li, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True
    )
    # 转换为 FeatureDataset
    fds = FeatureDataset(args.clean_txt)
    # 用属性传递
    fds.paths = li.samples  # 但我们只用 idx
    fds.C     = ds.C
    fds.transform = li.transform
    floader = DataLoader(
        fds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=8, pin_memory=True
    )
    extract_features(feat_model, floader, device)

    # 2) 随机生成 ECOC 编码矩阵 M ∈ {+1,-1}^{NUM_CLASSES×L}
    L = args.code_len
    rng = np.random.RandomState(0)
    M = rng.choice([-1, 1], size=(NUM_CLASSES, L))

    # 3) 训练 L 个二分类器
    classifiers = train_ecoc_classifiers(fds, M, device,
                                         epochs=args.epochs)

    # 4) 解码所有样本
    with open(args.model_out, 'w') as fout:
        for idx in tqdm(range(len(fds)), desc="Decoding ECOC", ncols=100):
            feat = fds.features[idx]
            pred = ecoc_decode(feat, classifiers, M, device)
            path = fds.paths[idx][0]  # ImageList 存 samples 为 (path,label)
            fout.write(f"{path} {pred}\n")

    print(f"✅ ECOC 伪标签已写入 {args.model_out}")

if __name__ == "__main__":
    main()
