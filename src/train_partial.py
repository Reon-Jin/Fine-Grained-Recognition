#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PartialLabelDataset
from src.model import PartialResNet
from src.config import (
    BATCH_SIZE, LR, MOMENTUM, WEIGHT_DECAY,
    EPOCHS, TOP_K
)

def partial_loss(logits, candidate_sets):
    """
    部分标签损失：-log( sum_{i in C(x)} p_i )
    logits: [B, N], candidate_sets: list of length B, each is list of label indices
    """
    probs = F.softmax(logits, dim=1)
    loss = 0.0
    for i, cset in enumerate(candidate_sets):
        # cset 已经是 list，直接索引
        loss -= torch.log(probs[i, cset].sum() + 1e-12)
    return loss / logits.size(0)

def collate_fn(batch):
    """
    自定义 collate，将 images 堆成 tensor，idxs 堆成 tensor，
    candidate_sets 保留为 list of lists
    """
    imgs, idxs, csets = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    idxs = torch.tensor(idxs, dtype=torch.long)
    return imgs, idxs, list(csets)

def main():
    parser = argparse.ArgumentParser(
        description="融合 Top-K 更新的部分标签学习训练脚本"
    )
    parser.add_argument(
        "--train_txt",
        default="../data/WebFG-400/clean_train.txt",
        help="clean_train.txt 路径"
    )
    parser.add_argument(
        "--model_out",
        default="../model/partial_model.pth",
        help="训练好的部分标签模型保存路径"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 构建模型并加载 ImageNet 预训练权重
    model = PartialResNet().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # 2. 加载部分标签数据集
    train_ds = PartialLabelDataset(args.train_txt)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 3. 训练循环
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(
            train_loader,
            desc=f"[Epoch {epoch}/{EPOCHS}] Training",
            ncols=100,
            leave=False
        )
        for x, idxs, csets in train_bar:
            x = x.to(device)
            logits = model(x)
            loss = partial_loss(logits, csets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # 4. 每轮结束后，用 Top-K **替换**候选集 C(x)
        model.eval()
        eval_loader = DataLoader(
            train_ds,
            batch_size=64,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=collate_fn
        )
        with torch.no_grad():
            for x, idxs, _ in tqdm(
                eval_loader,
                desc=f"[Epoch {epoch}/{EPOCHS}] Updating C(x)",
                ncols=100,
                leave=False
            ):
                x = x.to(device)
                # topk 返回 values, indices
                _, topk_idxs = model(x).topk(TOP_K, dim=1)
                topk_idxs = topk_idxs.cpu().tolist()
                for i, sid in enumerate(idxs):
                    train_ds.C[sid] = topk_idxs[i]  # 直接替换为新的 Top-K 列表

        avg_loss = running_loss / len(train_ds)
        print(f"Epoch {epoch}/{EPOCHS} — avg loss: {avg_loss:.4f}")

    # 5. 保存部分标签模型
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    torch.save(model.state_dict(), args.model_out)
    print(f"▶ Partial model saved to {args.model_out}")

if __name__ == "__main__":
    main()
