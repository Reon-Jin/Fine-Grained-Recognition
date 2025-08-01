# train_model1.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_prepare import prepare_dataloaders
from models.model1 import BaseModel, compute_topk_accuracy
from config import Config

def train():
    train_loader, val_loader, num_classes = prepare_dataloaders()
    print(f"Found {num_classes} classes")

    device     = Config.DEVICE
    model      = BaseModel().to(device)
    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(model.parameters(), lr=Config.LR)

    best_k_acc = 0.0
    for epoch in range(1, Config.EPOCHS + 1):
        # ——— 训练 ———
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [Train]", ncols=100)
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _, _ = model(imgs, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            # 更新当前 batch 的 loss
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(train_loader.dataset)

        # ——— 验证 ———
        model.eval()
        top1_corr = 0
        topk_corr = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [Val]  ", ncols=100)
        with torch.no_grad():
            for imgs, labels in val_bar:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, _ = model(imgs)

                # Top-1
                preds = logits.argmax(dim=1)
                top1_corr += (preds == labels).sum().item()
                # Top-K（按 batch 计算再累加）
                batch_k_acc = compute_topk_accuracy(logits, labels, Config.K)
                topk_corr += batch_k_acc * imgs.size(0)

                # 更新当前 batch 的 Top-1 和 Top-K
                val_bar.set_postfix(acc1=f"{(preds==labels).float().mean().item():.4f}",
                                    acck=f"{batch_k_acc:.4f}")

        val_top1 = top1_corr / len(val_loader.dataset)
        val_topk = topk_corr / len(val_loader.dataset)

        # 打印 epoch 级别汇总
        print(f"Epoch {epoch} summary — "
              f"Train Loss: {avg_loss:.4f} | "
              f"Val Acc@1: {val_top1:.4f} | "
              f"Val Acc@{Config.K}: {val_topk:.4f}")

        # 保存最佳模型
        if val_topk > best_k_acc:
            best_k_acc = val_topk
            torch.save(model.state_dict(), "trained_model/small_1/best_model1.pth")

    print(f"\nTraining complete — best Acc@{Config.K}: {best_k_acc:.4f}")

if __name__ == "__main__":
    train()
