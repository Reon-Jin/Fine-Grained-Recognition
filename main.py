# main.py

import os
# 禁用 symlink 警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# 让 transformers 只报 ERROR 级别
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# 直接关闭 HuggingFace Hub 的下载进度条
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import torch
import numpy as np
from torch import nn, optim
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from config import Config
from data_prepare import prepare_dataloaders
from models.model2 import TransFGDeiT
from visualize_attention import visualize_attention_patches  # 新可视化函数

def log_config(cfg: Config, log_path: str):
    with open(log_path, 'a') as f:
        f.write("======= Experiment Configuration =======\n")
        for attr in dir(cfg):
            if not attr.startswith("__") and not callable(getattr(cfg, attr)):
                f.write(f"{attr}: {getattr(cfg, attr)}\n")
        f.write("========================================\n\n")

def train():
    cfg = Config()
    train_loader, val_loader, num_classes = prepare_dataloaders(cfg)
    model = TransFGDeiT(num_classes).to(cfg.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.T_EPOCHS,  # 第一次重启的周期长度（单位：epoch），例如 10
        T_mult=2,  # 每次重启后周期乘以 2：10→20→40…
        eta_min=cfg.MIN_LR  # 学习率衰减到的下限
    )

    best_acc = 0.0
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    log_path = os.path.join(cfg.MODEL_DIR, 'experiment_log.txt')
    log_config(cfg, log_path)

    for epoch in range(cfg.EPOCHS):
        # ——— 训练 ———
        model.train()
        running_loss = 0.0
        running_acc  = 0
        for inputs, labels in tqdm(train_loader, desc=f"Train {epoch+1}/{cfg.EPOCHS}", ncols=100):
            inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            optimizer.zero_grad()

            logits, aux = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_acc  += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc  = running_acc  / len(train_loader.dataset)

        # ——— 验证 ———
        model.eval()
        val_loss = 0.0
        val_acc  = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Val   {epoch+1}/{cfg.EPOCHS}", ncols=100):
                inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                logits, aux = model(inputs)
                loss = criterion(logits, labels)

                preds = logits.argmax(dim=1)
                val_loss += loss.item() * inputs.size(0)
                val_acc  += (preds == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc  /= len(val_loader.dataset)

        # ——— 学习率衰减 ———
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # ——— 日志 ———
        print(f"\nEpoch {epoch+1}/{cfg.EPOCHS} Summary:")
        print(f"  Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"  LR: {current_lr:.6f}")

        with open(log_path, 'a') as f:
            f.write(f"Epoch {epoch+1}:\n")
            f.write(f"  Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}\n")
            f.write(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}\n")
            f.write(f"  LR: {current_lr:.6f}\n\n")

        # ——— 保存最优 ———
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(cfg.MODEL_DIR, 'best_model.pth'))
            print(f"↳ New best model saved: {best_acc:.4f}")

        # ——— 可视化注意力 ———
        if cfg.VISUALIZE_ATTENTION:
            img_dir = cfg.ATTN_IMAGE_DIR
            if os.path.isdir(img_dir):
                img_paths = [
                    os.path.join(img_dir, fn)
                    for fn in sorted(os.listdir(img_dir))
                    if fn.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                visualize_attention_patches(model, img_paths, cfg.DEVICE)
            else:
                print(f"ATTN_IMAGE_DIR 不存在或不是有效目录：{img_dir}")

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    train()
