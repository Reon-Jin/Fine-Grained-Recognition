# main.py

import os
import torch
import numpy as np
from torch import nn, optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import Config
from data_prepare import prepare_dataloaders
from models.model import FineGrainedModel
from visualize_attention import visualize_blocks  # 4×4 块可视化

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
    model = FineGrainedModel(num_classes).to(cfg.DEVICE)

    criterion      = nn.CrossEntropyLoss()
    optimizer      = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler      = ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=cfg.PATIENCE,
        factor=cfg.LR_DECAY_FACTOR,
        min_lr=cfg.MIN_LR
    )
    aux_weight     = getattr(cfg, 'AUX_WEIGHT', 0.3)
    entropy_weight = getattr(cfg, 'ENTROPY_WEIGHT', 0.01)

    best_acc = 0.0
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    log_path = os.path.join(cfg.MODEL_DIR, 'experiment_log.txt')
    log_config(cfg, log_path)

    for epoch in range(cfg.EPOCHS):
        # —— 训练 —— #
        model.train()
        running_loss = 0.0
        running_acc  = 0
        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{cfg.EPOCHS}", ncols=100)

        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            optimizer.zero_grad()

            # 前向：同时返回主、辅、块权重
            main_logits, aux_logits, block_wts = model(inputs)

            # 主分类损失 + 辅助分类损失
            loss_main = criterion(main_logits, labels)
            loss_aux  = criterion(aux_logits,  labels)
            loss = loss_main + aux_weight * loss_aux

            # 熵正则：鼓励注意力分布不要过度集中
            if entropy_weight > 0:
                ent = - (block_wts * torch.log(block_wts + 1e-8)).sum(dim=1).mean()
                loss = loss + entropy_weight * ent

            loss.backward()
            optimizer.step()

            bs = inputs.size(0)
            running_loss += loss.item() * bs
            running_acc  += (main_logits.argmax(1) == labels).sum().item()
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc':  f'{(main_logits.argmax(1) == labels).float().mean().item():.4f}'
            })

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc  = running_acc  / len(train_loader.dataset)

        # —— 验证 —— #
        model.eval()
        val_loss = 0.0
        val_acc  = 0.0
        val_pbar = tqdm(val_loader, desc=f"Val   Epoch {epoch+1}/{cfg.EPOCHS}", ncols=100)
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                main_logits, _, _ = model(inputs)
                loss = criterion(main_logits, labels)

                bs = inputs.size(0)
                val_loss += loss.item() * bs
                val_acc  += (main_logits.argmax(1) == labels).sum().item()
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc':  f'{(main_logits.argmax(1) == labels).float().mean().item():.4f}'
                })

        val_loss /= len(val_loader.dataset)
        val_acc  /= len(val_loader.dataset)

        # —— 调度器 —— #
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        # —— 日志 —— #
        print(f"\nEpoch {epoch+1}/{cfg.EPOCHS} Summary:")
        print(f"  Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"  Current LR: {current_lr:.6f}")

        with open(log_path, 'a') as f:
            f.write(f"Epoch {epoch+1}/{cfg.EPOCHS}:\n")
            f.write(f"  Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}\n")
            f.write(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}\n")
            f.write(f"  LR: {current_lr:.6f}\n\n")

        # —— 保存最优 —— #
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(cfg.MODEL_DIR, 'best_model.pth'))
            print(f"↳ New best model saved with accuracy: {best_acc:.4f}")

        # —— 可视化 4×4 块注意力 —— #
        if cfg.VISUALIZE_ATTENTION:
            # 优先指定目录，否则随机抽训练集
            if cfg.ATTN_IMAGE_DIR and os.path.isdir(cfg.ATTN_IMAGE_DIR):
                img_paths = [
                    os.path.join(cfg.ATTN_IMAGE_DIR, fn)
                    for fn in sorted(os.listdir(cfg.ATTN_IMAGE_DIR))
                    if fn.lower().endswith(('.jpg','.jpeg','.png'))
                ]
            else:
                subset   = train_loader.dataset
                idxs     = np.random.choice(len(subset), cfg.ATTN_VIS_SAMPLES, replace=False)
                img_paths= [subset.dataset.imgs[subset.indices[i]][0] for i in idxs]

            visualize_blocks(model, img_paths, cfg.DEVICE)

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    train()
