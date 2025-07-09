# train.py

import os
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import ImageFile
import numpy as np
from sklearn.mixture import GaussianMixture

from model import MultiStreamFeatureExtractor  # 新模型结构（只保留 3 个流）
from config import (
    get_train_dataset,
    get_val_dataset,
    TEMPERATURE,
    LAMBDA_NTC,
    LAMBDA_KL,
    WEIGHT_THRESHOLD
)

# 忽略因损坏图像导致的 PIL 警告
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images"
)


class NTCLoss(nn.Module):
    """Noise-Tolerated Supervised Contrastive Loss"""
    def __init__(self, temperature=TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_proj, labels, omega):
        B = z_proj.size(0)
        # 归一化
        z = F.normalize(z_proj, dim=1)
        # 相似度矩阵
        sim = (z @ z.T) / self.temperature
        # 同类 mask
        mask = labels.unsqueeze(1).eq(labels.unsqueeze(0)).float()
        # 排除自身
        exp_sim = torch.exp(sim) * (1 - torch.eye(B, device=z.device))
        # log-prob
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        # 对每对(i,j)取 min(ω_i, ω_j)
        w = torch.min(omega.unsqueeze(1), omega.unsqueeze(0))
        # 同类对平均 log-prob
        denom = mask.sum(dim=1) - 1 + 1e-8
        mean_log_pos = (w * mask * log_prob).sum(dim=1) / denom
        return - mean_log_pos.mean()


class StochasticModuleKL(nn.Module):
    """KL(N(mu,σ)||N(0,I)) 正则项"""
    def forward(self, mu, logvar):
        # KL = -0.5 * E[1 + logvar - mu^2 - exp(logvar)]
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    ce_base = nn.CrossEntropyLoss(reduction='none')
    ntc_fn = NTCLoss()
    kl_fn = StochasticModuleKL()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Train Epoch {epoch+1}", ncols=100, colour="white")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # —— 前向分类 & small-loss 选样 ——
        outputs = model(inputs)
        ce_losses = ce_base(outputs, labels)  # (B,)
        with torch.no_grad():
            arr = ce_losses.cpu().numpy().reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, max_iter=100).fit(arr)
            clean_comp = np.argmin(gmm.means_)
            probs = gmm.predict_proba(arr)[:, clean_comp]
            omega = torch.from_numpy(probs).to(device)
            omega = torch.where(omega > WEIGHT_THRESHOLD, torch.ones_like(omega), omega)

        # —— 软标签分类损失 ——
        prob = F.softmax(outputs, dim=1)
        num_classes = outputs.size(1)
        y_one = F.one_hot(labels, num_classes).float()
        y_soft = omega.unsqueeze(1) * y_one + (1 - omega).unsqueeze(1) * prob
        loss_cls = -(y_soft * torch.log(prob.clamp(min=1e-8))).sum(dim=1).mean()

        # —— SNSCL 对比 & 随机模块 KL ——
        z, z_proj, z_stoch, mu, logvar = model.forward_contrastive(inputs)
        loss_ntc = ntc_fn(z_proj, labels, omega)
        loss_kl = kl_fn(mu, logvar)

        # —— 总损失 & 更新 ——
        loss = loss_cls + LAMBDA_NTC * loss_ntc + LAMBDA_KL * loss_kl
        loss.backward()
        optimizer.step()

        # —— 累计统计 & 进度条更新 ——
        running_loss += loss_cls.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        current_loss = running_loss / total
        current_acc = correct / total
        progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

    print(f"[Train]  Loss: {current_loss:.4f}  Acc: {current_acc:.4f}")
    return current_loss, current_acc


def validate_epoch(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Validate", ncols=100, colour="white")
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            current_loss = running_loss / total
            current_acc = correct / total
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

    print(f"[Valid]  Loss: {current_loss:.4f}  Acc: {current_acc:.4f}\n")
    return current_loss, current_acc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = get_train_dataset()
    val_dataset = get_val_dataset()
    num_classes = len(train_dataset.classes)

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = MultiStreamFeatureExtractor(
        num_classes=num_classes,
        reduction_dim=512,
        dropout_rate=0.5,
        unfreeze_blocks_stream4=3
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=2
    )

    criterion = nn.CrossEntropyLoss()
    os.makedirs("model", exist_ok=True)
    best_acc = 0.0
    num_epochs = 50

    for epoch in range(num_epochs):
        print(f"=== Epoch {epoch+1}/{num_epochs} ===")
        train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        # 保持原调度调用方式不变
        scheduler.step(epoch + epoch/len(train_loader))
        _, val_acc = validate_epoch(model, val_loader, criterion, device, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "model/model.pth")
        torch.save(model.state_dict(), "model/model_latest.pth")

    print("Training complete. Best Val Acc: {:.4f}".format(best_acc))
