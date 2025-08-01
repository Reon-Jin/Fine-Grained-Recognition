"""
Stage-2 training (feature-level Co-Teaching)
-------------------------------------------
* 预加载并冻结 EfficientNet-B0 (model1)
* 钩出 backbone.features[-1] 的输出当作 feat_map
* 两个 ROIModel (modelA / modelB) 在 feat_map 上做 Co-Teaching
"""

import os, random, numpy as np, torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    SEED, DATA_DIR, BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS,
    MODEL1_CKPT, EPOCHS, LR, WEIGHT_DECAY,
    NOISE_RATE, FORGET_END, RUNS_DIR
)

from model1 import BaseModel         # model1 = EfficientNet-B0 封装
from model2 import ROIModel                   # 只接受 feat_map
from utils.dataset import get_data_loaders
from utils.noise_handler import NoiseHandler  # 提供 co_teaching_batch
class GCELoss(nn.Module):
    """
    Generalized Cross Entropy (Zhang & Sabuncu, NIPS 2018)
    q ∈ (0,1]，q→0 时退化为交叉熵
    """
    def __init__(self, q: float = 0.7):
        super().__init__()
        assert 0.0 <= q <= 1.0
        self.q = q

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # p(y|x) for correct class
        p = F.softmax(logits, dim=1).clamp_min(1e-8)
        p_y = p[torch.arange(logits.size(0), device=logits.device), target]
        if self.q == 0.0:
            return (-p_y.log()).mean()
        return ((1.0 - p_y.pow(self.q)) / self.q).mean()
# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------------------------------------------------------------- #
# linear forget-rate schedule
# --------------------------------------------------------------------------- #
def get_forget_rate(epoch: int) -> float:
    ramp = min(1.0, epoch / (EPOCHS * FORGET_END))
    return NOISE_RATE * ramp

# --------------------------------------------------------------------------- #
# 训练一个 epoch
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model1, modelA, modelB,
    loader, criterion,
    optimA, optimB,
    device, epoch: int
):
    model1.eval()          # 冻结
    modelA.train(); modelB.train()

    total_loss, total_correct, n_samples = 0., 0, 0
    pbar = tqdm(loader, desc=f"Train {epoch}", ncols=120)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimA.zero_grad(set_to_none=True)
        optimB.zero_grad(set_to_none=True)

        # ---------- 1. 抓取 feat_map ----------
        with torch.no_grad():
            _ = model1(imgs)         # 正向一次以触发钩子
            feat_map = model1._cached_feat      # (B,C,H,W)

        # ---------- 2. 双 ROIModel 前向 ----------
        logitsA = modelA(feat_map.detach())
        logitsB = modelB(feat_map.detach())

        # ---------- 3. Co-Teaching ----------
        fr = get_forget_rate(epoch)
        idxA, idxB = NoiseHandler.co_teaching_batch(
            logitsA.detach(), logitsB.detach(), labels, fr
        )

        lossA = criterion(logitsA[idxB], labels[idxB])
        lossB = criterion(logitsB[idxA], labels[idxA])
        loss  = lossA + lossB
        loss.backward()
        optimA.step(); optimB.step()

        # ---------- 4. 统计 ----------
        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(logitsA.detach(), 1)   # 任选 A 统计
        total_correct += (preds == labels).sum().item()
        n_samples += imgs.size(0)
        pbar.set_postfix({
            "loss": f"{total_loss/n_samples:.4f}",
            "acc":  f"{100.*total_correct/n_samples:.2f}%",
            "fr":   f"{fr:.2f}"
        })

    return total_loss / n_samples, 100.*total_correct / n_samples

# --------------------------------------------------------------------------- #
# 验证
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model1, modelA, modelB, loader, criterion, device):
    model1.eval(); modelA.eval(); modelB.eval()
    total_loss, total_correct, n_samples = 0., 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        _ = model1(imgs)
        feat_map = model1._cached_feat
        logits = (modelA(feat_map) + modelB(feat_map)) / 2.0   # 集成
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(logits, 1)
        total_correct += (preds == labels).sum().item()
        n_samples += imgs.size(0)

    return total_loss / n_samples, 100.*total_correct / n_samples

# --------------------------------------------------------------------------- #
# 注册钩子：抓 EfficientNet-B0 最后一个 features 子模块输出
# --------------------------------------------------------------------------- #
def register_feature_hook(model1):
    """
    给 model1 注册 forward hook，把 backbone.features[-1] 的输出
    存到 model1._cached_feat 里，供后续取用。
    """
    # 适配两种封装：直接有 .features 或者 .backbone.features
    if hasattr(model1, "features"):
        target_layer = model1.features[-1]
    else:
        target_layer = model1.backbone.features[-1]

    def hook(_, __, output):
        model1._cached_feat = output        # (B,C,H,W)
    target_layer.register_forward_hook(hook)

# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据
    train_loader, val_loader, num_classes = get_data_loaders(
        DATA_DIR, BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS
    )
    print(f"Dataset ready → {num_classes} classes")

    # -------- 预加载并冻结 EfficientNet-B0 --------
    model1 = BaseModel().to(device)
    ckpt = torch.load(MODEL1_CKPT, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model1.load_state_dict(state, strict=True)
    for p in model1.parameters():
        p.requires_grad = False
    register_feature_hook(model1)                # 捕获 feat_map
    print("model1 loaded & frozen")

    # -------- 两个 ROIModel (Co-Teaching) --------
    modelA = ROIModel(num_classes, in_channels=1280).to(device)
    modelB = ROIModel(num_classes, in_channels=1280).to(device)

    criterion = GCELoss(q=0.7)
    optimA = optim.AdamW(modelA.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimB = optim.AdamW(modelB.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    schA   = CosineAnnealingLR(optimA, T_max=EPOCHS, eta_min=1e-6)
    schB   = CosineAnnealingLR(optimB, T_max=EPOCHS, eta_min=1e-6)

    best_acc = 0.0
    os.makedirs(RUNS_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model1, modelA, modelB, train_loader,
            criterion, optimA, optimB, device, epoch
        )
        val_loss, val_acc = evaluate(
            model1, modelA, modelB, val_loader, criterion, device
        )

        schA.step(); schB.step()
        print(f"[Val] Epoch {epoch}: loss={val_loss:.4f}  acc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(RUNS_DIR, "best_stage2_feat_coteach.pth")
            torch.save({
                "epoch": epoch,
                "modelA": modelA.state_dict(),
                "modelB": modelB.state_dict(),
                "model1_ckpt": MODEL1_CKPT,
                "acc": best_acc,
            }, save_path)
            print(f"🏅  New best saved → {save_path}")

    print(f"Training finished. Best val acc = {best_acc:.2f}%")

if __name__ == "__main__":
    main()
