"""Stage‑2 training (feature‑level Co‑Teaching) ‑‑ SVM version
--------------------------------------------------------------
* Freeze EfficientNet‑B0 (model1) and cache its last feature map
* Two CoTeachHead (modelA / modelB) output linear‑SVM margins
* Loss uses multi‑class hinge (nn.MultiMarginLoss, reduction='none')
"""

import os, random, numpy as np, torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from config import (
    SEED, DATA_DIR, BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS,
    MODEL1_CKPT, EPOCHS, LR, WEIGHT_DECAY,
    NOISE_RATE, FORGET_END, RUNS_DIR
)

from model1 import BaseModel                 # EfficientNet‑B0 wrapper
from model2 import CoTeachHead           # ← new head: feat extractor + linear SVM
from utils.dataset import get_data_loaders
from utils.noise_handler import NoiseHandler  # provides co_teaching_batch

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
# linear forget‑rate schedule
# --------------------------------------------------------------------------- #
def get_forget_rate(epoch: int) -> float:
    ramp = min(1.0, epoch / (EPOCHS * FORGET_END))
    return NOISE_RATE * ramp

# --------------------------------------------------------------------------- #
# train one epoch
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model1, modelA, modelB,
    loader,
    optimA, optimB,
    device, epoch: int,
    criterion
):
    model1.eval()          # frozen
    modelA.train(); modelB.train()

    total_loss, total_correct, n_samples = 0., 0, 0
    pbar = tqdm(loader, desc=f"Train {epoch}", ncols=120)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimA.zero_grad(set_to_none=True)
        optimB.zero_grad(set_to_none=True)

        # ---------- 1. extract feat_map ----------
        with torch.no_grad():
            _ = model1(imgs)           # trigger hook
            feat_map = model1._cached_feat

        # ---------- 2. forward through two heads ----------
        logitsA, _ = modelA(feat_map.detach())
        logitsB, _ = modelB(feat_map.detach())

        # ---------- 3. Co‑Teaching ----------
        fr = get_forget_rate(epoch)
        idxA, idxB = NoiseHandler.co_teaching_batch(
            logitsA.detach(), logitsB.detach(), labels, fr
        )

        lossA_vec = criterion(logitsA[idxB], labels[idxB])        # [k]
        lossB_vec = criterion(logitsB[idxA], labels[idxA])
        loss = lossA_vec.mean() + lossB_vec.mean()
        loss.backward()
        optimA.step(); optimB.step()

        # ---------- 4. metrics ----------
        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(logitsA.detach(), 1)   # use head A for accuracy
        total_correct += (preds == labels).sum().item()
        n_samples += imgs.size(0)
        pbar.set_postfix({
            "loss": f"{total_loss/n_samples:.4f}",
            "acc":  f"{100.*total_correct/n_samples:.2f}%",
            "fr":   f"{fr:.2f}"
        })

    return total_loss / n_samples, 100.*total_correct / n_samples

# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model1, modelA, modelB, loader, device, criterion):
    model1.eval(); modelA.eval(); modelB.eval()
    total_loss, total_correct, n_samples = 0., 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        _ = model1(imgs)
        feat_map = model1._cached_feat
        logitsA, _ = modelA(feat_map)
        logitsB, _ = modelB(feat_map)
        logits = (logitsA + logitsB) / 2.0
        loss = criterion(logits, labels)   # reduction='none' → [B]
        total_loss += loss.mean().item() * imgs.size(0)
        _, preds = torch.max(logits, 1)
        total_correct += (preds == labels).sum().item()
        n_samples += imgs.size(0)

    return total_loss / n_samples, 100.*total_correct / n_samples

# --------------------------------------------------------------------------- #
# register hook to grab EfficientNet‑B0 last feature
# --------------------------------------------------------------------------- #
def register_feature_hook(model1):
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

    # data
    train_loader, val_loader, num_classes = get_data_loaders(
        DATA_DIR, BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS
    )
    print(f"Dataset ready → {num_classes} classes")

    # -------- backbone --------
    model1 = BaseModel().to(device)
    ckpt = torch.load(MODEL1_CKPT, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model1.load_state_dict(state, strict=True)
    for p in model1.parameters():
        p.requires_grad = False
    register_feature_hook(model1)
    print("model1 loaded & frozen")

    # -------- Co‑Teaching heads --------
    modelA = CoTeachHead(num_classes, in_channels=1280).to(device)
    modelB = CoTeachHead(num_classes, in_channels=1280).to(device)

    # hinge loss (multi‑class SVM)
    criterion = nn.MultiMarginLoss(p=1, margin=1.0, reduction="none")
    optimA = optim.AdamW(modelA.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimB = optim.AdamW(modelB.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    schA   = CosineAnnealingLR(optimA, T_max=EPOCHS, eta_min=1e-6)
    schB   = CosineAnnealingLR(optimB, T_max=EPOCHS, eta_min=1e-6)

    best_acc = 0.0
    os.makedirs(RUNS_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model1, modelA, modelB, train_loader,
            optimA, optimB, device, epoch, criterion
        )
        val_loss, val_acc = evaluate(
            model1, modelA, modelB, val_loader, device, criterion
        )

        schA.step(); schB.step()
        print(f"[Val] Epoch {epoch}: loss={val_loss:.4f}  acc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(RUNS_DIR, "best_stage2_feat_coteach_svm.pth")
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
