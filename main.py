
"""Stage‑2 Training with Co‑Teaching
-------------------------------------
1. 预加载训练好的 BaseModel (model1) → 提供 attention mask
2. 构建两个 ROIModel (model2_A / model2_B) 做 Co‑Teaching
3. 仅优化 model2_A/B；model1 参数全冻结
"""

import os, random, numpy as np, torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import (
    SEED,
    DATA_DIR,
    BATCH_SIZE,
    IMAGE_SIZE,
    NUM_WORKERS,
    MODEL1_CKPT,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    NOISE_RATE,
    FORGET_END,
    RUNS_DIR,
)

from model1 import BaseModel, GCELoss
from model2 import ROIModel
from utils.dataset import get_data_loaders
from utils.noise_handler import NoiseHandler  # static helper for co‑teaching


# --------------------------------------------------------------------------- #
# Utils
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_forget_rate(epoch: int) -> float:
    """Linear ramp‑up of forget_rate until FORGET_END × EPOCHS"""
    ramp = min(1.0, epoch / (EPOCHS * FORGET_END))
    return NOISE_RATE * ramp


# --------------------------------------------------------------------------- #
# Train & Eval
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model1,
    modelA,
    modelB,
    loader,
    criterion,
    optimA,
    optimB,
    device,
    epoch: int,
):
    model1.eval()  # frozen
    modelA.train()
    modelB.train()

    total_loss, total_correct, total_samples = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Train {epoch}", ncols=120)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimA.zero_grad(set_to_none=True)
        optimB.zero_grad(set_to_none=True)

        # 1. Attention from model1 (no grad)
        with torch.no_grad():
            _, attn = model1(imgs, return_attention=True)

        # 2. Forward through two ROI models
        logitsA = modelA(imgs, attn_mask=attn.detach())
        logitsB = modelB(imgs, attn_mask=attn.detach())

        # 3. Co‑Teaching sample selection
        fr = get_forget_rate(epoch)
        idxA, idxB = NoiseHandler.co_teaching_batch(
            logitsA.detach(), logitsB.detach(), labels, fr
        )

        lossA = criterion(logitsA[idxB], labels[idxB])
        lossB = criterion(logitsB[idxA], labels[idxA])
        loss = lossA + lossB
        loss.backward()
        optimA.step()
        optimB.step()

        # stats
        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(logitsA.detach(), 1)
        total_correct += (preds == labels).sum().item()
        total_samples += imgs.size(0)
        pbar.set_postfix(
            {
                "loss": f"{total_loss/total_samples:.4f}",
                "acc": f"{100*total_correct/total_samples:.2f}%",
                "fr": fr,
            }
        )

    return total_loss / total_samples, 100.0 * total_correct / total_samples


@torch.no_grad()
def evaluate(model1, modelA, modelB, loader, criterion, device):
    model1.eval()
    modelA.eval()
    modelB.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        _, attn = model1(imgs, return_attention=True)
        logitsA = modelA(imgs, attn_mask=attn)
        logitsB = modelB(imgs, attn_mask=attn)

        # 取平均 logits 以提升稳定性
        logits = (logitsA + logitsB) / 2.0
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(logits, 1)
        total_correct += (preds == labels).sum().item()
        total_samples += imgs.size(0)

    return total_loss / total_samples, 100.0 * total_correct / total_samples


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, num_classes = get_data_loaders(
        DATA_DIR, BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS
    )
    print(f"Dataset ready √  classes={num_classes}")

    # Load pretrained model1
    model1 = BaseModel(num_classes).to(device)
    ckpt = torch.load(MODEL1_CKPT, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model1.load_state_dict(state, strict=True)
    for p in model1.parameters():
        p.requires_grad = False
    print(f"Loaded model1 weights from → {MODEL1_CKPT}")

    # Build two ROI models for Co‑Teaching
    modelA = ROIModel(num_classes).to(device)
    modelB = ROIModel(num_classes).to(device)

    criterion = GCELoss(q=0.7)
    optimA = optim.AdamW(modelA.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimB = optim.AdamW(modelB.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    schA = CosineAnnealingLR(optimA, T_max=EPOCHS, eta_min=1e-6)
    schB = CosineAnnealingLR(optimB, T_max=EPOCHS, eta_min=1e-6)

    best_acc = 0.0
    os.makedirs(RUNS_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model1, modelA, modelB, train_loader, criterion, optimA, optimB, device, epoch
        )
        val_loss, val_acc = evaluate(model1, modelA, modelB, val_loader, criterion, device)

        schA.step()
        schB.step()

        print(
            f"[Val] Epoch {epoch}: loss={val_loss:.4f}  acc={val_acc:.2f}%  (best={best_acc:.2f}%)"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            path = os.path.join(RUNS_DIR, "best_stage2_coteach.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "modelA": modelA.state_dict(),
                    "modelB": modelB.state_dict(),
                    "model1_ckpt": MODEL1_CKPT,
                    "acc": best_acc,
                },
                path,
            )
            print(f"🏅  New best saved → {path}")

    print(f"Training completed.  best val acc = {best_acc:.2f}%")


if __name__ == "__main__":
    main()
