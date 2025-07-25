import os
import sys
import yaml
import json
import torch
import torch.backends.cudnn as cudnn
from multiprocessing import freeze_support
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim import SGD
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from opt_dg_tf2_new import DirectoryDataset
from models import construct_model

def main():
    # ----------- cuDNN Benchmark for fixed-size speedup -----------
    cudnn.benchmark = True

    # ---------------- Load configuration ----------------
    param_dir = "../config.yaml"
    with open(param_dir, 'r') as file:
        param = yaml.load(file, Loader=yaml.FullLoader)
    print('Loading Default parameter configuration:\n', json.dumps(param, sort_keys=True, indent=3))

    # ---------------- Hyperparameters ----------------
    nb_classes   = param['DATA']['nb_classes']
    image_size   = tuple(param['DATA']['image_size'])
    dataset_dir  = param['DATA']['dataset_dir']
    batch_size   = param['MODEL']['batch_size']
    lr           = param['MODEL']['learning_rate']
    model_name   = param['MODEL']['model_name']
    epochs       = param['TRAIN']['epochs']

    # Override from command line
    if len(sys.argv) > 2:
        for i in range(1, len(sys.argv), 2):
            var_name = sys.argv[i]
            new_val  = sys.argv[i + 1]
            try:
                exec(f"{var_name} = {new_val}")
            except Exception:
                exec(f"{var_name} = '{new_val}'")

    # ---------------- Paths ----------------
    train_data_dir = os.path.join(dataset_dir, "train")
    ckpt_dir       = "./best_checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir="./runs")

    # ---------------- Transforms ----------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ---------------- Dataset & Dataloader ----------------
    dataset = DirectoryDataset(
        [train_data_dir],
        augment=True,
        aug_config=param.get('AUGMENTATION', {}),
        preprocess=transform,
        target_size=image_size
    )

    train_len = int(0.8 * len(dataset))
    val_len   = len(dataset) - train_len
    train_ds, val_ds = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    # ---------------- Model setup ----------------
    if model_name == 'cbam_resnet':
        backbone = param['MODEL'].get('backbone', 'resnet50')
        model = construct_model(
            name=model_name,
            backbone=backbone,
            nb_classes=nb_classes
        )
    else:
        model = construct_model(
            name=model_name,
            pool_size=7,
            ROIS_resolution=42,
            ROIS_grid_size=3,
            minSize=2,
            nb_classes=nb_classes,
        )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)
    scaler    = GradScaler()

    # ---------------- Best-model tracking ----------------
    best_val_acc = 0.0

    # ---------------- Training loop ----------------
    for epoch in range(1, epochs+1):
        model.train()
        running_loss    = 0.0
        running_correct = 0
        running_total   = 0

        train_bar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{epochs}")
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(imgs)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = outputs.argmax(dim=1)
            bs    = labels.size(0)
            running_loss    += loss.item() * bs
            running_correct += (preds == labels).sum().item()
            running_total   += bs

            train_bar.set_postfix({
                "Loss":      f"{loss.item():.4f}",
                "Batch Acc": f"{(preds==labels).float().mean().item():.4f}",
                "Epoch Acc": f"{running_correct/running_total:.4f}"
            })

        avg_loss  = running_loss / running_total
        train_acc = running_correct / running_total

        # ---------------- Validation ----------------
        model.eval()
        val_correct = 0
        val_total   = 0

        val_bar = tqdm(val_loader, desc=f"[  Val] Epoch {epoch}/{epochs}")
        with torch.no_grad():
            for imgs, labels in val_bar:
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast():
                    outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
                val_bar.set_postfix({
                    "Epoch Acc": f"{val_correct/val_total:.4f}"
                })

        val_acc = val_correct / val_total if val_total else 0.0
        # -------- TensorBoard logging --------
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)

        # log attention map for first validation batch
        if hasattr(model, 'cbam') and hasattr(model.cbam, 'sa_weight'):
            attn = model.cbam.sa_weight
            if attn is not None:
                grid = make_grid(attn, normalize=True, scale_each=True)
                writer.add_image('Attention/CBAM', grid, epoch)
        elif hasattr(model, 'attention') and hasattr(model.attention, 'attention'):
            attn = model.attention.attention
            if attn is not None:
                attn_img = attn[0].unsqueeze(0).unsqueeze(0)
                grid = make_grid(attn_img, normalize=True, scale_each=True)
                writer.add_image('Attention/Self', grid, epoch)
        print(f"\n✅ Epoch {epoch}/{epochs} | Train Loss: {avg_loss:.4f} "
              f"| Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}\n")

        # -------- Save best model --------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(ckpt_dir, f"best_epoch{epoch:03d}_acc{val_acc:.4f}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"→ New best model saved to {ckpt_path}")

    writer.close()

if __name__ == "__main__":
    freeze_support()
    main()
