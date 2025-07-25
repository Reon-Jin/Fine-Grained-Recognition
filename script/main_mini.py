import os
import sys
import yaml
import json
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from multiprocessing import freeze_support
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torch.optim import SGD
from tqdm import tqdm
from opt_dg_tf2_new import DirectoryDataset
from models import construct_model

def main():
    # 加速卷积
    cudnn.benchmark = True

    # 加载配置
    param_dir = "../config.yaml"
    with open(param_dir, 'r') as f:
        param = yaml.load(f, Loader=yaml.FullLoader)
    print("Loaded config:", json.dumps(param, indent=2))

    # 快速验证用超参
    mini_pct       = 0.5
    default_epochs = param['TRAIN']['epochs']
    mini_epochs    = 100

    # 数据参数
    nb_classes  = param['DATA']['nb_classes']
    dataset_dir = param['DATA']['dataset_dir']
    train_dir   = os.path.join(dataset_dir, "train")

    # 模型参数
    batch_size = param['MODEL']['batch_size']
    lr         = param['MODEL']['learning_rate'] * 10
    model_name = param['MODEL']['model_name']

    # 预处理 Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 增强配置
    aug_config = param.get('AUGMENTATION', {})

    # 构造子集
    full_dataset = DirectoryDataset(
        [train_dir],
        augment=True,
        aug_config=aug_config,
        preprocess=transform,
        target_size=(128, 128)
    )
    subset_len = int(len(full_dataset) * mini_pct)
    dataset    = Subset(full_dataset, list(range(subset_len)))

    # 划分 train/val
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
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # 构建模型并冻结主干
    model = construct_model(
        name=model_name,
        pool_size=7,
        ROIS_resolution=28,
        ROIS_grid_size=2,
        minSize=2,
        nb_classes=nb_classes
    )
    for p in model.base.parameters():
        p.requires_grad = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    # 最佳验证精度追踪
    best_val_acc = 0.0
    ckpt_dir = "./best_checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # 训练循环
    for epoch in range(1, mini_epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc  = 0.0
        total_n    = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{mini_epochs} [Train]")
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            acc   = (preds == labels).float().mean().item()
            bs    = imgs.size(0)
            total_loss += loss.item() * bs
            total_acc  += acc * bs
            total_n    += bs

            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc':  f"{acc:.4f}"
            })

        avg_train_loss = total_loss / total_n
        avg_train_acc  = total_acc  / total_n

        # 验证
        model.eval()
        val_correct = 0
        val_total   = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds   = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
        val_acc = val_correct / val_total if val_total else 0.0

        print(
            f"Epoch {epoch}/{mini_epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
            f"Val Acc: {val_acc:.4f}"
        )

        # 如果验证集精度更好，就保存模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(ckpt_dir, f"best_epoch{epoch:03d}_acc{val_acc:.4f}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"→ New best model saved to {ckpt_path}")

if __name__ == "__main__":
    freeze_support()
    main()
