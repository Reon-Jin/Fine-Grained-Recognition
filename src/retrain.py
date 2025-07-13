#!/usr/bin/env python3
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image, ImageFile
from tqdm import tqdm

# 允许加载截断图像
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SupervisedDataset(Dataset):
    """
    读取 pseudo_train.txt 或 val.txt，返回 (img_tensor, label)
    """
    def __init__(self, list_file, train=True):
        lines = open(list_file).read().splitlines()
        self.samples = [(l.split()[0], int(l.split()[1])) for l in lines]
        # 训练用增强，验证/测试用中心裁剪
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],
                                     [0.229,0.224,0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],
                                     [0.229,0.224,0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except OSError:
            img = Image.new('RGB', (224,224), (0,0,0))
        return self.transform(img), label

def main():
    parser = argparse.ArgumentParser(
        description="最终再训练：标准交叉熵 on 伪标签"
    )
    parser.add_argument("--pseudo_txt",
                        default="pseudo_train.txt",
                        help="伪标签文件")
    parser.add_argument("--val_txt",
                        default="../data/WebFG-400/val.txt",
                        help="验证集列表")
    parser.add_argument("--out_model",
                        default="final_model.pth",
                        help="最终模型输出")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs",    type=int, default=110)
    parser.add_argument("--lr",        type=float, default=5e-3)
    parser.add_argument("--momentum",  type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=2e-5)
    parser.add_argument("--milestones", type=int, nargs="+",
                        default=[80,100],
                        help="学习率衰减里程碑")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集 & DataLoader
    train_ds = SupervisedDataset(args.pseudo_txt, train=True)
    val_ds   = SupervisedDataset(args.val_txt, train=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True
    )

    # 模型：ResNet-50 + 新 FC
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, train_ds.samples[0][1]+1)  # ensure correct num_classes
    # 或者硬编码 NUM_CLASSES=400： model.fc = nn.Linear(in_feats, 400)
    model = model.to(device)

    # 优化器 & scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=0.1
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        # ——— 训练 ———
        model.train()
        train_bar = tqdm(train_loader,
                         desc=f"[Epoch {epoch}/{args.epochs}] Train",
                         ncols=100, leave=False)
        running_loss = 0.0
        for x, y in train_bar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()*x.size(0)
            train_bar.set_postfix(loss=loss.item())

        scheduler.step()
        avg_loss = running_loss / len(train_ds)

        # ——— 验证 ———
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds==y).sum().item()
                total   += y.size(0)
        acc = correct/total*100

        print(f"Epoch {epoch}/{args.epochs} — "
              f"Train Loss: {avg_loss:.4f}, Val Acc: {acc:.2f}%")

        # 保存最优
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.out_model)

    print(f"✅ Training done, best Val Acc: {best_acc:.2f}%, "
          f"model saved to {args.out_model}")

if __name__ == "__main__":
    main()
