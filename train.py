# train.py
import os
import time
from typing import Optional
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import get_train_dataset, get_val_dataset
from model import AIModel

# 单轮训练
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    return avg_loss, acc

# 验证
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    return avg_loss, acc

# 主训练脚本
def main():
    parser = argparse.ArgumentParser()
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', default=default_device,
                        help='training device, e.g. "cuda:0" or "cpu"')
    parser.add_argument('--root', default='data/WebFG-400',
                        help='dataset root directory')
    parser.add_argument('--logdir', default=None,
                        help='directory for TensorBoard logs')
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    root = args.root  # 根目录，包含 train/ 和 test/
    print("device=", device)
    writer: Optional[SummaryWriter] = SummaryWriter(args.logdir) if args.logdir else None
    train_set = get_train_dataset(root)
    val_set   = get_val_dataset(root)
    num_classes = len(train_set.classes)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

    model = AIModel('efficientnet-b2', num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_acc = 0
    for epoch in range(1, 101):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Acc/train', train_acc, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Acc/val', val_acc, epoch)
        scheduler.step()
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'model/model.pth')
    # 最终模型
    torch.save(model.state_dict(), 'model/model_final.pth')
    if writer:
        writer.close()

if __name__ == '__main__':
    main()
