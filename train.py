# train.py
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import get_train_dataset, get_val_dataset
from model import AIModel

# 单轮训练
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

# 验证
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# 主训练脚本
def main():
    root = 'data/WebFG-400'  # 根目录，包含 train/ 和 test/
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device=",device)
    train_set = get_train_dataset(root)
    val_set   = get_val_dataset(root)
    num_classes = len(train_set.classes)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4)

    model = AIModel('efficientnet-b0', num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_acc = 0
    for epoch in range(1, 10):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        scheduler.step()
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'model/model.pth')
    # 最终模型
    torch.save(model.state_dict(), 'model/model_final.pth')

if __name__ == '__main__':
    main()