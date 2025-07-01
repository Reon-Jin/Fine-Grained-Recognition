# train.py -- Training script for ViT-B/16 on custom dataset
import os, argparse, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import get_train_dataset, get_val_dataset
from model import AIModel

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser()
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', default=default_device,
                        help='"cuda", "cuda:0", "cpu", or GPU index like "0"')
    parser.add_argument('--root', default='data/WebFG-400',
                        help='Dataset root containing train/ and test/')
    args = parser.parse_args()

    dev_arg = args.device.strip()
    if dev_arg.isdigit():
        dev_arg = f'cuda:{dev_arg}'
    device = torch.device(dev_arg)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    train_set = get_train_dataset(args.root)
    val_set = get_val_dataset(args.root)
    num_classes = len(train_set.classes)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

    model = AIModel('vit-b/16', num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_acc = 0.0
    os.makedirs('model', exist_ok=True)
    for epoch in range(1, 51):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch:03d} | Train {train_loss:.4f} | Val {val_loss:.4f} | Acc {val_acc:.4f}')
        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'model/model.pth')
    torch.save(model.state_dict(), 'model/model_final.pth')

if __name__ == '__main__':
    main()
