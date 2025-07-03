import os
from datetime import datetime
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import ImageFile

from model import BilinearCNN        # 改成新模型
from config import get_train_dataset, get_val_dataset

# 忽略 PIL 那个警告
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images"
)

# TensorBoard log dir
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("runs", current_time)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc=f"Train Epoch {epoch+1}", ncols=100):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    writer.add_scalar("Loss/Train", epoch_loss, epoch + 1)
    writer.add_scalar("Accuracy/Train", epoch_acc, epoch + 1)
    print(f"[Train]  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validate", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    writer.add_scalar("Loss/Val", epoch_loss, epoch + 1)
    writer.add_scalar("Accuracy/Val", epoch_acc, epoch + 1)
    print(f"[Valid]  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}\n")
    return epoch_loss, epoch_acc

if __name__ == "__main__":
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集
    train_dataset = get_train_dataset()
    val_dataset   = get_val_dataset()
    num_classes   = len(train_dataset.classes)

    # DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    # 模型
    model = BilinearCNN(
        num_classes=num_classes,
        dropout_rate=0.5,
        freeze_stream2=True     # 可根据需要微调
    ).to(device)

    # 优化器、调度器、损失
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    criterion = nn.CrossEntropyLoss()

    # 确保保存模型目录存在
    os.makedirs("model", exist_ok=True)
    best_acc = 0.0
    num_epochs = 50

    for epoch in range(num_epochs):
        print(f"=== Epoch {epoch+1}/{num_epochs} ===")
        train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step(epoch + epoch/len(train_loader))
        _, val_acc = validate_epoch(model, val_loader, criterion, device, epoch)

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "model/model.pth")

        # 每 epoch 都保存最新模型
        torch.save(model.state_dict(), "model/model_final.pth")

    writer.close()
