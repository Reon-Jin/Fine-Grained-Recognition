import os
from datetime import datetime
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import ImageFile

from model import MultiStreamFeatureExtractor  # 新模型
from config import get_train_dataset, get_val_dataset

# 忽略 PIL 警告
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images"
)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Train Epoch {epoch+1}", ncols=100, colour="white")
    for inputs, labels in progress_bar:
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

        current_loss = running_loss / total
        current_acc = correct / total
        progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

    print(f"[Train]  Loss: {current_loss:.4f}  Acc: {current_acc:.4f}")
    return current_loss, current_acc

def validate_epoch(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Validate", ncols=100, colour="white")
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            current_loss = running_loss / total
            current_acc = correct / total
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

    print(f"[Valid]  Loss: {current_loss:.4f}  Acc: {current_acc:.4f}\n")
    return current_loss, current_acc

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = get_train_dataset()
    val_dataset   = get_val_dataset()
    num_classes   = len(train_dataset.classes)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    model = MultiStreamFeatureExtractor(
        num_classes=num_classes,
        reduction_dim=512,
        dropout_rate=0.5,
        freeze_streams=[1,2,3,4]
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.2, patience=3, verbose=True
    )

    criterion = nn.CrossEntropyLoss()
    os.makedirs("model", exist_ok=True)
    best_acc = 0.0
    num_epochs = 50

    for epoch in range(num_epochs):
        print(f"=== Epoch {epoch+1}/{num_epochs} ===")
        train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step(epoch + epoch/len(train_loader))
        _, val_acc = validate_epoch(model, val_loader, criterion, device, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "model/model.pth")
        torch.save(model.state_dict(), "model/model_final.pth")
