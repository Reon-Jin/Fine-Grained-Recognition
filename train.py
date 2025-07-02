import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import AIModel
from config import *
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes should be converted to RGBA images")

# Log directory
current_time = datetime.now().strftime("%d-%H-%M-%S")
log_dir = os.path.join("runs", current_time)
writer = SummaryWriter(log_dir)


def train(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += (preds == labels).sum().item()
        total_predictions += labels.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_predictions / total_predictions
    writer.add_scalar("Loss/Train", epoch_loss, epoch + 1)
    writer.add_scalar("Accuracy/Train", epoch_acc, epoch + 1)
    print(f"Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")


def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_predictions / total_predictions
    writer.add_scalar("Loss/Validation", epoch_loss, epoch + 1)
    writer.add_scalar("Accuracy/Validation", accuracy, epoch + 1)
    print(f"Validation Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}\n")
    return accuracy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset from JSON
    train_dataset = get_train_dataset()
    val_dataset = get_val_dataset()

    # Automatically get number of classes
    num_classes = len(train_dataset.classes)

    # Build model
    model = AIModel(num_classes=num_classes).to(device)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Optimizer, scheduler, loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss()

    # Train loop
    num_epochs = 100
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step(epoch + epoch / len(train_loader))
        acc = validate(model, val_loader, criterion, device, epoch)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "model/model.pth")
        torch.save(model.state_dict(), "model/model_final.pth")

    writer.close()
