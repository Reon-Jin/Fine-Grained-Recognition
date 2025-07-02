import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import BCNN
from config import get_dataloaders, LEARNING_RATE, EPOCHS
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes")



def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data loaders
    train_loader, val_loader = get_dataloaders()

    # determine number of classes
    num_classes = len(train_loader.dataset.dataset.classes)

    # model, criterion, optimizer
    model = BCNN(num_classes=num_classes, unfreeze_last_stage=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # tensorboard
    writer = SummaryWriter()

    # prepare model save directory
    save_dir = 'model'
    os.makedirs(save_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training with progress bar
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", ncols=100)
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = imgs.size(0)
            running_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += batch_size

            # update progress bar
            train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Acc', epoch_acc, epoch)

        # validation with progress bar
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]  ", ncols=100)
            with torch.no_grad():
                for imgs, labels in val_bar:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                    batch_size = imgs.size(0)
                    val_loss += loss.item() * batch_size
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += batch_size

                    # update progress bar
                    val_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{val_correct/val_total:.4f}")

            val_epoch_loss = val_loss / val_total
            val_epoch_acc = val_correct / val_total
            writer.add_scalar('Val/Loss', val_epoch_loss, epoch)
            writer.add_scalar('Val/Acc', val_epoch_acc, epoch)

            # save best model
            if val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
                save_path = os.path.join(save_dir, 'model.pth')
                torch.save(model.state_dict(), save_path)
                tqdm.write(f"Saved best model with Val Acc: {best_acc:.4f} to {save_path}")

    writer.close()


if __name__ == '__main__':
    train()