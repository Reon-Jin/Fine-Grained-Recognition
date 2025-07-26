import os

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from tqdm import tqdm

from config import Config
from data_prepare import prepare_dataloaders
from models.model import FineGrainedModel


def accuracy(outputs, labels):
    _, preds = outputs.max(1)
    return (preds == labels).float().mean().item()


def train():
    config = Config()
    train_loader, val_loader, num_classes = prepare_dataloaders(config)
    model = FineGrainedModel(num_classes).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_acc = 0.0
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    plt.ion()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}", ncols=100)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_acc += (outputs.argmax(1) == labels).sum().item()
            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_acc / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_acc += (outputs.argmax(1) == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, 'best_model.pth'))

        # update plots
        ax[0].clear()
        ax[1].clear()
        ax[0].plot(train_losses, label='train_loss')
        ax[0].plot(val_losses, label='val_loss')
        ax[0].legend()
        ax[1].plot(train_accs, label='train_acc')
        ax[1].plot(val_accs, label='val_acc')
        ax[1].legend()
        plt.pause(0.01)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    train()
