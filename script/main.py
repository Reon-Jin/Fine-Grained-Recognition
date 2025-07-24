import os
import sys
import yaml
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim import SGD

from opt_dg_tf2_new import DirectoryDataset
from models import construct_model

# ---------------- Load configuration ----------------
param_dir = "../config.yaml"
with open(param_dir, 'r') as file:
    param = yaml.load(file, Loader=yaml.FullLoader)
print('Loading Default parameter configuration:\n', json.dumps(param, sort_keys=True, indent=3))

# Data parameters
nb_classes = param['DATA']['nb_classes']
image_size = tuple(param['DATA']['image_size'])
dataset_dir = param['DATA']['dataset_dir']

# Model parameters
batch_size = param['MODEL']['batch_size']
lr = param['MODEL']['learning_rate']
model_name = param['MODEL']['model_name']

# Training parameters
epochs = param['TRAIN']['epochs']

# Override from command line
if len(sys.argv) > 2:
    total_params = len(sys.argv)
    for i in range(1, total_params, 2):
        var_name = sys.argv[i]
        new_val = sys.argv[i + 1]
        try:
            exec(f"{var_name} = {new_val}")
        except Exception:
            exec(f"{var_name} = '{new_val}'")

# Paths
working_dir = os.path.dirname(os.path.realpath(__file__))
train_data_dir = f"{dataset_dir}/train/"

# Dataset and dataloaders
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = DirectoryDataset([train_data_dir], augment=True, preprocess=None, target_size=image_size)
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# Model
model = construct_model(
    name=model_name,
    pool_size=7,
    ROIS_resolution=42,
    ROIS_grid_size=3,
    minSize=2,
    nb_classes=nb_classes,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = correct / total if total > 0 else 0
    print(f"Epoch {epoch+1}/{epochs} - Val Acc: {acc:.4f}")

