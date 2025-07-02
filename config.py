from PIL import ImageFile
# allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

# dataset path (all images)
DATA_DIR = os.path.join('data', 'WebFG-400', 'train')

# training hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
EPOCHS = 50
VALID_SPLIT = 0.2  # proportion for validation
SEED = 42


def get_dataloaders(data_dir=DATA_DIR):
    # transforms
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # get splits
    base_dataset = datasets.ImageFolder(data_dir, transform=None)
    dataset_size = len(base_dataset)
    indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(SEED)).tolist()
    val_size = int(dataset_size * VALID_SPLIT)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    # create subsets with appropriate transforms
    train_dataset = Subset(
        datasets.ImageFolder(data_dir, transform=train_transforms), train_indices
    )
    val_dataset = Subset(
        datasets.ImageFolder(data_dir, transform=val_transforms), val_indices
    )

    # dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    return train_loader, val_loader