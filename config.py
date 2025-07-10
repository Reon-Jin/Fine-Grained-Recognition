import os
import json
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision.transforms import InterpolationMode

# Prevent PIL from crashing on truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ========== Global Hyperparameters ==========
# Training settings
BATCH_SIZE = 16
LR = 0.001
WD = 1e-4
SAVE_FREQ = 1
resume = ''  # Path to checkpoint for resuming training
# Model settings
PROPOSAL_NUM = 6  # Number of region proposals (M)
CAT_NUM = 400     # Number of target classes
NUM_CLASSES = CAT_NUM  # Alias for compatibility
INPUT_SIZE = (224, 224)  # Input size for backbone (width, height)
# Checkpoint & logging
test_model = 'model.ckpt'
save_dir = 'model/'

# ========== Dataset & Transforms Settings ==========
IMG_SIZE = 224              # Input size for crop & resize in dataset
ROOT_DIR = "data/WebFG-400"  # Root directory containing train/, val/, test/

# Data augmentation and normalization
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random')
    ]),
    "val": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

class MultiClassDataset(Dataset):
    """Multi-class dataset: loads image paths and labels from a JSON file."""
    def __init__(self, root_dir, json_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Directory containing class subfolders under train/
        train_folder = os.path.join(root_dir, "train")
        # Scan subdirectories as class labels
        self.classes = sorted([
            d for d in os.listdir(train_folder)
            if os.path.isdir(os.path.join(train_folder, d))
        ])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Read list of relative paths from JSON
        with open(json_file, 'r') as f:
            relative_paths = json.load(f)

        self.image_paths = []
        self.labels = []
        for rel in relative_paths:
            cls_name = os.path.normpath(rel).split(os.sep)[0]
            self.image_paths.append(os.path.join(train_folder, rel))
            self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

class TestDataset(Dataset):
    """Test dataset: loads images without labels from a directory."""
    def __init__(self, test_dir, transform=None):
        # Collect all image file paths
        self.image_paths = [
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir)
            if f.lower().endswith(('.png', 'jpg', 'jpeg', 'bmp'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # Return image tensor and filename for identification
        return img, os.path.basename(img_path)

# Helper functions to construct datasets

def get_train_dataset():
    return MultiClassDataset(
        root_dir=ROOT_DIR,
        json_file=os.path.join(ROOT_DIR, "train.json"),
        transform=data_transforms["train"]
    )


def get_val_dataset():
    return MultiClassDataset(
        root_dir=ROOT_DIR,
        json_file=os.path.join(ROOT_DIR, "val.json"),
        transform=data_transforms["val"]
    )


def get_test_dataset():
    return TestDataset(
        test_dir=os.path.join(ROOT_DIR, "test"),
        transform=data_transforms["test"]
    )
