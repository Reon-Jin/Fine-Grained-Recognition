# config.py -- JSON-based dataset loader configuration
import os, json
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Unified image size for ViT
imgsz = 224

# Common data transformation pipelines
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),    
}

def _load_list(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

class JsonLabeledDataset(Dataset):
    """Dataset for train/val with labels inferred from first directory level."""
    def __init__(self, root_dir: str, json_file: str, transform):
        self.root_dir = root_dir
        self.rel_paths = _load_list(os.path.join(root_dir, json_file))
        self.transform = transform

        # Build class list from first folder in each path
        folders = sorted({p.split(os.sep)[0] for p in self.rel_paths})
        self.classes = folders
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, idx):
        rel_path = self.rel_paths[idx]
        # Image path: root_dir/train/<rel_path>
        img_path = os.path.join(self.root_dir,  rel_path)
        img = Image.open(img_path).convert('RGB')
        label_name = rel_path.split(os.sep)[0]
        label = self.class_to_idx[label_name]
        return self.transform(img), label

class JsonTestDataset(Dataset):
    """Dataset for test set without labels."""
    def __init__(self, root_dir: str, json_file: str, transform):
        self.root_dir = root_dir
        self.rel_paths = _load_list(os.path.join(root_dir, json_file))
        self.transform = transform

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, idx):
        rel_path = self.rel_paths[idx]
        img_path = os.path.join(self.root_dir, rel_path)
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), rel_path  # return name for saving results

# Public factory functions
def get_train_dataset(root_dir: str):
    return JsonLabeledDataset(root_dir, 'train.json', data_transforms['train'])

def get_val_dataset(root_dir: str):
    return JsonLabeledDataset(root_dir, 'val.json', data_transforms['val'])

def get_test_dataset(root_dir: str):
    return JsonTestDataset(root_dir, 'test.json', data_transforms['test'])
