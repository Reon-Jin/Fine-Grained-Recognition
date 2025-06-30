# config.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

_train_subset = None
_val_subset = None


class ImageFolderSubset(Dataset):
    """Subset of ``ImageFolder`` that exposes ``classes`` attribute."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def _prepare_datasets(root_dir):
    """Split the training folder into train/val subsets (80/20)."""
    global _train_subset, _val_subset
    if _train_subset is not None and _val_subset is not None:
        return

    base_dir = os.path.join(root_dir, 'train')
    full_dataset = ImageFolder(base_dir)
    n_total = len(full_dataset)
    val_len = int(n_total * 0.2)
    train_len = n_total - val_len
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]

    train_set = ImageFolder(base_dir, transform=data_transforms['train'])
    val_set = ImageFolder(base_dir, transform=data_transforms['test'])
    _train_subset = ImageFolderSubset(train_set, train_indices)
    _val_subset = ImageFolderSubset(val_set, val_indices)

# 统一图像尺寸
imgsz = 224

# 数据预处理管道
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# 训练/验证集自动读取子目录作为类别
def get_train_dataset(root_dir):
    """Return the training subset (80%)."""
    _prepare_datasets(root_dir)
    return _train_subset

def get_val_dataset(root_dir):
    """Return the validation subset (20%)."""
    _prepare_datasets(root_dir)
    return _val_subset

# 测试集：无标签，仅返回图像和文件名
class TestDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root = root_dir
        self.names = sorted(os.listdir(root_dir))
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        fname = self.names[idx]
        path = os.path.join(self.root, fname)
        img = Image.open(path).convert('RGB')
        # return the file name with extension for saving results
        return self.transform(img), fname
