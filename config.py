import os
import json
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision.transforms import InterpolationMode
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 通用设置
IMG_SIZE = 224
NUM_CLASSES = 400  # 可自动获取，无需手动写死
ROOT_DIR = "data/WebFG-400"  # 根目录，包含 train/、train.json、val.json

# 图像增强与归一化
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
    """Multi-class classification dataset using JSON-specified image paths."""

    def __init__(self, root_dir, json_file, transform=None):
        """
        Args:
            root_dir (str): 根目录，包含 'train' 文件夹。
            json_file (str): 包含相对路径列表（相对于 'train' 子目录）的 JSON 文件。
            transform (callable, optional): 图像预处理函数。
        """
        self.root_dir = root_dir
        self.transform = transform

        # 构建类别映射
        train_folder = os.path.join(root_dir, "train")
        self.classes = sorted([
            d for d in os.listdir(train_folder)
            if os.path.isdir(os.path.join(train_folder, d))
        ])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # 读取 JSON 获取图像路径
        with open(json_file, 'r') as f:
            relative_paths = json.load(f)

        self.image_paths = []
        self.labels_list = []

        for rel_path in relative_paths:
            class_name = os.path.normpath(rel_path).split(os.sep)[0]
            label = self.class_to_idx[class_name]
            full_path = os.path.join(train_folder, rel_path)
            self.image_paths.append(full_path)
            self.labels_list.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels_list[idx]
        return image, label

class TestDataset(Dataset):
    """Test dataset that reads all images from a folder without labels."""
    def __init__(self, test_dir, transform=None):
        self.image_paths = [
            os.path.join(test_dir, fname)
            for fname in os.listdir(test_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        self.image_names = [os.path.basename(p) for p in self.image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_names[idx]

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
