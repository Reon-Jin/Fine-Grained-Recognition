import os
import json
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision.transforms import InterpolationMode

# 防止因损坏的 PNG 导致 PIL 报错
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 常量配置
IMG_SIZE = 224
NUM_CLASSES = 400
ROOT_DIR = "data/WebFG-400"
DATA_FOLDER = "train_filter"   # 训练／验证都从这里读类文件夹

# 图像增强与归一化
# ForegroundExtraction 会在 Resize 后，将背景设为黑

def extract_foreground(image_np):
    """
    对输入的 RGB numpy 图像执行 GrabCut 分割,
    将背景设为黑色, 返回前景提取后的图像
    """
    h, w = image_np.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # 使用中心大矩形作为前景初始化区域
    rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
    cv2.grabCut(image_np, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    # 将 sure bg / prob bg -> 0, sure fg / prob fg -> 1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # 应用掩码
    foreground = image_np * mask2[:, :, np.newaxis]
    return foreground

# 定义 transform pipeline
# 首先 Resize 到 IMG_SIZE, 提取前景, 然后其余变换
class PreprocessResizeAndSegment:
    def __call__(self, img: Image.Image) -> Image.Image:
        # Resize
        img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.BICUBIC)
        # 转为 numpy 并提取前景
        img_np = np.array(img)
        fg_np = extract_foreground(img_np)
        # 转回 PIL
        return Image.fromarray(fg_np)

# 数据增强与归一化
data_transforms = {
    "train": transforms.Compose([
        PreprocessResizeAndSegment(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random'),
    ]),
    "val": transforms.Compose([
        PreprocessResizeAndSegment(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "test": transforms.Compose([
        PreprocessResizeAndSegment(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}


class MultiClassDataset(Dataset):
    """多分类数据集：从 train_filter 目录加载 train/val 列表，自动跳过不存在的文件。"""
    def __init__(self, root_dir, json_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # 统一使用 train_filter 目录
        base_folder = os.path.join(root_dir, DATA_FOLDER)

        # 构建类别列表和映射
        self.classes = sorted([
            d for d in os.listdir(base_folder)
            if os.path.isdir(os.path.join(base_folder, d))
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 读取 JSON 相对路径
        with open(json_file, 'r') as f:
            relative_paths = json.load(f)

        # 过滤掉磁盘上不存在的图片
        valid_paths = []
        for rel in relative_paths:
            abs_path = os.path.join(base_folder, rel)
            if os.path.isfile(abs_path):
                valid_paths.append(rel)

        # 构建绝对路径和标签列表
        self.image_paths = [
            os.path.join(base_folder, rel) for rel in valid_paths
        ]
        self.labels_list = [
            self.class_to_idx[os.path.normpath(rel).split(os.sep)[0]]
            for rel in valid_paths
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels_list[idx]


class TestDataset(Dataset):
    """测试集：无标签，仅按目录加载所有图像。"""
    def __init__(self, test_dir, transform=None):
        self.transform = transform
        self.image_paths = [
            os.path.join(test_dir, f) for f in os.listdir(test_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(path)


# 构建 train/val dataset

def get_train_dataset():
    return MultiClassDataset(
        root_dir=ROOT_DIR,
        json_file=os.path.join(ROOT_DIR, "train.json"),
        transform=data_transforms['train']
    )

def get_val_dataset():
    return MultiClassDataset(
        root_dir=ROOT_DIR,
        json_file=os.path.join(ROOT_DIR, "val.json"),
        transform=data_transforms['val']
    )
