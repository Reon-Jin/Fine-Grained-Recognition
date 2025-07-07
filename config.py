# config.py

import os
import json
from collections import defaultdict

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision.transforms import InterpolationMode

from ultralytics import YOLO

# 防止因损坏的 PNG 导致 PIL 报错
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 初始化 YOLOv5s，用于噪声过滤
yolo_model = YOLO('yolov5su.pt')  # 可替换为更轻量的 'yolov5n.pt'

# 常量配置
IMG_SIZE = 224
NUM_CLASSES = 400
ROOT_DIR = "data/WebFG-400"
FILTERED_JSON = os.path.join(ROOT_DIR, "filtered_train.json")

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
    """多分类数据集：先用 YOLO 过滤噪声，然后按 JSON 列表加载。"""

    def __init__(self, root_dir, json_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        train_folder = os.path.join(root_dir, "train")
        # 自动扫描所有细分类别文件夹
        self.classes = sorted([
            d for d in os.listdir(train_folder)
            if os.path.isdir(os.path.join(train_folder, d))
        ])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # 读取原始路径列表
        with open(json_file, 'r') as f:
            relative_paths = json.load(f)

        # 如果已有缓存，直接加载
        if os.path.exists(FILTERED_JSON):
            filtered = json.load(open(FILTERED_JSON, 'r'))
            print(f"加载缓存的过滤列表：{FILTERED_JSON}")
        else:
            # 否则按组过滤一次
            grouped = defaultdict(list)
            for rel in relative_paths:
                top_cls = os.path.normpath(rel).split(os.sep)[0]
                grouped[top_cls].append(rel)

            filtered = []
            for cls_name, rels in grouped.items():
                counts = {'bird': 0, 'car': 0, 'plane': 0}
                cache = {}
                for rel in rels:
                    full_path = os.path.join(train_folder, rel)
                    try:
                        # 关闭日志，限制最多两个检测框
                        results = yolo_model(full_path, verbose=False, max_det=2)
                        if results and len(results) > 0:
                            res = results[0]
                            if hasattr(res, 'boxes') and len(res.boxes) > 0:
                                detected = {res.names[int(c)] for c in res.boxes.cls.cpu().numpy()}
                                if 'bird' in detected:
                                    counts['bird'] += 1; cache[rel] = 'bird'
                                elif 'car' in detected:
                                    counts['car'] += 1; cache[rel] = 'car'
                                elif 'airplane' in detected:
                                    counts['plane'] += 1; cache[rel] = 'plane'
                                else:
                                    cache[rel] = None
                            else:
                                cache[rel] = None
                        else:
                            cache[rel] = None
                    except Exception:
                        cache[rel] = None

                main = max(counts, key=counts.get)
                if counts[main] > 0:
                    filtered += [r for r in rels if cache.get(r) == main]
                else:
                    filtered += rels

            # 保存缓存，下次直接加载
            with open(FILTERED_JSON, 'w') as wf:
                json.dump(filtered, wf)
            print(f"过滤结果已保存到：{FILTERED_JSON}")

        # 构建最终的路径和标签列表
        self.image_paths = []
        self.labels_list = []
        for rel in filtered:
            cls0 = os.path.normpath(rel).split(os.sep)[0]
            self.image_paths.append(os.path.join(train_folder, rel))
            self.labels_list.append(self.class_to_idx[cls0])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels_list[idx]


class TestDataset(Dataset):
    """测试集：无标签，仅按路径加载所有图像。"""
    def __init__(self, test_dir, transform=None):
        self.image_paths = [
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir)
            if f.lower().endswith(('.png','jpg','jpeg','bmp'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(self.image_paths[idx])


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
