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

# 常量配置
IMG_SIZE = 224
NUM_CLASSES = 400
ROOT_DIR = "data/WebFG-400"
FILTERED_TRAIN = os.path.join(ROOT_DIR, "filtered_train.json")
FILTERED_VAL   = os.path.join(ROOT_DIR, "filtered_val.json")

# 图像增强与归一化

# 对训练集和验证集使用相同的增广策略（可根据需要区别）
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random')
    ]),
    "val": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# YOLOv5 用于噪声过滤
yolo_model = YOLO('yolov5su.pt')  # 可替换为更轻量的 'yolov5n.pt'

class MultiClassDataset(Dataset):
    """多分类数据集：
    训练集和验证集都会用 YOLO 过滤噪声并缓存；
    测试集直接按目录加载，无缓存。"""
    def __init__(self, root_dir, json_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # 数据均来自 train 目录
        basename = os.path.basename(json_file)
        if basename in ("train.json", "val.json"):
            data_folder = "train"
            cache_file = FILTERED_TRAIN if basename == "train.json" else FILTERED_VAL
        else:
            data_folder = None
            cache_file = None

        # classes 始终根据 train 目录获取
        train_dir = os.path.join(root_dir, "train")
        self.classes = sorted([
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 从 JSON 读取相对路径列表
        with open(json_file, 'r') as f:
            relative_paths = json.load(f)

        # 过滤或直接使用原列表
        if cache_file:
            if os.path.exists(cache_file):
                filtered = json.load(open(cache_file, 'r'))
                print(f"加载缓存过滤列表：{cache_file}")
            else:
                grouped = defaultdict(list)
                for rel in relative_paths:
                    cls_name = os.path.normpath(rel).split(os.sep)[0]
                    grouped[cls_name].append(rel)
                filtered = []
                for cls_name, rels in grouped.items():
                    counts = {'bird': 0, 'car': 0, 'plane': 0}
                    cache = {}
                    for rel in rels:
                        img_path = os.path.join(root_dir, data_folder, rel)
                        try:
                            results = yolo_model(img_path, verbose=False, max_det=2)
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
                # 写入缓存
                with open(cache_file, 'w') as wf:
                    json.dump(filtered, wf)
                print(f"过滤结果已保存：{cache_file}")
        else:
            filtered = relative_paths

        # 构建绝对路径和标签列表
        base_folder = os.path.join(root_dir, data_folder) if data_folder else root_dir
        self.image_paths = [os.path.join(base_folder, rel) for rel in filtered]
        self.labels_list = [self.class_to_idx[os.path.normpath(rel).split(os.sep)[0]] for rel in filtered]

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

# 构建函数
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
