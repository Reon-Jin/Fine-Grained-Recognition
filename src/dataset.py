# src/dataset.py
import os
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from .config import train_transform, val_transform

# 允许加载截断／损坏的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageList(Dataset):
    """
    普通图像数据集，返回 (img_tensor, label)
    """
    def __init__(self, list_file, train=True):
        lines = open(list_file).read().splitlines()
        self.samples = [(l.split()[0], int(l.split()[1])) for l in lines]
        self.transform = train_transform if train else val_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except OSError:
            img = Image.new('RGB', (224,224), (0,0,0))
        return self.transform(img), label

class PartialLabelDataset(Dataset):
    """
    部分标签学习 Dataset，维护每个样本的候选标签集 C(x)
    返回 (img_tensor, sample_index, candidate_label_list)
    """
    def __init__(self, list_file):
        lines = open(list_file).read().splitlines()
        self.paths = [l.split()[0] for l in lines]
        self.gt    = [int(l.split()[1]) for l in lines]
        # 初始候选集: 只有原标签（使用 list 而不是 set）
        self.C = [ [lbl] for lbl in self.gt ]
        self.transform = train_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert('RGB')
        except OSError:
            img = Image.new('RGB', (224,224), (0,0,0))
        img = self.transform(img)
        # 直接返回 list，不要用 set
        cand = self.C[idx]
        return img, idx, cand
