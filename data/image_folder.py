# -*- coding: utf-8 -*-
from torchvision.datasets import DatasetFolder
from PIL import Image
from tqdm import tqdm
import os
from PIL import ImageFile
import warnings

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class IndexedImageFolder(DatasetFolder):
    def __init__(self, root, use_cache=False, transform=None, target_transform=None,
                 loader=pil_loader, is_valid_file=None):
        super().__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                         transform=transform,
                         target_transform=target_transform,
                         is_valid_file=is_valid_file)
        self.imgs = self.samples  # list, element is (path, label)
        self.root_dir = root  # ✅ 记录根目录
        self.use_cache = use_cache

        if self.use_cache:
            self.loaded_samples = self._cache_dataset()  # list, element is (PIL image, label)
        else:
            self.loaded_samples = None

    def _cache_dataset(self):
        cached_dataset = []
        n_samples = len(self.samples)
        print('caching samples ... ')
        for idx, sample in enumerate(tqdm(self.samples, ncols=100, ascii=' >')):
            path, target = sample
            image = self.loader(path)
            cached_dataset.append((image, target))
        assert len(cached_dataset) == n_samples
        return cached_dataset

    def __getitem__(self, index):
        if self.use_cache:
            assert len(self.loaded_samples) == len(self.samples)
            sample, target = self.loaded_samples[index]
            path, _ = self.samples[index]  # ✅ 取回路径
        else:
            path, target = self.samples[index]
            sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # ✅ 新增路径字段（使用相对路径匹配 CSV）
        rel_path = os.path.relpath(path, self.root_dir)
        rel_path = rel_path.replace("\\", "/")  # 确保路径统一（000/1.jpg）

        return {
            'index': index,
            'data': sample,
            'label': target,
            'path': rel_path  # ✅ 新增字段
        }
