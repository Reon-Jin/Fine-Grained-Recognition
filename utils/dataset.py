
import os
import random
from typing import Tuple, Union, List, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as F


__all__ = ["get_data_loaders", "KeepRatioResizePad"]


# ------------------------------
# Transforms
# ------------------------------
class KeepRatioResizePad:
    """
    Resize while keeping aspect ratio, then pad to the target size (letterbox).

    Args:
        size: int or (H, W). Final canvas size.
        interpolation: PIL interpolation mode.
        fill: padding value (0-255). For RGB will be tripled to (fill, fill, fill).
    """
    def __init__(self, size: Union[int, Tuple[int, int]], interpolation=Image.BILINEAR, fill: int = 128):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = (int(size[0]), int(size[1]))
        self.interp = interpolation
        self.fill = int(fill)

    def __call__(self, img: Image.Image) -> Image.Image:
        th, tw = self.size
        w, h = img.size  # PIL: (W, H)
        if w == 0 or h == 0:
            raise ValueError("Invalid image with zero width/height")

        scale = min(tw / w, th / h)
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = img.resize((new_w, new_h), resample=self.interp)

        pad_left  = (tw - new_w) // 2
        pad_top   = (th - new_h) // 2

        if img.mode == "RGB":
            fill_color = (self.fill, self.fill, self.fill)
        elif img.mode == "L":
            fill_color = self.fill
        else:
            # Fallback for other modes
            fill_color = 0

        canvas = Image.new(img.mode, (tw, th), fill_color)
        canvas.paste(resized, (pad_left, pad_top))
        return canvas


def _to_hw(size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(size, int):
        return size, size
    return int(size[0]), int(size[1])


def _build_transforms(image_size: Union[int, Tuple[int, int]], keep_ratio: bool, train: bool):
    H, W = _to_hw(image_size)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    ops = []
    if keep_ratio:
        ops.append(KeepRatioResizePad((H, W)))
    else:
        ops.append(transforms.Resize((H, W)))
    if train:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(ops)


# ------------------------------
# Stratified split helpers
# ------------------------------
def _stratified_indices(samples: List[Tuple[str, int]], num_classes: int, val_split: float, seed: int):
    """
    samples: list of (path, class_idx)
    returns: train_indices, val_indices (indices into 'samples')
    """
    rng = random.Random(seed)
    per_class: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    for i, (_, c) in enumerate(samples):
        per_class[c].append(i)

    train_idx: List[int] = []
    val_idx: List[int] = []
    for c in range(num_classes):
        idxs = per_class[c]
        rng.shuffle(idxs)
        n = len(idxs)
        n_val = int(round(n * val_split))
        # Ensure at least 1 sample in val when possible
        if n > 0 and val_split > 0 and n_val == 0:
            n_val = 1
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    # Shuffle overall order for loaders (they will shuffle anyway for train)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _seed_worker(worker_id: int):
    # Ensure dataloader workers are deterministic
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ------------------------------
# Public API
# ------------------------------
def get_data_loaders(
    data_dir: str,
    batch_size: int,
    image_size: Union[int, Tuple[int, int]],
    num_workers: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
    keep_ratio: bool = True,
    shuffle_train: bool = True,
    pin_memory: bool = True,
):
    """
    Returns train_loader, val_loader, num_classes.

    Behavior:
      • If data_dir contains 'train/' and 'val/' subfolders, those are used as-is.
      • Otherwise, we perform a stratified split on-the-fly without touching disk.

    Directory formats supported:
      1) Pre-split:
         data_dir/
           train/class_x/xxx.jpg
           val/class_x/yyy.jpg
      2) Single folder (auto split):
         data_dir/
           class_x/xxx.jpg
           class_y/zzz.jpg
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")

    train_tf = _build_transforms(image_size, keep_ratio=keep_ratio, train=True)
    val_tf   = _build_transforms(image_size, keep_ratio=keep_ratio, train=False)

    g = torch.Generator()
    g.manual_seed(seed)

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        # Use existing split
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        val_ds   = datasets.ImageFolder(val_dir,   transform=val_tf)

        # Check class names align
        if train_ds.classes != val_ds.classes:
            # Map val targets to train's class_to_idx via target_transform
            train_class_to_idx = {c: i for i, c in enumerate(train_ds.classes)}
            def _map_target(t):
                cls_name = val_ds.classes[t]
                return train_class_to_idx[cls_name]
            val_ds = datasets.ImageFolder(val_dir, transform=val_tf, target_transform=_map_target)

        num_classes = len(train_ds.classes)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=shuffle_train,
            num_workers=num_workers, pin_memory=pin_memory,
            worker_init_fn=_seed_worker, generator=g, drop_last=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            worker_init_fn=_seed_worker, generator=g, drop_last=False
        )
        return train_loader, val_loader, num_classes

    # Auto-split path: a single root folder with class subfolders
    base_full_for_split = datasets.ImageFolder(data_dir, transform=None)
    num_classes = len(base_full_for_split.classes)
    train_idx, val_idx = _stratified_indices(base_full_for_split.samples, num_classes, val_split, seed)

    # Create two separate ImageFolder datasets so they can have different transforms
    train_full = datasets.ImageFolder(data_dir, transform=train_tf)
    val_full   = datasets.ImageFolder(data_dir, transform=val_tf)

    train_ds = Subset(train_full, train_idx)
    val_ds   = Subset(val_full,   val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=_seed_worker, generator=g, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=_seed_worker, generator=g, drop_last=False
    )
    return train_loader, val_loader, num_classes
