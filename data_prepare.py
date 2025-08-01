"""Utility script to split raw dataset into train/val folders.

Assumes the raw DATA_DIR contains one sub‑folder per class with images.
"""
import os, random, shutil
from config import DATA_DIR, TRAIN_DIR, VAL_DIR, SEED

def split_dataset(val_ratio: float = 0.1):
    random.seed(SEED)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR,   exist_ok=True)

    for cls in os.listdir(DATA_DIR):
        cls_path = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_path):
            continue
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        val_count   = int(len(images) * val_ratio)
        val_images  = images[:val_count]
        train_images = images[val_count:]

        os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR,   cls), exist_ok=True)

        for img in train_images:
            shutil.copy(os.path.join(cls_path, img), os.path.join(TRAIN_DIR, cls, img))
        for img in val_images:
            shutil.copy(os.path.join(cls_path, img), os.path.join(VAL_DIR, cls, img))
    print(f"Done. Train/val split saved to {TRAIN_DIR} and {VAL_DIR}.")

if __name__ == '__main__':
    split_dataset()
