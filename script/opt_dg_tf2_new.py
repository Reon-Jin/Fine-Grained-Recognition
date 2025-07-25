"""PyTorch dataset and dataloader utilities."""
import random
from os import listdir
from os.path import isdir, join, isfile

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DirectoryDataset(Dataset):
    def __init__(self, base_directories, augment=False, preprocess=None, target_size=(224, 224), aug_config=None):
        self.base_directories = base_directories
        self.augment = augment
        self.preprocess = preprocess
        self.target_size = target_size
        # Store augmentation parameters from config
        self.aug_config = aug_config or {}

        self.class_names = []
        self.files = []
        self.labels = []

        for base_directory in base_directories:
            class_names = [x for x in listdir(base_directory) if isdir(join(base_directory, x))]
            class_names = sorted(class_names)
            self.class_names.append(class_names)

            for i, c in enumerate(class_names):
                class_dir = join(base_directory, c)
                if isdir(class_dir):
                    for f in listdir(class_dir):
                        file_dir = join(class_dir, f)
                        if isfile(file_dir) and file_dir.lower().endswith((".jpg", ".png")):
                            self.files.append(file_dir)
                            self.labels.append(i)

        self.nb_classes = len(self.class_names[0]) if self.class_names else 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = cv2.imread(self.files[index])
        img = cv2.resize(img, self.target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)

        if self.augment:
            # Get augmentation params from config, with defaults
            theta = self.aug_config.get('aug_rotation', 20)
            tx    = self.aug_config.get('aug_tx',       10)
            ty    = self.aug_config.get('aug_ty',       10)
            scale = self.aug_config.get('aug_zoom',      1.0)
            flip  = self.aug_config.get('aug_flip',    False)
            img = self.cv2_image_augmentation(
                img,
                theta=theta,
                tx=tx,
                ty=ty,
                scale=scale,
                flip=flip
            )

        if self.preprocess:
            img = self.preprocess(img)

        # If preprocess didn't convert to tensor, convert now
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose(2, 0, 1))

        label = torch.tensor(self.labels[index], dtype=torch.long)
        return img, label

    @staticmethod
    def cv2_image_augmentation(img, theta=20, tx=10, ty=10, scale=1.0, flip=False):
        # Random scale
        if scale != 1.0:
            scale = np.random.uniform(1 - scale, 1 + scale)
        # Random rotation
        if theta != 0:
            theta = np.random.uniform(-theta, theta)
        # Compute affine matrix
        m_inv = cv2.getRotationMatrix2D(
            (img.shape[1] // 2, img.shape[0] // 2),
            theta,
            scale
        )
        # Random translation
        if tx != 0 or ty != 0:
            tx_val = np.random.uniform(-tx, tx)
            ty_val = np.random.uniform(-ty, ty)
            m_inv[0, 2] += tx_val
            m_inv[1, 2] += ty_val
        # Warp image
        image = cv2.warpAffine(
            img,
            m_inv,
            (img.shape[1], img.shape[0]),
            borderMode=1
        )
        # Random horizontal flip
        if flip and random.random() > 0.5:
            image = np.fliplr(image)
        return image
