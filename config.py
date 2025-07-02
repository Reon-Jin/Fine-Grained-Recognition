from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
imgsz = 224
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}


class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_paths = []
        for img_name in os.listdir(test_dir):
            img_path = os.path.join(test_dir, img_name)
            self.image_paths.append(img_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        id = os.path.splitext(os.path.basename(img_path))[0]
        return image, id


class MultiClassDataset(Dataset):
    """Generic dataset for multi-class classification."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [
            d
            for d in sorted(os.listdir(root_dir))
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        self.image_paths = []
        self.labels_list = []
        for idx, class_name in enumerate(self.classes):
            folder_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                self.image_paths.append(img_path)
                self.labels_list.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels_list[idx]
        return image, label

