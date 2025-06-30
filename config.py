# config.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# 统一图像尺寸
imgsz = 384

# 数据预处理管道
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# 训练/验证集自动读取子目录作为类别
def get_train_dataset(root_dir):
    # root_dir/train/下每个子文件夹即一个类别
    return ImageFolder(os.path.join(root_dir, 'train'), transform=data_transforms['train'])

def get_val_dataset(root_dir):
    """Validation set shares the same structure as ``train``."""
    return ImageFolder(os.path.join(root_dir, 'val'), transform=data_transforms['test'])

# 测试集：无标签，仅返回图像和文件名
class TestDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root = root_dir
        self.names = sorted(os.listdir(root_dir))
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        fname = self.names[idx]
        path = os.path.join(self.root, fname)
        img = Image.open(path).convert('RGB')
        # return the file name with extension for saving results
        return self.transform(img), fname
