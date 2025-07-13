import torch
from torchvision import transforms

# --- 数据和训练超参 ---
DATA_ROOT    = '../data'
NUM_CLASSES  = 400
IMG_SIZE     = 224
BATCH_SIZE   = 32
LR           = 0.01
MOMENTUM     = 0.9
WEIGHT_DECAY = 1e-4
EPOCHS       = 5
TOP_K        = 20       # 候选集扩展大小
TAU          = 0.6     # 开放集剔除阈值

# --- 图像变换 ---
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])
