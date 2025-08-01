# config.py

import torch

class Config:
    # —— 数据相关 —— 
    DATA_DIR    = 'data/small/train'   # 你的数据根目录，按类别子文件夹组织
    VAL_SPLIT   = 0.2           # 验证集比例
    SEED        = 42            # 随机种子，保证可复现

    # —— 模型相关 —— 
    NUM_CLASSES = 11             # 类别数
    K           = 2             # Top-K 值

    # —— 训练超参 —— 
    BATCH_SIZE  = 32
    LR          = 1e-3
    EPOCHS      = 20
    NUM_WORKERS = 4             # DataLoader 并行进程数

    # —— 设备 —— 
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
