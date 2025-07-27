import torch


class Config:
    DATA_DIR = 'data/WebFG-400/train'
    INPUT_SIZE = 224
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 25
    PATIENCE = 2  # 连续多少个 epoch 没提升就降低 lr
    LR_DECAY_FACTOR = 0.5  # 学习率衰减比例
    MIN_LR = 1e-6  # 最低学习率
    NUM_WORKERS = 4
    RANDOM_SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_DIR = 'trained_model/web400_2'
    WARMUP_EPOCHS = 5
    LOCAL_CROP_K = 1
    LOCAL_CROP_RATIO = 0.5

    VISUALIZE_ATTENTION = True
    # 如果指定了该目录，就对目录下**所有**图片做可视化；否则才随机选
    ATTN_IMAGE_DIR = "data/vis"
    ATTN_VIS_SAMPLES = 3  # 只有当 ATTN_IMAGE_DIR 为空时才生效
    # —— 归一化参数等也要在这里 —— #
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

    AUX_WEIGHT = 0.3  # 辅助分类 head 的损失权重
    ENTROPY_WEIGHT = 0.01  # 注意力块权重分布的熵正则系数
    # 注意力分布的温度参数，越小分布越尖锐
    ATTN_TEMPERATURE = 1.0

    GRID_SIZE = 3
    # 注意力头数
    BLOCK_HEADS = 3