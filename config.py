import torch

class Config:
    # ——— 基本训练配置 ———
    DATA_DIR        = 'data/mini'
    INPUT_SIZE      = 224
    BATCH_SIZE      = 32
    LEARNING_RATE   = 2e-3
    EPOCHS          = 25
    T_EPOCHS        = 25
    WARMUP_EPOCHS   = 5
    MIN_LR          = 1e-5
    WEIGHT_DECAY    = 1e-5
    NUM_WORKERS     = 4
    RANDOM_SEED     = 42
    DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_DIR       = 'trained_model/mini_14'

    # ——— 可视化 & 归一化 ———
    VISUALIZE_ATTENTION = True
    ATTN_IMAGE_DIR      = "data/vis"
    ATTN_VIS_SAMPLES    = 3
    NORMALIZE_MEAN      = [0.485, 0.456, 0.406]
    NORMALIZE_STD       = [0.229, 0.224, 0.225]

    # ——— 数据增强 ———
    USE_RANDAUGMENT = True
    RANDAUG_N       = 3
    RANDAUG_M       = 12

    # ——— TransFG 部件选择 ———
    NUM_PARTS    = 4      # 选 top-k patches
    PART_DROPOUT = 0.1    # 部件特征融合后的 dropout
