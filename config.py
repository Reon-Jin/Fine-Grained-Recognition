"""Project‑wide configuration file."""

import os

# ========= Paths =========
DATA_DIR   = os.getenv('DATA_DIR', './data')  # raw data root containing sub‑folders per class
TRAIN_DIR  = os.path.join(DATA_DIR, 'train')
VAL_DIR    = os.path.join(DATA_DIR, 'val')
RUNS_DIR   = './runs'                      # logs / checkpoints

# ========= Dataset & Loader =========
IMAGE_SIZE   = 224
BATCH_SIZE   = 32
NUM_WORKERS  = 4

# ========= Model & Training =========
NUM_CLASSES  = 100           # ⚠️  update after inspecting dataset
EPOCHS       = 100
LR           = 1e-3
WEIGHT_DECAY = 1e-4
TOP_K        = 3             # for first‑stage evaluation

# ========= Co‑Teaching =========
USE_CO_TEACHING = True
NOISE_RATE      = 0.4          # estimated noisy label proportion
FORGET_END      = 0.6          # ramp‑up end point as fraction of epochs

# ========= Attention / ROI =========
ATTENTION_THRESHOLD = 0.6      # keep top‑x% activation

# ========= Misc =========
SEED = 42
