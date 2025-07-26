import torch


class Config:
    DATA_DIR = 'data'
    INPUT_SIZE = 224
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 10
    NUM_WORKERS = 4
    RANDOM_SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_DIR = 'trained_model'
