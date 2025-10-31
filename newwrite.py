import os
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
import numpy as np
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from betterPNP import EfficientnetV2L

if not hasattr(np, "int"):
    np.int = int

from utils.utils import set_device, init_seeds
from utils.module import MLPHead

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ========= 配置 =========
INPUT_DIR = r"C:\Fine-Grained-Recognition\Datasets\test"
CKPT_PATH = r"C:\Fine-Grained-Recognition\Results\web-400\Better_PNP_efficientnet_v2L\webfg400\3heads-ECA-3407-20251030_131801\epoch_48.pth"
NUM_CLASSES = 400
RESCALE_SIZE = 480
CROP_SIZE = 480
BATCH_SIZE = 16
GPU = "0"
OUTPUT = None
USE_AMP = True
TEMPERATURE = 0.8  # 温度缩放参数


# ========== 模型定义 ==========
def init_weights(m, init_method='He'):
    if isinstance(m, nn.Linear):
        if init_method.lower() in ['he', 'kaiming']:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        else:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ========== 数据集 ==========
class ImageFolderDataset(Dataset):
    def __init__(self, folder):
        self.paths = []
        folder = Path(folder)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            self.paths.extend(sorted(folder.glob(ext)))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return {'image': img, 'name': os.path.basename(str(path)), 'path': str(path)}


# ========== 权重加载 ==========
def load_state_dict_strict(model, ckpt_path, map_location):
    state = torch.load(ckpt_path, map_location=map_location)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            if k.startswith('module.'):
                new_state[k[len('module.'):]] = v
            else:
                new_state[k] = v
        state = new_state
    model.load_state_dict(state, strict=True)


# ========== 批量处理的高效增强 ==========
class BatchTestTransform:
    """批量处理的高效测试时增强"""

    def __init__(self, base_transform, num_augs=5):
        self.base_transform = base_transform
        self.num_augs = num_augs

    def apply_augs(self, batch_imgs):
        """对整批图片应用多种增强"""
        all_augmented = []

        # 增强1: 原图
        original_batch = torch.stack([self.base_transform(img) for img in batch_imgs])
        all_augmented.append(original_batch)

        # 增强2: 水平翻转
        flip_batch = torch.stack([self.base_transform(transforms.functional.hflip(img)) for img in batch_imgs])
        all_augmented.append(flip_batch)

        # 增强3: 轻微亮度增强
        bright_batch = torch.stack([
            self.base_transform(transforms.functional.adjust_brightness(img, 1.15))
            for img in batch_imgs
        ])
        all_augmented.append(bright_batch)

        # 增强4: 轻微对比度增强
        contrast_batch = torch.stack([
            self.base_transform(transforms.functional.adjust_contrast(img, 1.1))
            for img in batch_imgs
        ])
        all_augmented.append(contrast_batch)

        # 增强5: 多尺度 (0.95x)
        if self.num_augs >= 5:
            scale_batch = []
            for img in batch_imgs:
                new_size = int(CROP_SIZE * 0.95)
                img_resized = transforms.functional.resize(img, new_size)
                img_cropped = transforms.functional.center_crop(img_resized, CROP_SIZE)
                scale_batch.append(self.base_transform(img_cropped))
            all_augmented.append(torch.stack(scale_batch))

        return all_augmented


# ========== 高效的批量集成预测 ==========
def efficient_ensemble_predict(model, batch_imgs, batch_transform, device, amp_ctx):
    """高效的批量集成预测"""

    # 权重配置 (对应5种增强)
    weights = [2.0, 2.0, 1.2, 1.2, 1.5]

    # 应用所有增强到整批图片
    augmented_batches = batch_transform.apply_augs(batch_imgs)

    all_logits = []

    with amp_ctx:
        # 批量处理每种增强
        for aug_batch in augmented_batches:
            aug_batch = aug_batch.to(device, non_blocking=True)
            logits = model(aug_batch)['logits']
            all_logits.append(logits)

    # 加权集成
    weighted_logits = sum(w * (logits / TEMPERATURE) for w, logits in zip(weights, all_logits)) / sum(weights)

    # 计算预测和置信度
    probs = F.softmax(weighted_logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)

    return predictions.cpu().tolist(), confidences.cpu().tolist()


# ========== 优化的主流程 ==========
def run_optimized():
    init_seeds(0)
    device = set_device(GPU)

    # 基础变换
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=RESCALE_SIZE),
        torchvision.transforms.CenterCrop(size=CROP_SIZE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))
    ])

    # 批量增强处理器
    batch_transform = BatchTestTransform(test_transform, num_augs=5)
    print("使用5种高效测试时增强")

    dataset = ImageFolderDataset(INPUT_DIR)

    def collate_fn(batch):
        images = [item['image'] for item in batch]
        names = [item['name'] for item in batch]
        return {'images': images, 'names': names}

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # 增加workers加速数据加载
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    model = EfficientnetV2L(
        num_classes=NUM_CLASSES,
        pretrained=False
    ).to(device)

    if not Path(CKPT_PATH).exists():
        raise FileNotFoundError(f"未找到权重文件 {CKPT_PATH}")

    load_state_dict_strict(model, CKPT_PATH, map_location='cpu')
    model.eval()

    out_csv = Path(OUTPUT if OUTPUT else f"pred_results_web400.csv")
    out_csv_with_conf = Path(OUTPUT if OUTPUT else f"pred_results_web400_with_confidence.csv")

    if device.type == 'cuda' and USE_AMP:
        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
    else:
        from contextlib import nullcontext
        amp_ctx = nullcontext()

    total = len(dataset)
    processed = 0

    # 预热GPU
    print("预热GPU...")
    dummy_input = torch.randn(2, 3, CROP_SIZE, CROP_SIZE).to(device)
    with torch.no_grad(), amp_ctx:
        _ = model(dummy_input)

    # 写入结果
    with open(out_csv, "w", encoding="utf-8") as f, \
            open(out_csv_with_conf, "w", encoding="utf-8") as f_conf, \
            torch.inference_mode():

        f_conf.write("filename,prediction,confidence\n")

        pbar = tqdm(loader, ncols=100, ascii=' >', leave=False, desc='批量推理')
        start_time = time.time()

        for batch in pbar:
            batch_imgs = batch['images']
            batch_names = batch['names']

            # 高效的批量集成预测
            predictions, confidences = efficient_ensemble_predict(
                model, batch_imgs, batch_transform, device, amp_ctx
            )

            # 写入结果
            for name, pred, conf in zip(batch_names, predictions, confidences):
                f.write(f"{name},{pred:04d}\n")
                f_conf.write(f"{name},{pred:04d},{conf:.4f}\n")

            processed += len(batch_imgs)
            pbar.set_description(f'进度: {processed}/{total}')

        end_time = time.time()
        total_time = end_time - start_time
        speed = total / total_time

    print(f"[OK] 推理完成! 耗时: {total_time:.2f}秒, 速度: {speed:.2f} 图片/秒")
    print(f"     预测结果: {out_csv}")
    print(f"     带置信度: {out_csv_with_conf}")

    # 打印优化信息
    print(f"\n优化信息:")
    print(f"- 使用5种高效增强")
    print(f"- 温度缩放: {TEMPERATURE}")
    print(f"- 数据加载workers: 4")


if __name__ == "__main__":
    run_optimized()