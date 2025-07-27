# visualize_attention.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from config import Config
from models.model import FineGrainedModel

def visualize_blocks(model: FineGrainedModel, img_paths, device):
    """
    对每张图片：
      1) 前向一次，得到 wts: [1, heads, num_blocks]
      2) 对 heads 维度取平均 → [1, num_blocks]
      3) reshape → [grid_size, grid_size]
      4) 打印 & 叠加可视化
    """
    cfg = Config()
    model.eval()

    # 和训练时一模一样的预处理
    val_tf = transforms.Compose([
        transforms.Resize((cfg.INPUT_SIZE, cfg.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.NORMALIZE_MEAN, cfg.NORMALIZE_STD)
    ])

    for path in img_paths:
        img = Image.open(path).convert("RGB")
        inp = val_tf(img).unsqueeze(0).to(device)  # [1,3,H,W]

        # 前向得到三项输出，其中 wts 形状是 [1, heads, num_blocks]
        with torch.no_grad():
            _, _, wts = model(inp)

        # 平均各头，得到 [1, num_blocks]
        blk = wts.mean(dim=1).view(-1)  # [num_blocks]
        # reshape 成 grid×grid
        grid_size = cfg.GRID_SIZE
        blocks = blk.cpu().numpy().reshape(grid_size, grid_size)

        # 打印数值
        print(f"\n{os.path.basename(path)} block attention:")
        for row in blocks:
            print("  ".join(f"{v:.3f}" for v in row))

        # 叠加可视化
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(img.resize((cfg.INPUT_SIZE, cfg.INPUT_SIZE)))
        H, W = cfg.INPUT_SIZE, cfg.INPUT_SIZE
        gh, gw = H / grid_size, W / grid_size

        # 归一化到 [0,1]
        norm = (blocks - blocks.min()) / (blocks.max() - blocks.min() + 1e-8)

        for i in range(grid_size):
            for j in range(grid_size):
                alpha = norm[i,j] * 0.7  # 最大透明度 0.7
                rect = plt.Rectangle(
                    (j*gw, i*gh), gw, gh,
                    color='red', alpha=alpha
                )
                ax.add_patch(rect)

        ax.set_axis_off()
        out_dir = os.path.join(cfg.MODEL_DIR, "attention_blocks")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(path))
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"→ saved to {out_path}")

if __name__ == "__main__":
    cfg = Config()
    device = cfg.DEVICE

    # 加载模型
    _, _, num_classes = None, None, None
    # 如果你有 prepare_dataloaders 返回 num_classes，可以这样：
    from data_prepare import prepare_dataloaders
    _, _, num_classes = prepare_dataloaders(cfg)
    model = FineGrainedModel(num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg.MODEL_DIR, "best_model.pth")))

    # 准备要可视化的图片列表
    img_dir = cfg.ATTN_IMAGE_DIR
    if img_dir and os.path.isdir(img_dir):
        img_paths = [
            os.path.join(img_dir, fn)
            for fn in sorted(os.listdir(img_dir))
            if fn.lower().endswith(('.jpg','.jpeg','.png'))
        ]
    else:
        # 随机抽几张训练集
        _, train_loader, _ = prepare_dataloaders(cfg)
        subset = train_loader.dataset
        idxs   = np.random.choice(len(subset), size=cfg.ATTN_VIS_SAMPLES, replace=False)
        img_paths = [subset.dataset.imgs[subset.indices[i]][0] for i in idxs]

    visualize_blocks(model, img_paths, device)
