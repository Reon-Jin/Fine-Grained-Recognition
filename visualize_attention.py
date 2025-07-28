# visualize_attention.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from config import Config
from models.model2 import TransFGDeiT

def visualize_attention_patches(model: TransFGDeiT, img_paths, device):
    """
    对每张图片：
      1) 前向一次，得到 attention scores: [1, num_patches]
      2) reshape → [grid, grid]
      3) 以 heatmap 叠加在原图上
    """
    cfg = Config()
    model.eval()

    # 与训练时相同的预处理
    preprocess = transforms.Compose([
        transforms.Resize((cfg.INPUT_SIZE, cfg.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.NORMALIZE_MEAN,
                             std=cfg.NORMALIZE_STD)
    ])

    for path in img_paths:
        # 读取与预处理
        img = Image.open(path).convert("RGB")
        inp = preprocess(img).unsqueeze(0).to(device)  # [1,3,224,224]

        # 前向得到 logits + aux dict
        with torch.no_grad():
            _, aux = model(inp)

        # 取出 attention scores [1, N]
        attn = aux['attn_scores'][0].cpu().numpy()  # [N]
        N = attn.shape[0]
        grid = int(np.sqrt(N))                    # e.g. 14 for patch16 on 224

        # reshape 成 [grid, grid]
        attn_map = attn.reshape(grid, grid)

        # 归一化到 [0,1]
        attn_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # 可视化 heatmap
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(img.resize((cfg.INPUT_SIZE, cfg.INPUT_SIZE)))
        ax.imshow(attn_norm, cmap='jet', alpha=0.5,
                  extent=(0, cfg.INPUT_SIZE, cfg.INPUT_SIZE, 0))
        ax.set_axis_off()

        # 保存
        out_dir = os.path.join(cfg.MODEL_DIR, "attention_patches")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(path))
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # 打印输出路径
        print(f"→ saved attention map to {out_path}")


if __name__ == "__main__":
    cfg = Config()
    device = cfg.DEVICE

    # 加载模型
    from data_prepare import prepare_dataloaders
    _, _, num_classes = prepare_dataloaders(cfg)
    model = TransFGDeiT(num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg.MODEL_DIR, "best_model.pth")))

    # 准备要可视化的图片列表
    img_dir = cfg.ATTN_IMAGE_DIR
    assert os.path.isdir(img_dir), f"指定的可视化图像目录不存在: {img_dir}"

    img_paths = [
        os.path.join(img_dir, fn)
        for fn in sorted(os.listdir(img_dir))
        if fn.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    assert len(img_paths) > 0, f"目录中没有找到图片: {img_dir}"

    visualize_attention_patches(model, img_paths, device)
