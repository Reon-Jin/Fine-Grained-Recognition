# visualize_attention_blocks.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from config import Config
from data_prepare import prepare_dataloaders
from models.model import FineGrainedModel

def compute_block_attention(sa_map, grid_size=4):
    """
    sa_map: torch.Tensor of shape [1,1,Hf,Wf], values in [0,1]
    返回 shape [grid_size, grid_size] 的 numpy 注意力分数矩阵
    """
    _, _, Hf, Wf = sa_map.shape
    gh, gw = Hf // grid_size, Wf // grid_size
    blocks = np.zeros((grid_size, grid_size), dtype=float)

    sa = sa_map[0,0].cpu().numpy()
    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i*gh, (i+1)*gh
            x1, x2 = j*gw, (j+1)*gw
            blocks[i, j] = sa[y1:y2, x1:x2].mean()
    return blocks

def visualize_blocks(model, img_paths, device):
    cfg = Config()
    model.eval()

    # 与训练一致的 transform
    val_tf = transforms.Compose([
        transforms.Resize((cfg.INPUT_SIZE, cfg.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.NORMALIZE_MEAN, cfg.NORMALIZE_STD)
    ])

    # 钩在 global_cbam 生成空间注意力上
    def get_sa_map(x):
        # x: [1,3,H,W]
        with torch.no_grad():
            feat = model.base.features(x)                      # [1,1280,hf,wf]
            feat = model.global_cbam(feat)                     # 通道+空间
            # 用 SpatialGate 拿出 raw spatial conv 结果
            comp = model.global_cbam.SpatialGate.compress(feat)
            spat = model.global_cbam.SpatialGate.spatial(comp)
            sa = torch.sigmoid(spat)                           # [1,1,hf,wf]
        return sa

    for path in img_paths:
        img = Image.open(path).convert("RGB")
        inp = val_tf(img).unsqueeze(0).to(device)

        sa_map = get_sa_map(inp)  # [1,1,hf,wf]
        blocks = compute_block_attention(sa_map, grid_size=4)

        # 打印 4x4 注意力矩阵
        print(f"\n{os.path.basename(path)} block attention:")
        for row in blocks:
            print("  ".join(f"{v:.3f}" for v in row))

        # 可视化：在原图上叠加 4x4 色块
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(img.resize((cfg.INPUT_SIZE, cfg.INPUT_SIZE)))
        H, W = cfg.INPUT_SIZE, cfg.INPUT_SIZE
        gh, gw = H/4, W/4

        # 归一化 blocks 到 [0,1]
        norm = (blocks - blocks.min())/(blocks.max()-blocks.min()+1e-8)

        # 画色块
        for i in range(4):
            for j in range(4):
                alpha = norm[i,j] * 0.8  # 调节透明度
                rect = plt.Rectangle((j*gw, i*gh), gw, gh,
                                     color='red', alpha=alpha)
                ax.add_patch(rect)
        ax.set_axis_off()
        plt.tight_layout()
        out_dir = os.path.join(cfg.MODEL_DIR, "attention_blocks")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(path))
        fig.savefig(out_path)
        plt.close(fig)
        print(f"→ saved visualization to {out_path}")

if __name__ == "__main__":
    cfg = Config()
    device = cfg.DEVICE

    # 加载模型
    model = FineGrainedModel(num_classes=len(prepare_dataloaders(cfg)[2]))
    model.load_state_dict(torch.load(os.path.join(cfg.MODEL_DIR, "best_model.pth")))
    model.to(device)

    # 获取待可视化的图片列表
    img_dir = cfg.ATTN_IMAGE_DIR
    if img_dir and os.path.isdir(img_dir):
        img_paths = [os.path.join(img_dir, fn)
                     for fn in sorted(os.listdir(img_dir))
                     if fn.lower().endswith(('.jpg','.jpeg','.png'))]
    else:
        # fallback: 随机从训练集中取 5 张
        _, train_loader, _ = prepare_dataloaders(cfg)
        subset = train_loader.dataset
        indices = np.random.choice(len(subset), size=5, replace=False)
        img_paths = [subset.dataset.imgs[subset.indices[i]][0] for i in indices]

    visualize_blocks(model, img_paths, device)
