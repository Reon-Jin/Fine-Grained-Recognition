# visualize_pipeline_full.py
import os
import cv2
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.cuda.amp import autocast
from torch.nn import functional as F
from opt_dg_tf2_new import DirectoryDataset
from models import construct_model
from utils import getROIS

# ----------------------------- 配置 -----------------------------
with open("../config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

ROI_RES   = cfg['DATA']['image_size'][0]         # 224
GRID_SIZE = 3                                   # 3×3 网格
MIN_SIZE  = 2                                   # 最小跨度 2 个格子单元
POOL_SIZE = 7                                   # Main 中也是 7×7 Pool
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME= cfg['MODEL']['model_name']
CHECKPT   = "best_checkpoints/best_epoch159_acc0.7273.pth"      # 换成你 main.py 保存的 checkpoint

# -------------------------- 加载模型 ----------------------------
model = construct_model(
    name=MODEL_NAME,
    pool_size=POOL_SIZE,
    ROIS_resolution=ROI_RES,
    ROIS_grid_size=GRID_SIZE,
    minSize=MIN_SIZE,
    nb_classes=cfg['DATA']['nb_classes']
)
state = torch.load(CHECKPT, map_location='cpu')
# 如果你确定 checkpoint 时就是相同的 3×3+minSize=2，可以直接 strict=True
model.load_state_dict(state, strict=True)
model.to(DEVICE).eval()

# hook SelfAttention 以收集 attention
attention_maps = []
def hook_fn(module, inp, out):
    attention_maps.append(module.attention.detach().cpu().numpy())

for m in model.modules():
    if m.__class__.__name__ == "SelfAttention":
        m.register_forward_hook(hook_fn)

# ------------------------ 图片预处理 --------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------- 主可视化函数 ------------------------
def visualize_one(img_path, ax_row):
    # 读取 + resize 到 224×224
    img = cv2.imread(img_path)[:, :, ::-1]
    img = cv2.resize(img, (ROI_RES, ROI_RES))

    # 获取 3×3 网格上所有宽/高 ≥ 2 单元的 ROIs
    rois = getROIS(resolution=ROI_RES, gridSize=GRID_SIZE, minSize=MIN_SIZE)

    # (1) 画 ROI 框
    ax = ax_row[0]
    ax.imshow(img)
    for idx, (x, y, w, h) in enumerate(rois):
        r = plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none', lw=1)
        ax.add_patch(r)
        ax.text(x+2, y+12, str(idx), color='r', fontsize=8)
    ax.axis('off'); ax.set_title("ROIs (3×3, minSize=2)")

    # (2) 前向并收集 attention
    attention_maps.clear()
    inp = transform(img).unsqueeze(0).to(DEVICE)
    with autocast():
        out = model(inp)

    # (3) 聚合每个 ROI 的平均 attention
    roi_weights = [ beta[0].mean() for beta in attention_maps ]
    roi_weights = np.array(roi_weights)

    # (4) 热力覆盖
    ax = ax_row[1]
    overlay = img.astype(float) / 255
    norm_w  = (roi_weights - roi_weights.min()) / (roi_weights.ptp() + 1e-6)
    for idx, (x, y, w, h) in enumerate(rois):
        c = plt.cm.Reds(norm_w[idx])[:3]
        patch = overlay[y:y+h, x:x+w]
        overlay[y:y+h, x:x+w] = 0.6*patch + 0.4*np.array(c)
    ax.imshow(overlay); ax.axis('off'); ax.set_title("ROI Attention")

    # (5) 条形图 + Top‑3 预测
    ax = ax_row[2]
    ax.bar(range(len(rois)), roi_weights, color='gray')
    ax.set_title("Avg ROI Attn + Preds")
    ax.set_xlabel("ROI #"); ax.set_ylabel("Weight")

    probs = F.softmax(out, dim=1).detach().cpu().numpy()[0]
    top3  = probs.argsort()[-3:][::-1]
    for i, cls in enumerate(top3):
        ax.text(1.02, 0.9 - 0.1*i,
                f"{cls}: {probs[cls]:.2f}",
                transform=ax.transAxes,
                ha='left', va='top')

# ------------------------- 批量处理 ---------------------------
def batch_visualize(img_list, out_png="pipeline_vis_full.png", cols=2):
    n    = len(img_list)
    rows = (n + cols - 1)//cols
    fig, axes = plt.subplots(rows*3, cols, figsize=(cols*4, rows*6))
    axes = axes.reshape(rows, 3, cols)

    for i, img_path in enumerate(img_list):
        r, c = divmod(i, cols)
        visualize_one(img_path, axes[r,:,c])

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    print("Saved visualization to", out_png)

if __name__ == "__main__":
    # 只取 train/000 目录下前 10 张
    folder = os.path.join(cfg['DATA']['dataset_dir'], "train", "000")
    imgs   = sorted(os.listdir(folder))[:10]
    img_paths = [os.path.join(folder, fn) for fn in imgs]
    batch_visualize(img_paths, out_png="pipeline_vis_full.png", cols=2)
