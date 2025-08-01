# visualize_attention.py

import os
import torch
import numpy as np
from PIL import Image
from matplotlib import cm
from torchvision.models.efficientnet import EfficientNet_B0_Weights
from config import Config
from models.model1 import BaseModel

def visualize_attention(
    model_path: str = "trained_model/small_1/best_model1.pth",
    vis_dir: str    = "data/vis",
    out_dir: str    = "trained_model/small_1/vis_out"
):
    """
    加载第一阶段模型，对 vis_dir 下所有图片：
      1. 钩取 backbone.features[-1] 的输出特征图 (B,C,H,W)
      2. 在通道维度求平均得到 (H,W) 的热图
      3. 上采样并伪彩色叠加到原图
    结果保存在 out_dir。
    """
    device = Config.DEVICE
    os.makedirs(out_dir, exist_ok=True)

    # —— 1) 加载模型并注册特征钩子 ——
    model = BaseModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    feat_maps = []
    def feat_hook(module, inp, out):
        # out: [B, C, h, w]
        feat_maps.append(out.detach())
    # 主干最后一个特征块（MBConv）的输出
    model.backbone.features[-1].register_forward_hook(feat_hook)

    # —— 2) 预处理 ——
    weights   = EfficientNet_B0_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    # —— 3) 遍历图片并可视化 ——
    for fname in os.listdir(vis_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # 3.1 读图 & 预处理
        img = Image.open(os.path.join(vis_dir, fname)).convert("RGB")
        x   = transform(img).unsqueeze(0).to(device)

        # 清空缓存 & 推理
        feat_maps.clear()
        with torch.no_grad():
            _ = model(x)

        # 3.2 提取特征图 & 生成热图
        # feat_maps[0]: [1,C,h,w] -> squeeze batch -> [C,h,w]
        feat = feat_maps[0].squeeze(0).cpu().numpy()
        # 在通道维度求平均 -> [h,w]
        att_map = feat.mean(axis=0)
        # 归一化到 [0,1]
        att_map = att_map - att_map.min()
        if att_map.max() > 0:
            att_map = att_map / att_map.max()

        # 3.3 转为 8-bit 灰度图 & resize
        att_uint8 = (att_map * 255).astype(np.uint8)
        att_img   = Image.fromarray(att_uint8, mode='L')
        att_img   = att_img.resize(img.size, Image.BILINEAR)
        att_res   = np.array(att_img) / 255.0  # [H,W]

        # 3.4 伪彩色叠加
        heatmap = cm.jet(att_res)[..., :3]      # [H,W,3]
        orig    = np.array(img).astype(np.float32) / 255.0
        overlay = (orig * 0.5 + heatmap * 0.5)
        overlay = (overlay * 255).astype(np.uint8)

        # 3.5 保存
        out_path = os.path.join(out_dir, fname)
        Image.fromarray(overlay).save(out_path)
        print(f"[✓] Saved visualization: {out_path}")

if __name__ == "__main__":
    visualize_attention()
