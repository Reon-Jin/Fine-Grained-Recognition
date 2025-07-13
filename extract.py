import os
import cv2
import numpy as np
import torch
from torchvision import transforms, models
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

# =============================================
# 脚本：批量从 train_filter 提取前景并保存到 train_extract
# 保持原有子文件夹结构不变（400 个类别目录）
# =============================================

# ========== 参数配置 ==========
SRC_DIR = "data/WebFG-400/train_filter/369"    # 源数据目录
DST_DIR = "data/WebFG-400/train_extract/369"   # 输出目录
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCORE_THRESHOLD = 0.3 # Mask R-CNN 实例置信度阈值

# ========== 加载 Mask R-CNN ==========
mask_weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = models.detection.maskrcnn_resnet50_fpn(weights=mask_weights)
model.to(DEVICE).eval()

to_tensor = transforms.Compose([
    transforms.ToTensor(),
])

# ========== 前景提取函数 ==========
def extract_foreground(image_np: np.ndarray) -> np.ndarray:
    """
    使用 Mask R-CNN 提取前景，将背景设为黑色。
    输入: HxWx3 RGB numpy array
    输出: HxWx3 numpy array
    """
    # Resize 到IMG_SIZE
    img = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE))
    tensor = to_tensor(img).to(DEVICE)
    with torch.no_grad():
        output = model([tensor])[0]
    masks = output['masks'][:, 0]  # [N, H, W]
    scores = output['scores']
    keep = scores > SCORE_THRESHOLD
    if keep.sum().item() == 0:
        return np.zeros_like(img)
    masks = masks[keep]
    # 合并所有实例 mask
    mask_bin = (masks > 0.5).any(dim=0).cpu().numpy().astype(np.uint8)
    # 应用掩码
    fg = img * mask_bin[:, :, None]
    return fg

# ========== 批量处理函数 ==========
def process_dataset(src_dir: str, dst_dir: str):
    """
    遍历 src_dir 下所有子文件夹，提取前景并保存到 dst_dir，
    保持子文件夹结构一致。
    """
    for root, dirs, files in os.walk(src_dir):
        # 计算相对路径
        rel_path = os.path.relpath(root, src_dir)
        # 对应输出路径
        out_folder = os.path.join(dst_dir, rel_path)
        # 保持子文件夹结构
        os.makedirs(out_folder, exist_ok=True)
        for fname in files:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            src_path = os.path.join(root, fname)
            dst_path = os.path.join(out_folder, fname)
            # 读取BGR->RGB
            img_bgr = cv2.imread(src_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # 前景提取
            fg = extract_foreground(img_rgb)
            # 保存前景图 (RGB->BGR)
            fg_bgr = cv2.cvtColor(fg, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dst_path, fg_bgr)

if __name__ == "__main__":
    print(f"Extracting foreground from '{SRC_DIR}' into '{DST_DIR}'...")
    process_dataset(SRC_DIR, DST_DIR)
    print("Done. Output directory structure matches source.")