import cv2
import numpy as np
import torch
from torchvision import transforms, models
import matplotlib.pyplot as plt

# === 参数设置 ===
IMAGE_PATH = "../data/WebFG-400/train/001/1d40feb699584b2faf2c6e9e2fcef9cc.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCORE_THRESHOLD = 0.5  # 只保留置信度高于此值的实例

# === 1. 读取并resize图像 ===
orig = cv2.imread(IMAGE_PATH)
orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(orig_rgb, (224,224))

# === 2. 准备输入张量 ===
transform = transforms.Compose([
    transforms.ToTensor(),  # uint8 HWC -> float [0,1] CHW
])
input_tensor = transform(img_resized).to(DEVICE)

# === 3. 加载预训练 Mask R-CNN 并预测 ===
model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.to(DEVICE).eval()

with torch.no_grad():
    outputs = model([input_tensor])
output = outputs[0]

# === 4. 合并所有高置信度实例的 mask ===
masks = output['masks']      # [N, 1, H, W], float mask logits
scores = output['scores']    # [N]
H, W = img_resized.shape[:2]

# 取出置信度足够高的那些实例
keep = scores > SCORE_THRESHOLD
if keep.sum() == 0:
    raise RuntimeError("No instance with score > {:.2f}".format(SCORE_THRESHOLD))

masks = masks[keep, 0]       # [K, H, W]
# 二值化并合并：任何实例中如果 mask>0.5 就算前景
mask_binary = (masks > 0.5).any(dim=0).cpu().numpy().astype(np.uint8)  # [H, W], 0或1

# === 5. 生成前背景分离图 ===
# 前景
foreground = img_resized * mask_binary[:, :, None]
# 背景（反掩码）
background = img_resized * (1 - mask_binary)[:, :, None]

# === 6. 可视化 ===
fig, axs = plt.subplots(1, 4, figsize=(20,5))

axs[0].imshow(img_resized)
axs[0].set_title("Original")
axs[0].axis('off')

axs[1].imshow(mask_binary, cmap='gray')
axs[1].set_title("Combined Mask")
axs[1].axis('off')

axs[2].imshow(foreground)
axs[2].set_title("Extracted Foreground")
axs[2].axis('off')

axs[3].imshow(background)
axs[3].set_title("Extracted Background")
axs[3].axis('off')

plt.tight_layout()
plt.show()
