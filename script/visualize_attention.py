import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from models import CBAMResNet

def compute_overlay(model, img_path, transform, device):
    img = Image.open(img_path).convert('RGB')
    inp = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        _ = model(inp)
        attn = model.cbam.sa_weight[0, 0].detach().cpu().numpy()

    attn = cv2.resize(attn, img.size)
    attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(attn_norm * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (np.array(img) * 0.5 + heatmap * 0.5).astype(np.uint8)
    return overlay


def visualize(model_path, image_folder, backbone='resnet50', nb_classes=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CBAMResNet(backbone=backbone, nb_classes=nb_classes)
    state_dict = torch.load(model_path, map_location=device)

    # Ensure checkpoint was trained with CBAM or attention maps will be random
    if not any(k.startswith("cbam") for k in state_dict.keys()):
        raise ValueError(
            f"Checkpoint '{model_path}' does not contain CBAM weights. "
            "Load a model trained with CBAM to visualize attention correctly."
        )

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if os.path.isdir(image_folder):
        files = sorted([f for f in os.listdir(image_folder)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        image_paths = [os.path.join(image_folder, f) for f in files]
    else:
        image_paths = [image_folder]

    idx = 0
    total = len(image_paths)
    window_name = 'CBAM Attention'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("浏览键: n/右箭头 = 下一张, p/左箭头 = 上一张, Esc = 退出")
    while True:
        path = image_paths[idx]
        overlay = compute_overlay(model, path, transform, device)

        cv2.imshow(window_name, overlay)
        print(f"显示 ({idx+1}/{total}): {path}")
        key = cv2.waitKey(0)
        # 退出
        if key == 27:
            break
        # 下一张: 'n' 或 83(右箭头)
        elif key & 0xFF == ord('n') or key == 83:
            idx = (idx + 1) % total
        # 上一张: 'p' 或 81(左箭头)
        elif key & 0xFF == ord('p') or key == 81:
            idx = (idx - 1) % total
        else:
            print(f"按键 {key} 未绑定，使用 n/p 或左右箭头")

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='浏览 CBAM 注意力热图')
    parser.add_argument('--model_path', default='best_checkpoints/best_epoch186_acc0.8717.pth',
                        help='CBAMResNet 检查点路径')
    parser.add_argument('--image_folder', default='../data/small/train/000',
                        help='单张图片路径或图片文件夹')
    parser.add_argument('--backbone', default='resnet50', help='ResNet 骨干网络')
    parser.add_argument('--nb_classes', type=int, default=400,
                        help='模型类别数（与 checkpoint 一致）')
    args = parser.parse_args()

    visualize(args.model_path, args.image_folder,
              backbone=args.backbone, nb_classes=args.nb_classes)

if __name__ == '__main__':
    main()
