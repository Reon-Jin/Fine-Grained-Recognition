import argparse
import os
import torch
from PIL import Image, ImageDraw
from model import AIModel
from config import data_transforms

def draw_boxes(img, boxes, out_path):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        draw.rectangle((x1, y1, x2, y2), outline='red', width=2)
    img.save(out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data/WebFG-400/train/000', help='input image folder')
    parser.add_argument('--output_dir', default='data/proposals', help='output folder')
    parser.add_argument('--weights', default='model/model.pth')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--topk', type=int, default=5, help='number of proposals to visualize')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    model = AIModel('efficientnet-b2', num_classes=400).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    transform = data_transforms['test']

    for fname in os.listdir(args.input_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        img_path = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[跳过] 无法读取图像 {fname}: {e}")
            continue

        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            _, boxes = model(tensor, return_boxes=True)

        boxes = boxes[0][:args.topk].cpu().numpy()

        # 获取特征图尺寸以便缩放 proposal 回原图尺寸
        feat_h, feat_w = model.backbone.extract_features(tensor).shape[-2:]
        scale_x = img.width / feat_w
        scale_y = img.height / feat_h

        scaled_boxes = []
        for x1, y1, x2, y2 in boxes:
            scaled_boxes.append((x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y))

        draw_boxes(img, scaled_boxes, out_path)
        print(f"[保存] {out_path}")

if __name__ == '__main__':
    main()
