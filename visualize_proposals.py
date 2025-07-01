import argparse
import torch
from PIL import Image, ImageDraw
from model import AIModel
from config import data_transforms


def draw_boxes(img, boxes, out_file):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(box, outline='red', width=2)
    img.save(out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='path to image')
    parser.add_argument('--weights', default='model/model.pth')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--topk', type=int, default=5, help='number of proposals to visualize')
    args = parser.parse_args()

    device = torch.device(args.device)
    model = AIModel('efficientnet-b2', num_classes=400).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    img = Image.open(args.image).convert('RGB')
    transform = data_transforms['test']
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        _, boxes = model(tensor, return_boxes=True)

    boxes = boxes[0][:args.topk].cpu().numpy()
    # scale boxes to original image size
    feat_h, feat_w = model.backbone.extract_features(tensor).shape[-2:]
    scale_x = img.width / feat_w
    scale_y = img.height / feat_h
    scaled_boxes = []
    for x1, y1, x2, y2 in boxes:
        scaled_boxes.append((x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y))

    draw_boxes(img, scaled_boxes, 'proposals.png')

if __name__ == '__main__':
    main()
