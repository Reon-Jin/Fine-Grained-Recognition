import argparse
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from models import CBAMResNet


def visualize(model_path, image_path, backbone='resnet50', nb_classes=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CBAMResNet(backbone=backbone, nb_classes=nb_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    inp = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        _ = model(inp)
        attn = model.cbam.sa_weight[0, 0].detach().cpu().numpy()

    attn = cv2.resize(attn, img.size)
    attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(attn_norm * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (np.array(img) * 0.5 + heatmap * 0.5).astype(np.uint8)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Input')

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.axis('off')
    plt.title('CBAM Attention')
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize CBAM attention')
    parser.add_argument('model_path', help='Path to trained model (.pth)')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--backbone', default='resnet50', help='ResNet backbone')
    parser.add_argument('--nb_classes', type=int, default=1000,
                        help='Number of classes for model')
    args = parser.parse_args()
    visualize(args.model_path, args.image_path,
              backbone=args.backbone, nb_classes=args.nb_classes)


if __name__ == '__main__':
    main()
