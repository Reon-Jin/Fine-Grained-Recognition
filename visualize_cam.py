import argparse
import torch
from PIL import Image
from model import AIModel
from config import data_transforms
import numpy as np
import os
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, inp, out):
        self.activations = out.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()

        # 显式启用梯度追踪（即使在 no_grad 外层环境中）
        with torch.enable_grad():
            input_tensor.requires_grad = True
            output = self.model(input_tensor)

            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            loss = output[:, class_idx].sum()
            loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(1),
            size=input_tensor.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_cam(img, cam, out_path, threshold=0.6):
    if isinstance(img, Image.Image):
        img = np.array(img)

    h, w, _ = img.shape
    cam_resized = cv2.resize(cam, (w, h))
    cam_uint8 = np.uint8(255 * cam_resized)

    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    mask = cam_resized > threshold
    mask = mask.astype(np.uint8)
    mask = np.expand_dims(mask, axis=-1)

    overlay = img.astype(np.float32)
    heatmap = heatmap.astype(np.float32)
    overlay = overlay * (1 - mask * 0.5) + heatmap * (mask * 0.5)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    Image.fromarray(overlay).save(out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data/WebFG-400/train/000', help='input image folder')
    parser.add_argument('--output_dir', default='data/outputs_cam', help='output image folder')
    parser.add_argument('--weights', default='model/model.pth', help='model weights')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    model = AIModel('efficientnet-b2', num_classes=400).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    cam_extractor = GradCAM(model, model.cbam)

    for filename in os.listdir(args.input_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        img_path = os.path.join(args.input_dir, filename)
        out_path = os.path.join(args.output_dir, filename)

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
            continue

        tensor = data_transforms['test'](img).unsqueeze(0).to(device)

        with torch.no_grad():
            cam = cam_extractor(tensor)

        overlay_cam(img, cam, out_path)
        print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
