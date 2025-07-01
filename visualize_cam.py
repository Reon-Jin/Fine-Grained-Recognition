import argparse
import torch
from PIL import Image
from model import AIModel
from config import data_transforms
import matplotlib.pyplot as plt
import numpy as np

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
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[:, class_idx].sum()
        loss.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam.unsqueeze(1), size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_cam(img, cam, out_file):
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='path to image')
    parser.add_argument('--weights', default='model/model.pth', help='model weights')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    model = AIModel('efficientnet-b2', num_classes=400).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    cam_extractor = GradCAM(model, model.cbam)

    img = Image.open(args.image).convert('RGB')
    transform = data_transforms['test']
    tensor = transform(img).unsqueeze(0).to(device)

    cam = cam_extractor(tensor)
    overlay_cam(np.array(img), cam, 'grad_cam.png')

if __name__ == '__main__':
    main()
