import os
import argparse
import torch
import pandas as pd
import csv
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFile
from model import BCNN

# allow truncated images in inference as well
ImageFile.LOAD_TRUNCATED_IMAGES = True

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.transform = transform
        # list all image files
        self.samples = sorted(
            [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname = self.samples[idx]
        img_path = os.path.join(self.root, fname)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, fname

# inference
def predict(model, loader, device):
    model.eval()
    ids, preds = [], []
    with torch.no_grad():
        for imgs, names in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            pred = outputs.argmax(dim=1).cpu().numpy()
            ids.extend(names)
            preds.extend(pred)
    return ids, preds

# save to CSV
def save_results(ids, preds, class_names, out_file):
    labels = [f'="{class_names[int(p)].zfill(4)}"' for p in preds]
    df = pd.DataFrame({'id': ids, 'label': labels})
    df.to_csv(
        out_file,
        index=False,
        header=False,
        quoting=csv.QUOTE_NONE,
        escapechar='\\'
    )
    # print first 5 lines to verify
    with open(out_file, 'r') as f:
        print("CSV前5行内容:")
        for _ in range(5):
            print(f.readline().strip())

# main script
def main():
    parser = argparse.ArgumentParser()
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', default=default_device,
                        help='inference device, e.g. cuda or cpu')
    parser.add_argument('--root', default='data/WebFG-400',
                        help='dataset root directory')
    parser.add_argument('--weights', default='model/model.pth',
                        help='path to model weights')
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # load class names from training split
    from torchvision.datasets import ImageFolder
    train_folder = ImageFolder(os.path.join(args.root, 'train'))
    class_names = train_folder.classes

    # load model
    num_classes = len(class_names)
    model = BCNN(num_classes=num_classes, unfreeze_last_stage=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    # test transforms
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # test dataset & loader
    test_dir = os.path.join(args.root, 'test')
    test_set = TestDataset(test_dir, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)

    ids, preds = predict(model, test_loader, device)
    save_results(ids, preds, class_names, 'submission.csv')

if __name__ == '__main__':
    main()