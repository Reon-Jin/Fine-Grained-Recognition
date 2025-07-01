# main.py -- Inference / submission generation script
import os, argparse, csv
import torch
import pandas as pd
from torch.utils.data import DataLoader

from config import get_test_dataset, get_train_dataset
from model import AIModel

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

def save_results(ids, preds, class_names, out_file):
    labels = [f'="{str(class_names[int(p)]).zfill(4)}"' for p in preds]
    pd.DataFrame({'id': ids, 'label': labels}).to_csv(
        out_file, index=False, header=False,
        quoting=csv.QUOTE_NONE, escapechar='\\'
    )
    print('Saved', out_file)

def main():
    parser = argparse.ArgumentParser()
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', default=default_device,
                        help='"cuda", "cuda:0", "cpu", or GPU index like "0"')
    parser.add_argument('--root', default='data/WebFG-400',
                        help='Dataset root directory')
    parser.add_argument('--weights', default='model/model.pth',
                        help='Path to model weights')
    parser.add_argument('--outfile', default='submission.csv',
                        help='Output CSV file name')
    args = parser.parse_args()

    dev_arg = args.device.strip()
    if dev_arg.isdigit():
        dev_arg = f'cuda:{dev_arg}'
    device = torch.device(dev_arg)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Load class names from training data
    class_names = get_train_dataset(args.root).classes
    num_classes = len(class_names)

    # Build model and load weights
    model = AIModel('vit-b/16', num_classes).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    # Test loader
    test_set = get_test_dataset(args.root)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)

    ids, preds = predict(model, test_loader, device)
    save_results(ids, preds, class_names, args.outfile)

if __name__ == '__main__':
    main()
