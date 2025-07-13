import torch
from torch.utils.data import DataLoader
from src.dataset import ImageList
from src.model import PartialResNet
from src.config import BATCH_SIZE, NUM_CLASSES

def evaluate(model_path, list_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PartialResNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    val_ds = ImageList(list_file, train=False)
    loader = DataLoader(val_ds, BATCH_SIZE, num_workers=8)

    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    acc = correct / total * 100
    print(f'Val Acc: {acc:.2f}%')

if __name__ == '__main__':
    # 传入参数或硬编码
    evaluate('../model/partial_model.pth', '../data/WebFG-400/val.txt')
