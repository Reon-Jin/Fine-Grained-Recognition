#!/usr/bin/env python3
import argparse
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm
from src.model import PartialResNet
from src.config import val_transform  # 用于推断

# 允许加载截断图像
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PseudoDataset(Dataset):
    """
    读取 clean_train.txt，返回 (img_tensor, img_path)
    """
    def __init__(self, list_file):
        lines = open(list_file).read().splitlines()
        self.paths = [l.split()[0] for l in lines]
        self.transform = val_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert('RGB')
        except OSError:
            img = Image.new('RGB', (224,224), (0,0,0))
        return self.transform(img), path

def main():
    parser = argparse.ArgumentParser(description="用 partial_model 生成伪标签")
    parser.add_argument("--clean_txt",
                        default="../data/WebFG-400/clean_train.txt",
                        help="clean_train.txt 路径")
    parser.add_argument("--partial_model",
                        default="../model/partial_model.pth",
                        help="部分标签模型权重")
    parser.add_argument("--out_txt",
                        default="../data/WebFG-400/pseudo_train.txt",
                        help="输出伪标签列表")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # 确保输出目录存在
    out_dir = os.path.dirname(args.out_txt)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 加载模型
    model = PartialResNet().to(device)
    model.load_state_dict(torch.load(args.partial_model,
                                     map_location=device))
    model.eval()

    # 2) 构建推断数据集
    ds = PseudoDataset(args.clean_txt)
    loader = DataLoader(ds,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True)

    # 3) 推断并收集 (path, pred)
    pseudo_lines = []
    with torch.no_grad():
        for x, paths in tqdm(loader, desc="Generating pseudo labels", ncols=100):
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().tolist()
            for path, p in zip(paths, preds):
                pseudo_lines.append(f"{path} {p}\n")

    # 4) 写文件
    with open(args.out_txt, 'w') as f:
        f.writelines(pseudo_lines)
    print(f"✅ Generated {args.out_txt} with {len(pseudo_lines)} entries")

if __name__ == "__main__":
    main()
