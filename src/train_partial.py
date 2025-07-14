# partial_train.py

import os
import random
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, models
from PIL import Image
import PIL.ImageFile
from tqdm import tqdm

# 允许加载被截断的图像
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


class FineGrainedDataset(Dataset):
    """ clean_train.txt 每行: <image_path> <label> """
    def __init__(self, txt_path, transform=None):
        with open(txt_path, 'r') as f:
            lines = [l.strip().split() for l in f]
        self.samples = [(p, int(lbl)) for p, lbl in lines]
        self.transform = transform

        # 构建类别到索引列表映射
        self.class_to_indices = {}
        for idx, (_, lbl) in enumerate(self.samples):
            self.class_to_indices.setdefault(lbl, []).append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except OSError:
            # 跳过损坏样本，随机重采样
            new_idx = random.randint(0, len(self.samples)-1)
            return self.__getitem__(new_idx)

        if self.transform:
            img = self.transform(img)
        return img, lbl, idx


class CategoryBalancedBatchSampler(Sampler):
    """
    每个 batch 随机选 C 类，每类采样 n_star 个样本
    batch_size = C * n_star
    """
    def __init__(self, class_to_indices, batch_size, n_star):
        assert batch_size % n_star == 0, "batch_size 必须能被 n_star 整除"
        self.class_to_indices = class_to_indices
        self.batch_size = batch_size
        self.n_star = n_star
        self.C = batch_size // n_star
        self.classes = list(class_to_indices.keys())
        # 每 epoch 的 batch 数
        total_samples = sum(len(idxs) for idxs in class_to_indices.values())
        self.num_batches = total_samples // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen = random.sample(self.classes, self.C)
            batch = []
            for cls in chosen:
                batch.extend(random.sample(self.class_to_indices[cls], self.n_star))
            yield batch

    def __len__(self):
        return self.num_batches


class TopKRecallLoss(nn.Module):
    """ Top-k Recall Optimization Loss """
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, feats, labels):
        B, D = feats.size()
        feats = F.normalize(feats, p=2, dim=1)
        S = feats @ feats.t()         # [B,B] 余弦相似度
        loss = feats.new_tensor(0.)

        for i in range(B):
            sim = S[i].clone()
            sim[i] = -1e9             # 排除自身
            topk_vals, topk_idx = torch.topk(sim, self.k, largest=True)
            topk_set = set(topk_idx.tolist())

            # 负集 N：top-k 中标签 ≠ yi
            neg = [j for j in topk_idx.tolist() if labels[j] != labels[i]]
            sum_neg = sim[neg].sum() if neg else sim.new_tensor(0.)

            # 正集 P：其他位置标签 = yi
            others = set(range(B)) - topk_set - {i}
            pos = [j for j in others if labels[j] == labels[i]]
            sum_pos = sim[pos].sum() if pos else sim.new_tensor(0.)

            loss = loss + (sum_neg - sum_pos)

        return loss / B


class PartialModel(nn.Module):
    """ ResNet50 Backbone + Projector """
    def __init__(self, embed_dim=512):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.projector = nn.Linear(2048, embed_dim)

    def forward(self, x):
        x = self.backbone(x)          # [B,2048,1,1]
        x = x.view(x.size(0), -1)     # [B,2048]
        return self.projector(x)      # [B,embed_dim]


def train(args):
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    ds = FineGrainedDataset(args.data, transform=transform)

    # 类别均衡 batch 采样
    sampler = CategoryBalancedBatchSampler(
        ds.class_to_indices,
        batch_size=args.batch_size,
        n_star=args.n_star
    )
    dl = DataLoader(ds,
                    batch_sampler=sampler,
                    num_workers=4,
                    pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PartialModel(embed_dim=args.embed_dim).to(device)
    criterion = TopKRecallLoss(k=args.k)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    # 线性 warm-up 10 epoch 后保持 lr 不变
    def adjust_lr(epoch):
        if epoch < args.warmup:
            lr = args.lr * float(epoch + 1) / args.warmup
        else:
            lr = args.lr
        for g in optimizer.param_groups:
            g['lr'] = lr

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        adjust_lr(epoch)

        total_loss = 0.0
        with tqdm(dl,
                  desc=f"[Epoch {epoch+1}/{args.epochs}]",
                  unit="batch") as pbar:
            for imgs, labels, _ in pbar:
                imgs, labels = imgs.to(device), labels.to(device)
                feats = model(imgs)
                loss = criterion(feats, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dl)
        print(f"Epoch {epoch+1} done — avg loss: {avg_loss:.4f}")

    # 保存模型
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"=> partial_model saved to {args.save_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data',
                        type=str,
                        default='../data/WebFG-400/clean_train.txt',
                        help='干净训练集 txt 文件路径')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch_size = C * n_star')
    parser.add_argument('--n_star',
                        type=int,
                        default=4,
                        help='每个类别采样 n* 个样本 (论文中 n*=4)')  # :contentReference[oaicite:2]{index=2}
    parser.add_argument('--k',
                        type=int,
                        default=5,
                        help='Top-k 参数 k=5 (论文中)')  # :contentReference[oaicite:3]{index=3}
    parser.add_argument('--embed_dim',
                        type=int,
                        default=512,
                        help='特征维度')
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help='总训练 epoch 数 (论文中 110)')  # :contentReference[oaicite:4]{index=4}
    parser.add_argument('--warmup',
                        type=int,
                        default=5,
                        help='warm-up epoch 数 (论文中 10)')  # :contentReference[oaicite:5]{index=5}
    parser.add_argument('--lr',
                        type=float,
                        default=5e-3,
                        help='初始学习率 5e-3')  # :contentReference[oaicite:6]{index=6}
    parser.add_argument('--weight_decay',
                        type=float,
                        default=2e-5,
                        help='weight decay = 2e-5')  # :contentReference[oaicite:7]{index=7}
    parser.add_argument('--save_path',
                        type=str,
                        default='../model/partial_model.pth',
                        help='保存模型路径')
    args = parser.parse_args()

    train(args)
