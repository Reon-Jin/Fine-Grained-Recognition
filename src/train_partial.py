#!/usr/bin/env python3
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PartialLabelDataset
from src.model import PartialResNet
from src.config import (
    BATCH_SIZE, LR, MOMENTUM, WEIGHT_DECAY,
    EPOCHS, TOP_K
)

def partial_loss(logits, candidate_sets):
    probs = F.softmax(logits, dim=1)
    loss = 0.0
    for i, cset in enumerate(candidate_sets):
        loss -= torch.log(probs[i, cset].sum() + 1e-12)
    return loss / logits.size(0)

def collate_fn(batch):
    imgs, idxs, csets = zip(*batch)
    return torch.stack(imgs), torch.tensor(idxs), list(csets)

def main():
    parser = argparse.ArgumentParser(
        description="部分标签学习（改良版）"
    )
    parser.add_argument("--train_txt",
                        default="../data/WebFG-400/clean_train.txt")
    parser.add_argument("--model_out",
                        default="../model/partial_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PartialResNet().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR,
        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    train_ds = PartialLabelDataset(args.train_txt)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=True, collate_fn=collate_fn
    )

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=100)
        for x, idxs, csets in bar:
            x = x.to(device)
            logits = model(x)
            loss = partial_loss(logits, csets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            bar.set_postfix(loss=loss.item())

        # **关键**：每轮结束后直接用 Top-K **替换**候选集
        model.eval()
        eval_loader = DataLoader(
            train_ds, batch_size=64, shuffle=False,
            num_workers=8, pin_memory=True, collate_fn=collate_fn
        )
        with torch.no_grad():
            for x, idxs, _ in tqdm(eval_loader, desc="Updating C(x)", ncols=100):
                x = x.to(device)
                _, preds = model(x).topk(TOP_K, dim=1)
                for i, sid in enumerate(idxs):
                    train_ds.C[sid] = preds[i].cpu().tolist()

        avg_loss = total_loss / len(train_ds)
        print(f"Epoch {epoch} complete — avg loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), args.model_out)
    print("▶ Partial model saved to", args.model_out)

if __name__=="__main__":
    main()
