#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
from tqdm import tqdm
from ultralytics import YOLO

def detect_folder(
    img_paths, model, conf_thres, keep_ratio, min_ratio, rerun=True
):
    """
    对单个文件夹的图片列表运行检测，返回要保留的索引列表。
    如果第一次保留比例 < keep_ratio 并且 rerun=True，
    会用 conf_thres/2 重跑一次；如果二次保留比例 < min_ratio，则返回 None（保留全部）。
    """
    preds = []
    model.conf = conf_thres
    for p in img_paths:
        results = model(p)          # 返回 Results 对象列表
        r = results[0]
        # boxes.conf, boxes.cls
        if r.boxes.shape[0] == 0:
            preds.append(None)
        else:
            # 取最高置信度的检测
            confs = r.boxes.conf.cpu().numpy()
            clsids = r.boxes.cls.cpu().numpy().astype(int)
            idx = confs.argmax()
            preds.append(int(clsids[idx]))

    # 找出现次数最多的主标签
    freq = {}
    for c in preds:
        if c is not None:
            freq[c] = freq.get(c, 0) + 1
    if not freq:
        return None

    main_cls, main_cnt = max(freq.items(), key=lambda x: x[1])
    N = len(img_paths)
    keep_idxs = [i for i,c in enumerate(preds) if c == main_cls]
    ratio = len(keep_idxs) / N

    if ratio >= keep_ratio:
        return keep_idxs

    if rerun:
        return detect_folder(
            img_paths, model,
            conf_thres=conf_thres * 0.5,
            keep_ratio=keep_ratio,
            min_ratio=min_ratio,
            rerun=False
        )

    if ratio < min_ratio:
        return None

    return keep_idxs

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv5 主标签过滤 + 8:2 划分 clean_train/clean_val"
    )
    parser.add_argument("--train_dir",
                        default="../data/WebFG-400/train",
                        help="原始训练集根目录，每个子文件夹一个类别")
    parser.add_argument("--out_train",
                        default="../data/WebFG-400/clean_train.txt",
                        help="输出 clean_train.txt")
    parser.add_argument("--out_val",
                        default="../data/WebFG-400/clean_val.txt",
                        help="输出 clean_val.txt")
    parser.add_argument("--model",
                        default="../yolov5su.pt",
                        help="YOLOv5 自定义权重文件")
    parser.add_argument("--conf_thres", type=float, default=0.25,
                        help="检测置信度阈值")
    parser.add_argument("--keep_ratio", type=float, default=0.7,
                        help="首次过滤后，保留比例 ≥ keep_ratio 则采用结果")
    parser.add_argument("--min_ratio", type=float, default=0.5,
                        help="二次过滤后，保留比例 < min_ratio 则保留全部")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                        help="训练/验证划分比例，默认 0.8")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，保证划分可复现")
    args = parser.parse_args()

    random.seed(args.seed)

    # 1. 加载模型（无需网络下载）
    model = YOLO(args.model)

    per_class = {}
    # 2. 对每个子文件夹做过滤
    for cls_idx, cls_name in enumerate(sorted(os.listdir(args.train_dir))):
        cls_dir = os.path.join(args.train_dir, cls_name)
        if not os.path.isdir(cls_dir): continue

        imgs = sorted([f for f in os.listdir(cls_dir)
                       if f.lower().endswith(('.jpg','.png','.jpeg'))])
        paths = [os.path.join(cls_dir, f) for f in imgs]

        keep = detect_folder(
            paths, model,
            conf_thres=args.conf_thres,
            keep_ratio=args.keep_ratio,
            min_ratio=args.min_ratio
        )

        lines = []
        if keep is None:
            # 全部保留
            for p in paths:
                lines.append(f"{p} {cls_idx}\n")
        else:
            for i in keep:
                lines.append(f"{paths[i]} {cls_idx}\n")

        per_class[cls_idx] = lines

    # 3. 按类打乱并按比例划分
    train_lines, val_lines = [], []
    for cls_idx, lines in per_class.items():
        random.shuffle(lines)
        cut = int(len(lines) * args.split_ratio)
        train_lines.extend(lines[:cut])
        val_lines.extend(lines[cut:])

    # 4. 写文件
    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)
    with open(args.out_train, 'w') as f:
        f.writelines(train_lines)
    with open(args.out_val, 'w') as f:
        f.writelines(val_lines)

    total = sum(len(v) for v in per_class.values())
    kept = len(train_lines) + len(val_lines)
    print(f"✅ 完成：原始 {total} 张 → 保留 {kept} 张")
    print(f"   train: {len(train_lines)} → {args.out_train}")
    print(f"    val : {len(val_lines)} → {args.out_val}")

if __name__ == "__main__":
    main()
