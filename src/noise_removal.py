#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
import cv2
from ultralytics import YOLO
from collections import Counter


def detect_main_class(model, img_paths, conf_thres):
    """
    对单个文件夹的图片列表运行检测，统计出现频次最多的类别ID作为主类别。
    返回主类别ID及每张图片的检测结果（列表 of sets）。
    """
    model.conf = conf_thres
    preds = []  # 每张图片预测出的类别集合
    for p in img_paths:
        im = cv2.imread(p)
        if im is None:
            preds.append(set())
            continue
        try:
            results = model(p)[0]
            clsids = results.boxes.cls.cpu().numpy().astype(int)
            preds.append(set(clsids.tolist()))
        except Exception:
            preds.append(set())

    # 统计所有出现的类别
    all_classes = []
    for s in preds:
        all_classes.extend(s)
    if not all_classes:
        return None, preds
    main_cls = Counter(all_classes).most_common(1)[0][0]
    return main_cls, preds


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv5 主标签过滤 & 全局 8:2 划分 clean_train/clean_val"
    )
    parser.add_argument("--train_dir",
                        default="../data/WebFG-400/train",
                        help="原始训练集根目录，每个子文件夹一个类别")
    parser.add_argument("--out_train",
                        default="../data/WebFG-400/clean_train.txt",
                        help="输出 clean_train.txt 列表，每行: <path> <cls_idx>")
    parser.add_argument("--out_val",
                        default="../data/WebFG-400/clean_val.txt",
                        help="输出 clean_val.txt 列表，每行: <path> <cls_idx>")
    parser.add_argument("--model",
                        default="../yolov5su.pt",
                        help="本地 YOLOv5 权重文件")
    parser.add_argument("--conf_thres", type=float, default=0.3,
                        help="检测置信度阈值")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                        help="训练/验证划分比例，默认 0.8")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，保证划分可复现")
    args = parser.parse_args()

    random.seed(args.seed)

    # 加载模型
    model = YOLO(args.model)
    kept = []  # 全局保留列表

    # 遍历每个子文件夹
    for cls_idx, cls_name in enumerate(sorted(os.listdir(args.train_dir))):
        cls_dir = os.path.join(args.train_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        imgs = sorted([f for f in os.listdir(cls_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        paths = [os.path.join(cls_dir, f) for f in imgs]

        main_cls, preds = detect_main_class(model, paths, args.conf_thres)
        if main_cls is None:
            # 未检测到任何目标，保留全部图片
            kept.extend([(p, cls_idx) for p in paths])
        else:
            for p, dets in zip(paths, preds):
                if main_cls in dets:
                    kept.append((p, cls_idx))
                else:
                    # 删除未检测到主类别的图片
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    # 全局打乱并按比例分割
    random.shuffle(kept)
    cut = int(len(kept) * args.split_ratio)
    train_list = kept[:cut]
    val_list = kept[cut:]

    # 写入文件
    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)
    with open(args.out_train, 'w') as f_train:
        for p, idx in train_list:
            f_train.write(f"{p} {idx}\n")

    os.makedirs(os.path.dirname(args.out_val), exist_ok=True)
    with open(args.out_val, 'w') as f_val:
        for p, idx in val_list:
            f_val.write(f"{p} {idx}\n")

    total = sum(len(files) for _, _, files in os.walk(args.train_dir) if False)  # placeholder
    print(f"✅ 完成：保留 {len(kept)} 张 → train: {len(train_list)}, val: {len(val_list)}")

if __name__ == "__main__":
    main()
