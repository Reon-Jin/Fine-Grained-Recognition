#!/usr/bin/env python3
import os
import json
import argparse

def json_to_txt(json_file, root_dir, output_txt, classes):
    """
    从 json 文件读取图像相对路径列表，推断类别索引并写入 txt:
      full_image_path label_idx
    支持 Windows 风格反斜杠和 POSIX 风格斜杠。
    """
    with open(json_file, 'r') as f:
        img_list = json.load(f)

    lines = []
    for img_rel_path in img_list:
        # 1. 统一分隔符：将所有反斜杠替换为正斜杠
        img_rel_path = img_rel_path.replace('\\', '/')

        # 2. 拼接、规范化路径
        if os.path.isabs(img_rel_path):
            img_path = os.path.normpath(img_rel_path)
        else:
            img_path = os.path.normpath(os.path.join(root_dir, img_rel_path))

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"路径不存在: {img_path}")

        # 3. 根据父文件夹名确定类别索引
        cls_name = os.path.basename(os.path.dirname(img_path))
        if cls_name not in classes:
            raise ValueError(f"未在 '{root_dir}' 下找到类别文件夹 '{cls_name}'")
        cls_idx = classes.index(cls_name)

        lines.append(f"{img_path} {cls_idx}\n")

    # 写入 txt
    with open(output_txt, 'w') as f:
        f.writelines(lines)
    print(f"✔ 生成 {output_txt} （共 {len(lines)} 条记录）")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="根据 train.json/val.json 生成 train.txt 和 val.txt"
    )
    parser.add_argument(
        '--root_dir',
        default='../data/WebFG-400/train',
        help="原始图像根目录（包含 400 个子文件夹）"
    )
    parser.add_argument(
        '--split_dir',
        default='../data/WebFG-400/',
        help="输出 train.txt/val.txt 的目录"
    )
    parser.add_argument(
        '--train_json',
        default='../data/WebFG-400/train.json',
        help="保存训练集相对路径列表的 JSON 文件"
    )
    parser.add_argument(
        '--val_json',
        default='../data/WebFG-400/val.json',
        help="保存验证集相对路径列表的 JSON 文件"
    )
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.split_dir, exist_ok=True)

    # 获取类别列表（按文件夹名排序，确保索引一致）
    classes = sorted([
        d for d in os.listdir(args.root_dir)
        if os.path.isdir(os.path.join(args.root_dir, d))
    ])
    print(f"✔ 发现 {len(classes)} 个类别：{classes[:5]} ...")

    # 生成 train.txt / val.txt
    json_to_txt(
        args.train_json,
        args.root_dir,
        os.path.join(args.split_dir, 'train.txt'),
        classes
    )
    json_to_txt(
        args.val_json,
        args.root_dir,
        os.path.join(args.split_dir, 'val.txt'),
        classes
    )
