from ultralytics import YOLO
import os, shutil
from collections import Counter

# 初始化模型
yolo = YOLO('yolov5su.pt')
conf_threshold = 0.3

src_root = 'data/WebFG-400/train'
dst_root = 'data/WebFG-400/train_filter'
os.makedirs(dst_root, exist_ok=True)

for cls_folder in os.listdir(src_root):
    src_cls = os.path.join(src_root, cls_folder)

    # —— 统计每个类别出现次数，找出“主”类别 ——
    counts = Counter()
    for img_name in os.listdir(src_cls):
        img_path = os.path.join(src_cls, img_name)
        res = yolo(img_path)[0]
        for cls_id, conf in zip(res.boxes.cls, res.boxes.conf):
            if conf >= conf_threshold:
                counts[int(cls_id)] += 1

    if not counts:
        print(f"[跳过] 文件夹 {cls_folder} 没有任何检测结果")
        continue

    maj_cls_id, _ = counts.most_common(1)[0]
    maj_cls_name = yolo.names[maj_cls_id]
    print(f"{cls_folder} → 主检测类别：{maj_cls_name}(id={maj_cls_id})")

    # —— 只保留含有主类别的图片，拷贝到 train_filter/cls_folder ——
    dst_cls = os.path.join(dst_root, cls_folder)
    os.makedirs(dst_cls, exist_ok=True)
    for img_name in os.listdir(src_cls):
        img_path = os.path.join(src_cls, img_name)
        res = yolo(img_path)[0]
        if any((int(c) == maj_cls_id and f >= conf_threshold)
               for c, f in zip(res.boxes.cls, res.boxes.conf)):
            shutil.copy(img_path, os.path.join(dst_cls, img_name))
