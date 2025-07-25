import os
import cv2
import numpy as np
from utils import getROIS
import argparse

def visualize_rois(image_path, out_dir,
                   resolution=224,
                   grid_size=3,
                   min_size=1):
    """
    在原图上绘制 ROI 边框，并保存每个 ROI 裁剪图。

    Args:
        image_path (str): 输入图片路径
        out_dir    (str): 可视化结果保存目录
        resolution (int): 输出图像分辨率，和模型输入一致
        grid_size  (int): 网格大小，即每行/列划分数
        min_size   (int): 最小 ROI 边长占比（以网格单元数计）
    """
    os.makedirs(out_dir, exist_ok=True)

    # 读取并缩放到指定分辨率
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (resolution, resolution))

    # 生成 ROI 坐标列表：[[x0,y0,w,h], ...]
    rois = getROIS(resolution=resolution,
                   gridSize=grid_size,
                   minSize=min_size)

    # 在原图上画矩形
    vis = img_resized.copy()
    for idx, (x0, y0, w, h) in enumerate(rois):
        x1, y1 = x0, y0
        x2, y2 = x0 + w, y0 + h
        # 画红色矩形框，线宽 2
        cv2.rectangle(vis, (x1, y1), (x2, y2),
                      color=(255, 0, 0), thickness=2)
        # 在左上角标上编号
        cv2.putText(vis, str(idx),
                    (x1+3, y1+15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 0, 0),
                    thickness=1)

        # 裁剪并保存该 ROI
        crop = img_resized[y1:y2, x1:x2]
        if crop.size > 0:
            crop = cv2.resize(crop, (resolution//grid_size, resolution//grid_size))
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, f"roi_{idx}.jpg"), crop_bgr)

    # 保存带框的整图
    overlay_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_dir, "rois_overlay.jpg"), overlay_bgr)

    print(f"Saved overlay and {len(rois)} crops to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ROIs on an image")
    parser.add_argument(
        "image",
        nargs="?",
        default="../data/WebFG-400/train/000/0f081948f4d9411eb119c8d0644e574d.jpg",
        help="Path to input image (default preset)"
    )
    parser.add_argument(
        "--out", "-o",
        default="roi_vis",
        help="Output directory"
    )
    parser.add_argument(
        "--res", "-r",
        type=int,
        default=224,
        help="Image resolution (square)"
    )
    parser.add_argument(
        "--grid", "-g",
        type=int,
        default=3,
        help="Grid size (divide image into grid×grid ROIs)"
    )
    parser.add_argument(
        "--min", "-m",
        type=int,
        default=1,
        dest="min_size",
        help="Min ROI size in grid units"
    )
    args = parser.parse_args()

    visualize_rois(
        image_path=args.image,
        out_dir=args.out,
        resolution=args.res,
        grid_size=args.grid,
        min_size=args.min_size
    )
