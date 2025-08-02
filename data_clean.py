"""
使用YOLO检测并保留包含飞机、车辆、鸟类的图像，删除其他无关图像
重点过滤：非目标对象、车辆/飞机内部、不完整对象、一群鸟
改进：先生成最终数据集，人工审查时可直接复制补充
"""

import os
import shutil
import subprocess
import platform
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import json
from datetime import datetime

# YOLO检测的相关类别ID (COCO数据集)
RELEVANT_CLASSES = {
    'airplane': [4],  # airplane
    'car': [2, 3, 5, 6, 7],  # bicycle, car, motorcycle, bus, truck
    'bird': [14],  # bird
}

# 将所有相关类别ID合并
KEEP_CLASS_IDS = []
for category, class_ids in RELEVANT_CLASSES.items():
    KEEP_CLASS_IDS.extend(class_ids)

print(f"保留的YOLO类别ID: {KEEP_CLASS_IDS}")

# 人工审查阈值设置
MANUAL_REVIEW_THRESHOLDS = {
    'deletion_ratio': 0.5,  # 删除比例超过50%需要审查
    'min_remaining': 10      # 剩余图片少于10张需要审查
}

# 精确过滤配置
FILTER_CONFIG = {
    # 面积相关
    'min_area_ratio': 0.03,      # 最小面积占比
    'max_area_ratio': 0.85,      # 最大面积占比（防止内部视角）
    'min_confidence': 0.3,       # 最低置信度

    # 边界切割检测
    'edge_tolerance': 0.08,      # 边界容忍度，超过此值认为被切割
    'min_complete_ratio': 0.7,   # 最小完整度（对象在图像中的完整程度）

    # 内部视角检测
    'vehicle_interior_threshold': 0.7,  # 车辆占比超过此值可能是内部
    'airplane_interior_threshold': 0.8, # 飞机占比超过此值可能是内部

    # 多对象检测（一群鸟）
    'max_bird_count': 2,         # 最多允许的鸟类数量
    'bird_cluster_threshold': 3, # 超过此数量认为是鸟群

    # 长宽比检测
    'aspect_ratio_range': {
        'car': (0.3, 4.0),       # 汽车合理长宽比
        'airplane': (0.5, 6.0),  # 飞机合理长宽比
        'bird': (0.4, 2.5)       # 单只鸟合理长宽比
    }
}


def open_folder_in_explorer(folder_path):
    """在文件资源管理器中打开文件夹"""
    try:
        if platform.system() == 'Windows':
            os.startfile(folder_path)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', folder_path])
        else:  # Linux
            subprocess.run(['xdg-open', folder_path])
        return True
    except Exception as e:
        print(f"无法打开文件夹 {folder_path}: {e}")
        return False


def analyze_detection_quality(boxes_info, img_width, img_height):
    """
    分析检测质量，判断是否应该保留图像

    Args:
        boxes_info: 检测框信息列表 [{'class_id': int, 'confidence': float, 'bbox': [x1,y1,x2,y2]}]
        img_width: 图像宽度
        img_height: 图像高度

    Returns:
        tuple: (是否保留, 原因)
    """
    if not boxes_info:
        return False, "未检测到目标对象"

    target_objects = []
    bird_count = 0

    for obj in boxes_info:
        class_id = obj['class_id']
        confidence = obj['confidence']
        x1, y1, x2, y2 = obj['bbox']

        # 基本检查
        if confidence < FILTER_CONFIG['min_confidence']:
            continue

        # 计算面积和位置信息
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        img_area = img_width * img_height
        area_ratio = box_area / img_area
        aspect_ratio = box_width / box_height

        # 面积检查
        if area_ratio < FILTER_CONFIG['min_area_ratio']:
            continue  # 对象太小

        if area_ratio > FILTER_CONFIG['max_area_ratio']:
            return False, f"对象过大，可能是内部视角 ({area_ratio:.1%})"

        # 边界切割检查
        edge_tolerance = FILTER_CONFIG['edge_tolerance']
        edge_x = img_width * edge_tolerance
        edge_y = img_height * edge_tolerance

        is_cut = (x1 < edge_x) or (x2 > img_width - edge_x) or \
                 (y1 < edge_y) or (y2 > img_height - edge_y)

        if is_cut:
            # 检查切割程度
            cut_left = max(0, edge_x - x1) / box_width
            cut_right = max(0, x2 - (img_width - edge_x)) / box_width
            cut_top = max(0, edge_y - y1) / box_height
            cut_bottom = max(0, y2 - (img_height - edge_y)) / box_height

            total_cut = cut_left + cut_right + cut_top + cut_bottom
            complete_ratio = 1 - total_cut

            if complete_ratio < FILTER_CONFIG['min_complete_ratio']:
                return False, f"对象被严重切割，完整度 {complete_ratio:.1%}"

        # 长宽比检查
        if class_id in [2, 3, 5, 6, 7]:  # 车辆
            obj_type = "车辆"
            min_ratio, max_ratio = FILTER_CONFIG['aspect_ratio_range']['car']
            # 车辆内部检查
            if area_ratio > FILTER_CONFIG['vehicle_interior_threshold']:
                return False, f"疑似车辆内部视角 ({area_ratio:.1%})"

        elif class_id == 4:  # 飞机
            obj_type = "飞机"
            min_ratio, max_ratio = FILTER_CONFIG['aspect_ratio_range']['airplane']
            # 飞机内部检查
            if area_ratio > FILTER_CONFIG['airplane_interior_threshold']:
                return False, f"疑似飞机内部视角 ({area_ratio:.1%})"

        elif class_id == 14:  # 鸟
            obj_type = "鸟类"
            min_ratio, max_ratio = FILTER_CONFIG['aspect_ratio_range']['bird']
            bird_count += 1
        else:
            continue

        if not (min_ratio <= aspect_ratio <= max_ratio):
            return False, f"{obj_type}长宽比异常 {aspect_ratio:.2f}"

        target_objects.append({
            'type': obj_type,
            'confidence': confidence,
            'area_ratio': area_ratio,
            'aspect_ratio': aspect_ratio
        })

    # 检查鸟群
    if bird_count > FILTER_CONFIG['max_bird_count']:
        return False, f"检测到鸟群 ({bird_count}只鸟)"

    # 如果有合格的目标对象，保留图像
    if target_objects:
        best_obj = max(target_objects, key=lambda x: x['confidence'])
        return True, f"保留：{best_obj['type']} (置信度:{best_obj['confidence']:.2f}, 面积:{best_obj['area_ratio']:.1%})"

    return False, "无合格的目标对象"


def detect_relevant_objects(image_path, model, confidence_threshold=0.3):
    """
    检测图像中是否包含符合要求的对象
    """
    try:
        results = model(image_path, conf=confidence_threshold, verbose=False)

        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                classes = boxes.cls.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                box_coords = boxes.xyxy.cpu().numpy()

                img_height, img_width = result.orig_shape

                # 收集目标对象信息
                boxes_info = []
                for i, class_id in enumerate(classes):
                    class_id = int(class_id)
                    if class_id in KEEP_CLASS_IDS:
                        boxes_info.append({
                            'class_id': class_id,
                            'confidence': confidences[i],
                            'bbox': box_coords[i]
                        })

                # 分析检测质量
                should_keep, reason = analyze_detection_quality(boxes_info, img_width, img_height)
                return should_keep

        return False

    except Exception as e:
        print(f"检测失败 {image_path}: {e}")
        return False


def create_temp_review_folder(class_name, removed_images, temp_base_dir):
    """创建临时审查文件夹，复制被删除的图片"""
    temp_folder = os.path.join(temp_base_dir, f"review_{class_name}")
    os.makedirs(temp_folder, exist_ok=True)

    # 清空临时文件夹
    for file in os.listdir(temp_folder):
        file_path = os.path.join(temp_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # 复制被删除的图片到临时文件夹
    copied_files = []
    for img_path in removed_images:
        dest_path = os.path.join(temp_folder, os.path.basename(img_path))
        try:
            shutil.copy2(img_path, dest_path)
            copied_files.append(dest_path)
        except Exception as e:
            print(f"复制失败 {img_path}: {e}")

    return temp_folder, copied_files


def manual_review_folder_with_final_dataset(class_name, folder_info, temp_base_dir, final_class_dir):
    """人工审查界面（最终数据集已生成）"""
    print(f"\n=== 人工审查类别: {class_name} ===")
    print(f"原始图片数: {folder_info['original_count']}")
    print(f"自动保留数: {folder_info['auto_kept']}")
    print(f"被删除数: {len(folder_info['removed_images'])}")
    print(f"删除比例: {folder_info['deletion_ratio']:.1%}")
    print(f"删除原因: {folder_info['review_reason']}")

    # 显示最终数据集位置
    if os.path.exists(final_class_dir):
        final_count = len([f for f in os.listdir(final_class_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        print(f"\n✓ 最终数据集已生成: {final_class_dir}")
        print(f"  当前包含 {final_count} 张图片")
    else:
        print(f"\n⚠️ 该类别无保留图片，最终数据集中无此文件夹")
        final_count = 0

    if not folder_info['removed_images']:
        print("没有被删除的图片，无需审查")
        return {'action': 'no_change'}

    # 创建临时审查文件夹
    temp_folder, copied_files = create_temp_review_folder(
        class_name, folder_info['removed_images'], temp_base_dir
    )

    print(f"\n已创建临时审查文件夹: {temp_folder}")
    print(f"包含 {len(copied_files)} 张被删除的图片")

    # 获取原始文件夹路径
    if folder_info['kept_images']:
        original_folder = os.path.dirname(folder_info['kept_images'][0])
    elif folder_info['removed_images']:
        original_folder = os.path.dirname(folder_info['removed_images'][0])
    else:
        original_folder = None

    while True:
        print(f"\n请选择操作:")
        print(f"1. 打开原始文件夹（查看所有图片）")
        print(f"2. 打开审查文件夹（查看被删除的图片）")
        print(f"3. 打开最终数据集文件夹（当前保留的图片）")
        print(f"4. 接受当前结果，继续下一个")
        print(f"5. 完成所有审查，退出")

        choice = input(f"请输入选择 (1-5): ").strip()

        if choice == '1':
            # 打开原始文件夹
            if original_folder:
                print(f"正在打开原始文件夹: {original_folder}")
                open_folder_in_explorer(original_folder)
            else:
                print("无法确定原始文件夹路径")

        elif choice == '2':
            # 打开审查文件夹
            print(f"正在打开审查文件夹: {temp_folder}")
            print("💡 提示：从这里复制需要保留的图片到最终数据集文件夹")
            open_folder_in_explorer(temp_folder)

        elif choice == '3':
            # 打开最终数据集文件夹
            if os.path.exists(final_class_dir):
                print(f"正在打开最终数据集文件夹: {final_class_dir}")
                print("💡 提示：直接将审查文件夹中的图片复制到这里即可")
                open_folder_in_explorer(final_class_dir)
            else:
                print("最终数据集文件夹不存在，正在创建...")
                os.makedirs(final_class_dir, exist_ok=True)
                open_folder_in_explorer(final_class_dir)

        elif choice == '4':
            # 接受当前结果，继续
            return {'action': 'accept_current'}

        elif choice == '5':
            # 完成所有审查
            return {'action': 'finish_all'}

        else:
            print("无效选择，请重新输入!")


def save_review_report(review_folders, output_dir):
    """保存需要人工审查的文件夹报告"""
    report_path = os.path.join(output_dir, 'manual_review_report.json')

    report_data = {
        'timestamp': datetime.now().isoformat(),
        'thresholds': MANUAL_REVIEW_THRESHOLDS,
        'filter_config': FILTER_CONFIG,
        'folders_need_review': review_folders,
        'total_folders_need_review': len(review_folders),
        'instructions': {
            'how_to_review': [
                "1. 程序已生成初步的最终数据集",
                "2. 打开审查文件夹查看被删除的图片",
                "3. 如有误删的图片，直接复制到最终数据集对应文件夹",
                "4. 最终数据集路径在输出目录的 webfg400_train/train/ 下"
            ]
        }
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"\n人工审查报告已保存至: {report_path}")


def clean_dataset(root_dir, output_dir, confidence_threshold=0.3, enable_manual_review=True):
    """清洗数据集，先生成最终数据集，再进行人工审查"""

    print("加载YOLO模型...")
    model = YOLO('yolov8n.pt')

    train_path = os.path.join(root_dir, 'webfg400_train_dirty', 'train')
    output_path = os.path.join(output_dir, 'webfg400_train', 'train')
    temp_base_dir = os.path.join(output_dir, 'temp_review')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练数据目录未找到: {train_path}")

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(temp_base_dir, exist_ok=True)

    total_images = 0
    review_folders = []

    print(f"\n=== 第一阶段：自动检测（精确过滤模式）===")
    print(f"过滤目标:")
    print(f"  - 非飞机/车/鸟图像")
    print(f"  - 车辆/飞机内部视角")
    print(f"  - 被边界切割的不完整对象")
    print(f"  - 鸟群图像（超过{FILTER_CONFIG['max_bird_count']}只鸟）")
    print(f"  - 长宽比异常的对象")

    folder_results = {}

    for class_name in os.listdir(train_path):
        class_dir = os.path.join(train_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"\n处理类别: {class_name}")

        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(Path(class_dir).glob(f'*{ext}'))
            image_files.extend(Path(class_dir).glob(f'*{ext.upper()}'))

        class_total = len(image_files)
        class_kept = 0
        kept_images_list = []
        removed_images_list = []

        for img_path in tqdm(image_files, desc=f"检测 {class_name}"):
            total_images += 1

            if detect_relevant_objects(str(img_path), model, confidence_threshold):
                kept_images_list.append(str(img_path))
                class_kept += 1
            else:
                removed_images_list.append(str(img_path))

        deletion_ratio = (class_total - class_kept) / class_total if class_total > 0 else 0

        needs_review = False
        review_reason = []

        if deletion_ratio > MANUAL_REVIEW_THRESHOLDS['deletion_ratio']:
            needs_review = True
            review_reason.append(f"删除比例 {deletion_ratio:.1%} 超过阈值")

        if class_kept < MANUAL_REVIEW_THRESHOLDS['min_remaining']:
            needs_review = True
            review_reason.append(f"剩余图片 {class_kept} 张过少")

        folder_results[class_name] = {
            'original_count': class_total,
            'auto_kept': class_kept,
            'deletion_ratio': deletion_ratio,
            'kept_images': kept_images_list,
            'removed_images': removed_images_list,
            'needs_review': needs_review,
            'review_reason': ', '.join(review_reason)
        }

        if needs_review:
            review_folders.append({
                'class_name': class_name,
                'original_count': class_total,
                'auto_kept': class_kept,
                'deletion_ratio': deletion_ratio,
                'review_reason': ', '.join(review_reason)
            })

        print(f"  结果: {class_kept}/{class_total} 张保留 (删除 {deletion_ratio:.1%})")
        if needs_review:
            print(f"  ⚠️  需要人工审查: {', '.join(review_reason)}")

    # 第二阶段：立即生成最终数据集
    print(f"\n=== 第二阶段：生成初步最终数据集 ===")

    final_kept_images = 0
    final_removed_images = 0

    for class_name, folder_info in folder_results.items():
        if not folder_info['kept_images']:
            print(f"类别 {class_name}: 完全删除（无保留图片）")
            final_removed_images += folder_info['original_count']
            continue

        output_class_dir = os.path.join(output_path, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        for img_path in folder_info['kept_images']:
            output_img_path = os.path.join(output_class_dir, os.path.basename(img_path))
            shutil.copy2(img_path, output_img_path)

        final_kept_images += len(folder_info['kept_images'])
        final_removed_images += len(folder_info['removed_images'])

        print(f"类别 {class_name}: {len(folder_info['kept_images'])}/{folder_info['original_count']} 张已复制到最终数据集")

    print(f"\n✓ 初步最终数据集已生成: {output_path}")
    print(f"  保留图片: {final_kept_images} 张")
    print(f"  删除图片: {final_removed_images} 张")

    # 保存审查报告
    if review_folders:
        save_review_report(review_folders, output_dir)

    # 第三阶段：人工审查（可选）
    if enable_manual_review and review_folders:
        print(f"\n=== 第三阶段：人工审查 ({len(review_folders)} 个文件夹) ===")
        print(f"💡 最终数据集已生成，你可以直接将误删的图片复制到对应文件夹")
        print(f"💡 最终数据集位置: {output_path}")

        for i, review_folder in enumerate(review_folders):
            class_name = review_folder['class_name']
            folder_info = folder_results[class_name]
            final_class_dir = os.path.join(output_path, class_name)

            print(f"\n--- 审查进度: {i+1}/{len(review_folders)} ---")

            review_result = manual_review_folder_with_final_dataset(
                class_name, folder_info, temp_base_dir, final_class_dir
            )

            if review_result['action'] == 'finish_all':
                print("用户选择完成所有审查")
                break

    # 统计最终结果
    print(f"\n=== 数据清洗完成 ===")

    # 重新统计最终数据集
    final_total = 0
    for class_name in os.listdir(output_path):
        class_dir = os.path.join(output_path, class_name)
        if os.path.isdir(class_dir):
            class_count = len([f for f in os.listdir(class_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            final_total += class_count
            print(f"最终 {class_name}: {class_count} 张图片")

    print(f"\n总结:")
    print(f"原始图像数: {total_images}")
    print(f"最终保留数: {final_total} ({final_total / total_images * 100:.1f}%)")
    print(f"需要人工审查的文件夹: {len(review_folders)}")
    print(f"最终数据集保存在: {output_path}")

    # 清理临时文件夹
    try:
        shutil.rmtree(temp_base_dir)
        print(f"已清理临时文件夹: {temp_base_dir}")
    except Exception as e:
        print(f"清理临时文件夹失败: {e}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='使用YOLO清洗数据集（先生成最终数据集版本）')
    parser.add_argument('--root', default='.', help='原始数据集根目录')
    parser.add_argument('--output', default='./cleaned_data', help='清洗后数据集输出目录')
    parser.add_argument('--confidence', type=float, default=0.3, help='YOLO检测置信度阈值')
    parser.add_argument('--no_manual_review', action='store_true', help='禁用人工审查')
    parser.add_argument('--deletion_threshold', type=float, default=0.5, help='人工审查删除比例阈值')
    parser.add_argument('--min_remaining', type=int, default=10, help='人工审查最小剩余数阈值')
    parser.add_argument('--max_birds', type=int, default=2, help='允许的最大鸟类数量')

    args = parser.parse_args()

    MANUAL_REVIEW_THRESHOLDS['deletion_ratio'] = args.deletion_threshold
    MANUAL_REVIEW_THRESHOLDS['min_remaining'] = args.min_remaining
    FILTER_CONFIG['max_bird_count'] = args.max_birds

    print("=== 细粒度数据集清洗工具（先生成最终数据集版本）===")
    print(f"原始数据目录: {args.root}")
    print(f"输出目录: {args.output}")
    print(f"YOLO置信度阈值: {args.confidence}")
    print(f"过滤目标: 非目标对象、内部视角、不完整对象、鸟群")
    print(f"最大鸟类数量: {args.max_birds}")
    print(f"人工审查: {'禁用' if args.no_manual_review else '启用（最终数据集已生成）'}")

    clean_dataset(args.root, args.output, args.confidence,
                 enable_manual_review=not args.no_manual_review)


if __name__ == '__main__':
    main()
