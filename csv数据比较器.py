import pandas as pd
import cv2
import os
import numpy as np
from pathlib import Path
import argparse



class CSVDifferenceValidator:
    def __init__(self, csv1_path, csv2_path, train_data_path, test_images_path):
        """
        初始化CSV差异验证器

        Args:
            csv1_path: 第一个CSV文件路径 (如submission_b6_enhanced.csv)
            csv2_path: 第二个CSV文件路径 (如submission_b6.csv)
            train_data_path: 训练数据文件夹路径
            test_images_path: 测试图片文件夹路径
        """
        self.csv1_path = csv1_path
        self.csv2_path = csv2_path
        self.train_data_path = train_data_path
        self.test_images_path = test_images_path

        # 读取CSV文件
        self.df1 = pd.read_csv(csv1_path, header=None, names=['image', 'label'])
        self.df2 = pd.read_csv(csv2_path, header=None, names=['image', 'label'])

        print(f"📊 CSV1 ({os.path.basename(csv1_path)}): {len(self.df1)} 条记录")
        print(f"📊 CSV2 ({os.path.basename(csv2_path)}): {len(self.df2)} 条记录")

        # 找出差异
        self.find_differences()

        # 界面控制变量
        self.current_diff_index = 0
        self.window_name = "CSV Difference Validator"

        # 多图片切换相关变量
        self.csv1_images = []  # CSV1标签的所有图片
        self.csv2_images = []  # CSV2标签的所有图片
        self.csv1_current_index = 0  # CSV1当前显示的图片索引
        self.csv2_current_index = 0  # CSV2当前显示的图片索引

        # 放大窗口管理
        self.zoom_windows = set()  # 记录所有打开的放大窗口

        # 创建主窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # 设置鼠标回调
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # 屏幕尺寸
        self.screen_width = 1920
        self.screen_height = 1080

        # 图片区域定义（用于点击检测）
        self.image_regions = {}
        self.current_images = {}  # 存储当前显示的图片

        print(f"🔍 发现 {len(self.differences)} 个不同的预测结果")

    def find_differences(self):
        """找出两个CSV文件中的差异"""
        merged = self.df1.merge(self.df2, on='image', suffixes=('_csv1', '_csv2'))
        self.differences = merged[merged['label_csv1'] != merged['label_csv2']].reset_index(drop=True)

        if len(self.differences) == 0:
            print("✅ 两个CSV文件预测结果完全一致！")
        else:
            print(f"🔍 差异统计:")
            print(f"   总样本数: {len(merged)}")
            print(f"   差异数量: {len(self.differences)}")
            print(f"   一致率: {(len(merged) - len(self.differences)) / len(merged) * 100:.2f}%")

    def read_image_safe(self, image_path):
        """安全读取图片"""
        try:
            with open(image_path, 'rb') as f:
                img_data = f.read()
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"❌ 读取图片失败 {image_path}: {e}")
            return None

    def get_folder_path(self, label):
        """根据4位标签获取对应的3位文件夹路径"""
        folder_name = str(label).zfill(4)[1:]  # 0001 -> 001
        folder_path = os.path.join(self.train_data_path, folder_name)
        return folder_path

    def load_all_images_for_label(self, label, max_images=10):
        """加载某个标签的所有图片"""
        folder_path = self.get_folder_path(label)
        if not os.path.exists(folder_path):
            print(f"⚠️ 文件夹不存在: {folder_path}")
            return []

        images_data = []
        try:
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            image_files.sort()  # 确保顺序一致

            for img_file in image_files[:max_images]:
                img_path = os.path.join(folder_path, img_file)
                img = self.read_image_safe(img_path)
                if img is not None:
                    images_data.append((img, img_file))

        except Exception as e:
            print(f"❌ 读取标签图片出错: {e}")

        return images_data

    def load_images_for_current_difference(self):
        """为当前差异加载所有相关图片"""
        if self.current_diff_index >= len(self.differences):
            return

        diff_row = self.differences.iloc[self.current_diff_index]
        csv1_label = str(diff_row['label_csv1']).zfill(4)
        csv2_label = str(diff_row['label_csv2']).zfill(4)

        # 加载CSV1和CSV2标签的所有图片
        self.csv1_images = self.load_all_images_for_label(csv1_label)
        self.csv2_images = self.load_all_images_for_label(csv2_label)

        # 重置索引
        self.csv1_current_index = 0
        self.csv2_current_index = 0

        print(f"📸 CSV1标签 {csv1_label}: 加载了 {len(self.csv1_images)} 张图片")
        print(f"📸 CSV2标签 {csv2_label}: 加载了 {len(self.csv2_images)} 张图片")

    def get_current_csv1_image(self):
        """获取当前CSV1显示的图片"""
        if self.csv1_images and 0 <= self.csv1_current_index < len(self.csv1_images):
            return self.csv1_images[self.csv1_current_index]
        return None, None

    def get_current_csv2_image(self):
        """获取当前CSV2显示的图片"""
        if self.csv2_images and 0 <= self.csv2_current_index < len(self.csv2_images):
            return self.csv2_images[self.csv2_current_index]
        return None, None

    def resize_image_keep_ratio(self, img, target_width, target_height):
        """调整图片尺寸并保持比例"""
        if img is None:
            return None

        h, w = img.shape[:2]
        scale = min(target_width / w, target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h))
        result = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return result

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数 - 处理点击放大（修复版）"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 检查点击位置是否在某个图片区域内
            for region_name, (rx, ry, rw, rh) in self.image_regions.items():
                if rx <= x <= rx + rw and ry <= y <= ry + rh:
                    if region_name in self.current_images:
                        print(f"🖱️ 点击了 {region_name} 区域，准备放大...")
                        self.show_zoomed_image(self.current_images[region_name], region_name)
                        break

    def show_zoomed_image(self, img, title):
        """显示放大的图片（修复版 - 非阻塞）"""
        if img is None:
            print("❌ 图片为空，无法放大")
            return

        print(f"🔍 正在放大图片: {title}")

        # 创建独立的放大窗口（使用时间戳确保唯一性）
        import time
        timestamp = int(time.time() * 1000) % 10000
        zoom_window = f"Zoomed_{title}_{timestamp}"
        self.zoom_windows.add(zoom_window)  # 记录窗口

        cv2.namedWindow(zoom_window, cv2.WINDOW_NORMAL)

        # 计算放大后的尺寸
        margin = 100
        max_width = self.screen_width - margin
        max_height = self.screen_height - margin

        h, w = img.shape[:2]
        scale = min(max_width / w, max_height / h)

        # 确保至少放大2倍
        if scale < 2.0:
            scale = 2.0

        new_w = int(w * scale)
        new_h = int(h * scale)

        try:
            # 使用更好的插值方法
            zoomed_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            # 创建带标题的画布
            title_height = 80
            canvas_height = new_h + title_height + 40
            canvas_width = max(new_w, 800)
            title_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            # 添加深色背景
            title_canvas.fill(20)

            # 居中放置放大的图片
            img_x = (canvas_width - new_w) // 2
            img_y = title_height
            title_canvas[img_y:img_y + new_h, img_x:img_x + new_w] = zoomed_img

            # 添加标题信息
            cv2.putText(title_canvas, f"Zoomed: {title}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(title_canvas, f"Original Size: {w}x{h} | Zoom: {scale:.1f}x", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.putText(title_canvas, "Press any key or click X to close", (20, canvas_height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

            # 显示放大窗口
            cv2.imshow(zoom_window, title_canvas)

            # 设置窗口位置（避免重叠）
            window_offset = len(self.zoom_windows) * 50
            window_x = 100 + (window_offset % 300)
            window_y = 50 + (window_offset % 200)
            cv2.moveWindow(zoom_window, window_x, window_y)

            # 非阻塞显示
            cv2.waitKey(1)  # 刷新显示
            print(f"✅ 放大窗口已打开: {zoom_window}")

        except Exception as e:
            print(f"❌ 放大图片时出错: {e}")
            if zoom_window in self.zoom_windows:
                self.zoom_windows.remove(zoom_window)
            try:
                cv2.destroyWindow(zoom_window)
            except:
                pass

    def close_all_zoom_windows(self):
        """关闭所有放大窗口"""
        try:
            # 关闭记录的所有放大窗口
            for window_name in list(self.zoom_windows):
                try:
                    cv2.destroyWindow(window_name)
                except:
                    pass

            self.zoom_windows.clear()
            print("🔒 已关闭所有放大窗口")

            # 重新设置主窗口回调（确保鼠标功能正常）
            cv2.setMouseCallback(self.window_name, self.mouse_callback)

        except Exception as e:
            print(f"⚠️ 关闭窗口时出现问题: {e}")

    def create_four_image_layout(self):
        """创建四图布局界面"""
        if self.current_diff_index >= len(self.differences):
            return None

        diff_row = self.differences.iloc[self.current_diff_index]
        image_name = diff_row['image']
        csv1_label = str(diff_row['label_csv1']).zfill(4)
        csv2_label = str(diff_row['label_csv2']).zfill(4)

        # 读取测试图片（答案图）
        test_img_path = os.path.join(self.test_images_path, image_name)
        test_img = self.read_image_safe(test_img_path)

        # 获取当前显示的CSV1和CSV2图片
        csv1_img, csv1_filename = self.get_current_csv1_image()
        csv2_img, csv2_filename = self.get_current_csv2_image()

        # 创建画布
        canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        canvas.fill(25)

        # 计算布局参数
        padding = 40
        title_height = 140
        info_height = 200
        available_height = self.screen_height - title_height - info_height

        half_width = (self.screen_width - 3 * padding) // 2
        half_height = (available_height - padding) // 2

        # 计算四个图片的位置
        left_x = padding
        top_y = title_height
        left_bottom_y = top_y + half_height + padding
        right_x = padding + half_width + padding
        right_bottom_y = top_y + half_height + padding

        # 清空之前的记录
        self.image_regions.clear()
        self.current_images.clear()

        # 绘制标题和信息
        title_text = f"CSV Difference Validator - Progress: {self.current_diff_index + 1}/{len(self.differences)}"
        cv2.putText(canvas, title_text, (padding, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

        cv2.putText(canvas, f"Test Image: {image_name}", (padding, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200),
                    2)
        cv2.putText(canvas, "Click any image to zoom in", (padding, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (100, 255, 100), 2)

        # 1. 左上：答案图1
        if test_img is not None:
            answer1_resized = self.resize_image_keep_ratio(test_img, half_width, half_height)
            canvas[top_y:top_y + half_height, left_x:left_x + half_width] = answer1_resized

            self.image_regions['answer1'] = (left_x, top_y, half_width, half_height)
            self.current_images['answer1'] = test_img

        cv2.rectangle(canvas, (left_x, top_y), (left_x + half_width, top_y + half_height), (100, 100, 100), 2)
        cv2.putText(canvas, "ANSWER 1 (Click to zoom)", (left_x + 15, top_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 2. 左下：答案图2
        if test_img is not None:
            answer2_resized = self.resize_image_keep_ratio(test_img, half_width, half_height)
            canvas[left_bottom_y:left_bottom_y + half_height, left_x:left_x + half_width] = answer2_resized

            self.image_regions['answer2'] = (left_x, left_bottom_y, half_width, half_height)
            self.current_images['answer2'] = test_img

        cv2.rectangle(canvas, (left_x, left_bottom_y), (left_x + half_width, left_bottom_y + half_height),
                      (100, 100, 100), 2)
        cv2.putText(canvas, "ANSWER 2 (Click to zoom)", (left_x + 15, left_bottom_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 3. 右上：CSV1预测标签图
        csv1_color = (0, 255, 0)
        if csv1_img is not None:
            csv1_resized = self.resize_image_keep_ratio(csv1_img, half_width, half_height)
            canvas[top_y:top_y + half_height, right_x:right_x + half_width] = csv1_resized

            self.image_regions['csv1'] = (right_x, top_y, half_width, half_height)
            self.current_images['csv1'] = csv1_img

        cv2.rectangle(canvas, (right_x, top_y), (right_x + half_width, top_y + half_height), csv1_color, 3)

        # 显示CSV1信息和切换状态
        csv1_info = f"CSV1: {csv1_label} ({self.csv1_current_index + 1}/{len(self.csv1_images)})"
        cv2.putText(canvas, csv1_info, (right_x + 15, top_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, csv1_color, 2)
        if csv1_filename:
            cv2.putText(canvas, f"File: {csv1_filename[:25]}...", (right_x + 15, top_y + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)

        # 4. 右下：CSV2预测标签图
        csv2_color = (0, 0, 255)
        if csv2_img is not None:
            csv2_resized = self.resize_image_keep_ratio(csv2_img, half_width, half_height)
            canvas[right_bottom_y:right_bottom_y + half_height, right_x:right_x + half_width] = csv2_resized

            self.image_regions['csv2'] = (right_x, right_bottom_y, half_width, half_height)
            self.current_images['csv2'] = csv2_img

        cv2.rectangle(canvas, (right_x, right_bottom_y), (right_x + half_width, right_bottom_y + half_height),
                      csv2_color, 3)

        # 显示CSV2信息和切换状态
        csv2_info = f"CSV2: {csv2_label} ({self.csv2_current_index + 1}/{len(self.csv2_images)})"
        cv2.putText(canvas, csv2_info, (right_x + 15, right_bottom_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, csv2_color, 2)
        if csv2_filename:
            cv2.putText(canvas, f"File: {csv2_filename[:25]}...", (right_x + 15, right_bottom_y + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)

        # 添加操作说明（底部信息栏）
        info_y = self.screen_height - info_height + 20
        instructions = [
            "🔍 CONTROLS:",
            f"[Enter] Next Difference | [Left Shift] Next CSV1 Image | [Right Shift] Next CSV2 Image",
            f"[Q/W] Prev/Next CSV1 | [E/R] Prev/Next CSV2 | [Mouse Click] Zoom | [C] Close Zooms | [ESC] Exit",
            f"CSV1: {csv1_label} ({self.csv1_current_index + 1}/{len(self.csv1_images)}) | CSV2: {csv2_label} ({self.csv2_current_index + 1}/{len(self.csv2_images)})",
            f"Files: {os.path.basename(self.csv1_path)} vs {os.path.basename(self.csv2_path)}"
        ]

        colors = [(255, 255, 0), (200, 200, 200), (150, 255, 150), (150, 200, 255), (180, 180, 180)]
        font_sizes = [1.0, 0.8, 0.8, 0.7, 0.7]

        for i, (instruction, color, font_size) in enumerate(zip(instructions, colors, font_sizes)):
            cv2.putText(canvas, instruction, (padding, info_y + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 2)

        return canvas

    def run(self):
        """运行差异验证器（修复版）"""
        if len(self.differences) == 0:
            print("✅ 没有差异需要检查！")
            return

        print("=" * 80)
        print("                CSV差异验证器 (多图切换版)")
        print("=" * 80)
        print("🖱️  操作说明:")
        print("- 点击任意图片: 放大查看（可同时打开多个）")
        print("- Enter键: 查看下一个差异")
        print("- Left Shift: 切换到CSV1标签的下一张图片")
        print("- Right Shift: 切换到CSV2标签的下一张图片")
        print("- Q/W键: CSV1标签图片的上一张/下一张")
        print("- E/R键: CSV2标签图片的上一张/下一张")
        print("- C键: 关闭所有放大窗口")
        print("- ESC键: 退出程序")
        print("=" * 80)

        while self.current_diff_index < len(self.differences):
            # 加载当前差异的所有图片
            self.load_images_for_current_difference()

            # 打印当前差异信息
            diff_row = self.differences.iloc[self.current_diff_index]
            print(f"\n🔍 差异 {self.current_diff_index + 1}/{len(self.differences)}:")
            print(f"   图片: {diff_row['image']}")
            print(f"   CSV1预测: {diff_row['label_csv1']} (共{len(self.csv1_images)}张)")
            print(f"   CSV2预测: {diff_row['label_csv2']} (共{len(self.csv2_images)}张)")

            # 内部循环处理同一差异下的图片切换
            while True:
                # 创建显示界面
                display_img = self.create_four_image_layout()

                if display_img is None:
                    break

                cv2.imshow(self.window_name, display_img)

                # 等待按键（短时间等待，避免阻塞鼠标事件）
                key = cv2.waitKey(30) & 0xFF

                # 只有在按下有效按键时才处理
                if key != 255:  # 255表示没有按键
                    print(f"🔧 按键: {key}")

                    if key == 27:  # ESC退出
                        print("\n👋 退出程序")
                        self.close_all_zoom_windows()
                        cv2.destroyAllWindows()
                        return
                    elif key == 13:  # Enter - 下一个差异
                        self.current_diff_index += 1
                        print("➡️ 下一个差异")
                        # 关闭所有放大窗口
                        self.close_all_zoom_windows()
                        break  # 跳出内部循环
                    elif key in [225, 16, 1]:  # Left Shift - CSV1下一张
                        if self.csv1_images:
                            self.csv1_current_index = (self.csv1_current_index + 1) % len(self.csv1_images)
                            print(f"🔄 CSV1: 切换到第 {self.csv1_current_index + 1}/{len(self.csv1_images)} 张")
                    elif key in [226, 17, 2]:  # Right Shift - CSV2下一张
                        if self.csv2_images:
                            self.csv2_current_index = (self.csv2_current_index + 1) % len(self.csv2_images)
                            print(f"🔄 CSV2: 切换到第 {self.csv2_current_index + 1}/{len(self.csv2_images)} 张")
                    elif key == ord('q') or key == ord('Q'):  # Q - CSV1上一张
                        if self.csv1_images:
                            self.csv1_current_index = (self.csv1_current_index - 1) % len(self.csv1_images)
                            print(f"🔄 CSV1: 第 {self.csv1_current_index + 1}/{len(self.csv1_images)} 张")
                    elif key == ord('w') or key == ord('W'):  # W - CSV1下一张
                        if self.csv1_images:
                            self.csv1_current_index = (self.csv1_current_index + 1) % len(self.csv1_images)
                            print(f"🔄 CSV1: 第 {self.csv1_current_index + 1}/{len(self.csv1_images)} 张")
                    elif key == ord('e') or key == ord('E'):  # E - CSV2上一张
                        if self.csv2_images:
                            self.csv2_current_index = (self.csv2_current_index - 1) % len(self.csv2_images)
                            print(f"🔄 CSV2: 第 {self.csv2_current_index + 1}/{len(self.csv2_images)} 张")
                    elif key == ord('r') or key == ord('R'):  # R - CSV2下一张
                        if self.csv2_images:
                            self.csv2_current_index = (self.csv2_current_index + 1) % len(self.csv2_images)
                            print(f"🔄 CSV2: 第 {self.csv2_current_index + 1}/{len(self.csv2_images)} 张")
                    elif key == ord('c') or key == ord('C'):  # C - 关闭所有放大窗口
                        self.close_all_zoom_windows()

            if self.current_diff_index >= len(self.differences):
                print("\n🎉 所有差异已检查完毕！")
                break

        self.close_all_zoom_windows()
        cv2.destroyAllWindows()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='🔍 CSV预测结果差异对比工具（多图切换版）')
    parser.add_argument('--csv1', required=True, help='第一个CSV文件路径')
    parser.add_argument('--csv2', required=True, help='第二个CSV文件路径')
    parser.add_argument('--train_data', required=True, help='训练数据文件夹路径')
    parser.add_argument('--test_images', required=True, help='测试图片文件夹路径')

    args = parser.parse_args()

    # 检查文件是否存在
    for path, name in [(args.csv1, 'CSV1'), (args.csv2, 'CSV2'), (args.train_data, '训练数据'),
                       (args.test_images, '测试图片')]:
        if not os.path.exists(path):
            print(f"❌ 错误: {name}路径不存在: {path}")
            return

    # 创建验证器并运行
    validator = CSVDifferenceValidator(args.csv1, args.csv2, args.train_data, args.test_images)
    validator.run()


# 使用示例
if __name__ == "__main__":
    # 直接运行示例 - 修改这些路径为您的实际路径
    csv1_path = "pred_results_web400.csv"
    csv2_path = "final_predictions400.csv"
    train_data_path = "data/WebFG-400/train"
    test_images_path = "data/WebFG-400/test"

    # 检查文件是否存在
    for path, name in [(csv1_path, 'CSV1'), (csv2_path, 'CSV2'), (train_data_path, '训练数据'),
                       (test_images_path, '测试图片')]:
        if not os.path.exists(path):
            print(f"❌ 错误: {name}路径不存在: {path}")
            exit(1)

    validator = CSVDifferenceValidator(csv1_path, csv2_path, train_data_path, test_images_path)
    validator.run()

    # 或者使用命令行参数
    # main()
