# clean_icc.py
import os
from PIL import Image

# 支持的图像格式
EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def strip_icc_from_dir(root_dir):
    """
    递归遍历 root_dir 下所有子目录，重新保存每个图像文件，
    从而清除所有 metadata（包括 iCCP profile）。
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in EXTS:
                continue
            full_path = os.path.join(dirpath, fname)
            try:
                # 打开并转换为 RGB（丢弃原有 metadata）
                img = Image.open(full_path).convert("RGB")
                # 重新保存，覆盖原文件
                img.save(full_path)
                print(f"✅ Cleaned: {full_path}")
            except Exception as e:
                print(f"⚠️ Failed: {full_path}  Error: {e}")

if __name__ == "__main__":
    # 计算项目根目录下的 data/WebFG-400 的绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "WebFG-400")
    if not os.path.isdir(data_dir):
        print(f"❌ 找不到目录：{data_dir}")
        exit(1)

    for part in ["train" ]:
        folder = os.path.join(data_dir, part)
        if os.path.isdir(folder):
            print(f"\n--- Cleaning folder: {folder} ---")
            strip_icc_from_dir(folder)
        else:
            print(f"⚠️ 目录不存在，跳过：{folder}")

    print("\n🎉 All done. Please rerun train.py and you should no longer see libpng warnings.")
