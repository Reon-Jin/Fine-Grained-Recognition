import os
import json

# 根目录和数据文件夹
ROOT_DIR = "data/WebFG-400"
DATA_FOLDER = "train_filter"
TRAIN_JSON = os.path.join(ROOT_DIR, "train.json")
VAL_JSON   = os.path.join(ROOT_DIR, "val.json")

# 输出的 filter json
TRAIN_FILTER_JSON = os.path.join(ROOT_DIR, "train_filter.json")
VAL_FILTER_JSON   = os.path.join(ROOT_DIR, "val_filter.json")

def filter_paths(input_json, output_json):
    """读取 input_json，保留在 DATA_FOLDER 下存在的路径，写入 output_json。"""
    base_folder = os.path.join(ROOT_DIR, DATA_FOLDER)

    # 载入原始列表
    with open(input_json, 'r') as f:
        paths = json.load(f)

    # 过滤
    valid = []
    for rel in paths:
        abs_path = os.path.join(base_folder, rel)
        if os.path.isfile(abs_path):
            valid.append(rel)
        # else: 路径不存在，自动跳过

    # 写入新的 json
    with open(output_json, 'w') as f:
        json.dump(valid, f, indent=2, ensure_ascii=False)

    print(f"{input_json} → {output_json}：{len(valid)} / {len(paths)} paths kept.")

if __name__ == "__main__":
    filter_paths(TRAIN_JSON, TRAIN_FILTER_JSON)
    filter_paths(VAL_JSON,   VAL_FILTER_JSON)
