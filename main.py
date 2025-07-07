import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from config import *  # 包含 data_transforms, TestDataset 等
from model import MultiStreamFeatureExtractor


def predict(model, dataloader, device):
    model.eval()
    predictions = []
    filenames = []
    with torch.no_grad():
        for inputs, names in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            predictions.extend(preds)
            filenames.extend(names)
    return filenames, predictions


def save_results(filenames, predictions, output_path, idx_to_class):
    # 构造 DataFrame：图片名加后缀，标签写名称并补零
    rows = []
    for name, pred in zip(filenames, predictions):
        # 保留原始后缀
        if not os.path.splitext(name)[1]:
            name += ".jpg"
        label_str = '="' + idx_to_class[pred].zfill(4) + '"'  # 强制 Excel 识别为文本
        rows.append([name, label_str])

    df = pd.DataFrame(rows, columns=["id", "label"])
    df.sort_values(by="id", inplace=True)
    df.to_csv(output_path, index=False, header=False)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 路径配置
    root_dir = "data/WebFG-400/train"
    test_dir = "data/WebFG-400/test"  # 测试图片文件夹路径
    model_path = "model/del_noise_97/model.pth"
    output_path = "submission.csv"

    # 加载测试集
    test_dataset = TestDataset(test_dir, transform=data_transforms["test"])
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=8
    )

    # 加载训练集类名顺序（用于 idx -> class name）
    train_dataset = get_train_dataset()
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    # 模型构建与加载
    model = MultiStreamFeatureExtractor(num_classes=len(idx_to_class))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # 预测 + 保存
    filenames, predictions = predict(model, test_loader, device)
    save_results(filenames, predictions, output_path, idx_to_class)
