# main.py
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from config import TestDataset, data_transforms, get_train_dataset
from model import AIModel

# 推理
def predict(model, loader, device):
    model.eval()
    ids, preds = [], []
    with torch.no_grad():
        for imgs, names in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            pred = outputs.argmax(dim=1).cpu().numpy()
            ids.extend(names)
            preds.extend(pred)
    return ids, preds

# 保存结果到 CSV
def save_results(ids, preds, class_names, out_file):
    """Write numeric class ids starting from 1."""
    labels = [p + 1 for p in preds]
    df = pd.DataFrame({'id': ids, 'label': labels})
    df.to_csv(out_file, index=False)

# 主推理脚本
def main():
    root = 'data/WebFG-400'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载类别名称
    train_set = get_train_dataset(root)
    class_names = train_set.classes

    # 加载模型
    num_classes = len(class_names)
    model = AIModel('efficientnet-b0', num_classes).to(device)
    model.load_state_dict(torch.load('model/model.pth', map_location=device))

    # 构建测试 DataLoader
    test_set = TestDataset(os.path.join(root, 'test'), transform=data_transforms['test'])
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)

    ids, preds = predict(model, test_loader, device)
    save_results(ids, preds, class_names, 'submission.csv')

if __name__ == '__main__':
    main()
