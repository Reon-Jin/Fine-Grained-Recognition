import os

import torch
from torch.utils.data import DataLoader
import pandas as pd
from config import *
from model import AIModel
root_dir = "data/WebFG-400/train"


def predict(model, dataloader, device):
    model.eval()
    predictions = []
    filenames = []
    with torch.no_grad():
        for inputs, names in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)
            filenames.extend(names)
    return filenames, predictions


def save_results(filenames, predictions, output_path):
    results = pd.DataFrame({"id": filenames, "label": predictions})
    results.sort_values(by="id", inplace=True)
    results.to_csv(output_path, index=False, header=False)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = "/testdata"
    test_dataset = TestDataset(test_dir, transform=data_transforms["test"])
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=16
    )
    model_path = "model.pth"
    model = AIModel(num_classes=len(os.listdir(root_dir)))
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    filenames, predictions = predict(model, test_loader, device)
    output_path = "../cla_pre.csv"
    save_results(filenames, predictions, output_path)

