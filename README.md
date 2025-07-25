# Fine Grained Recognition (PyTorch)

This project contains a PyTorch implementation for fine-grained recognition using a ResNet50 backbone enhanced with a CBAM attention module. During training, a live Matplotlib window displays losses, accuracies and attention maps.

## Setup
Create an environment and install the dependencies:
```bash
conda create -n fg_env python=3.9
conda activate fg_env
pip install -r requirements.txt
```

## Dataset Structure
```
datasets/
  dataset_1/
    train/
      class_01/
      class_02/
      ...
    test/
      class_01/
      class_02/
      ...
```

## Training
Run training from the project root:
```bash
python ./script/main.py dataset_dir ./datasets/Cars nb_classes 196 epochs 150 model_name cbam_resnet
```
Running the command will open a Matplotlib window that updates after each epoch with the latest loss curves, accuracies and attention maps.
All configuration parameters can be changed in `config.yaml` or overridden via the command line.

## Visualizing Attention
To visualise the spatial attention produced by a trained model on a single image, run:
```bash
python ./script/visualize_attention.py path/to/model.pth path/to/image.jpg
```
