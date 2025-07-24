# Fine Grained Recognition (PyTorch)

This project contains a simple implementation of SR-GNN using PyTorch. The original TensorFlow code was replaced with PyTorch modules.

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
python ./script/main.py dataset_dir ./datasets/Cars nb_classes 196 epochs 150 model_name srgnn
```
Configuration parameters can be changed in `config.yaml` or passed via command line.

