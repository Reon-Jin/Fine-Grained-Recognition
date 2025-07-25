# Fine Grained Recognition (PyTorch)

This project contains a PyTorch implementation for fine grained recognition. Besides the previous SR-GNN baseline, it now provides a ResNet50 based model enhanced with a CBAM attention module. Training statistics and attention maps are logged with TensorBoard for real‑time visualisation.

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
During training you can launch TensorBoard to monitor the metrics:
```bash
tensorboard --logdir runs
```
All configuration parameters can be changed in `config.yaml` or overridden via the command line.

