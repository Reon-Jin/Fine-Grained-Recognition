# Progressively Robust Loss for Deep Learning with Noisy Labels
## Introduction
This is the PyTorch implementation for our paper **Progressively Robust Loss for Deep Learning with Noisy Labels**

## Environment
After creating a virtual environment, please install all dependencies:

    $  pip install -r requirements.txt

## Data Preparation
Prepare datasets, namely [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz), WebFG496 (including [Web-CUB](https://wsnfg-sh.oss-cn-shanghai.aliyuncs.com/web-bird.tar.gz), [Web-Car](https://wsnfg-sh.oss-cn-shanghai.aliyuncs.com/web-car.tar.gz) and [Web-Aircraft](https://wsnfg-sh.oss-cn-shanghai.aliyuncs.com/web-aircraft.tar.gz)) and [WebVision](https://data.vision.ee.ethz.ch/cvl/webvision/download.html). 
  ```
  ---dataroot
     ├── cifar100
     │   └── cifar-100-python
     ├── web-bird
     │   ├── train
     │   └── val
     ├── web-car
     │   ├── train
     │   └── val
     ├── web-aircraft
     │   ├── train
     │   └── val
     └── miniwebvision
         ├── info
         ├── val_images_256
         ├── test_images_256
         ├── flickr
         ├── flickr_meta
         ├── google
         └── google_meta
  ```

## Training

- Training on CIFAR-100N/CIFAR-80N under 0.4 symmetric/asymmetric label noise:

```python
CUDA_VISIBLE_DEVICES=0     python maincifar.py --synthetic-data cifar100nc/cifar80no  --noise-type  symmetric/asymmetric  --closeset_ratio  0.4  --loss ptce/pgce/ptceplus/pgceplus --t  20/5
```
or
```python
CUDA_VISIBLE_DEVICES=0     python maincifar.py --synthetic-data cifar100nc/cifar80no  --noise-type  symmetric/asymmetric  --closeset_ratio  0.4  --method  ptce/pgce/ptceplus/pgceplus --t  20/5
```

- Training on WebFG496：
```python
CUDA_VISIBLE_DEVICES=0     python mainweb.py    --dataset  web-bird/web-car/web-aircraft   --loss  ptce  
```

- Training on miniWebvision：
```python
CUDA_VISIBLE_DEVICES=0     python mainminiwebvision.py   --loss  ptce  
```

