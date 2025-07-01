# Fine-Grained Recognition

本项目实现细粒度网络监督多分类任务，模型基于 EfficientNet 架构并结合 Transformer 编码器。

整体流程：EfficientNet Backbone 提取特征后接 SEBlock，然后将特征展平并加入位置编码送入 Transformer Encoder，最后进行分类。

## 环境要求
- Python 3.10
- 安装依赖：
  ```bash
  pip install -r requirements.txt
  ```

## 使用方法

### 训练
`train.py` 用于训练模型，可指定数据集路径和设备等参数：
```bash
python train.py --root path/to/dataset --device cuda
```

### 预测
`main.py` 用于加载训练好的权重并对测试集进行预测：
```bash
python main.py --root path/to/dataset --weights model/model.pth --device cuda
```

