# Fine-Grained Recognition

本项目实现细粒度网络监督多分类任务，模型基于 EfficientNet 架构并结合 Transformer 编码器。

整体流程：EfficientNet Backbone 提取特征后，依次加入 SEBlock、CBAM 以及轻量级 RPN，
在自动聚焦局部判别区域的同时抽取候选部件特征。然后将全局特征展平并加入位置编码送入
更深的 Transformer Encoder，融合局部候选特征后进行分类。

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

若数据集中包含带 alpha 通道的 RGBA 图像，数据加载器会自动转为 RGB，避免训练时的
相关警告。

### 预测
`main.py` 用于加载训练好的权重并对测试集进行预测：
```bash
python main.py --root path/to/dataset --weights model/model.pth --device cuda
```

