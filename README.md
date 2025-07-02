# Fine-Grained Recognition

本项目实现细粒度网络监督多分类任务，模型基于 ConvNeXt Tiny 架构并结合 Transformer 编码器。

整体流程：ConvNeXt Tiny 提取特征后先经过 CBAM 增强，再用 1x1 卷积降维并再次通过 CBAM，
随后展平特征并加入位置编码送入 Transformer Encoder，最后全局池化后完成分类。

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
训练脚本会将指标写入 TensorBoard，默认目录为 `runs/`，可通过 `--logdir` 修改：
```bash
python train.py --root path/to/dataset --logdir my_runs
```

若数据集中包含带 alpha 通道的 RGBA 图像，数据加载器会自动转为 RGB，避免训练时的
相关警告。

### 预测
`main.py` 用于加载训练好的权重并对测试集进行预测：
```bash
python main.py --root path/to/dataset --weights model/model.pth --device cuda
```

### 可视化
- `visualize_cam.py` 使用 Grad-CAM 查看注意力区域
- 训练过程的指标会自动写入 `runs/` 目录，可使用 TensorBoard 实时查看。

