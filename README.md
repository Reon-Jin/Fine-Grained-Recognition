# Fine-Grained Recognition with Progressive Noise-robust PNP

## 项目简介
本仓库实现了一套面向网络数据集的细粒度图像分类流程，核心模型基于 EfficientNet-V2-L 主干并在分类阶段加入以下改进：

- **通道注意力**：通过 ECA (Efficient Channel Attention) 在全局平均池化前抑制噪声干扰。
- **多分支分类头**：同时使用线性层与两个不同深度的 MLP 头输出分类 logits，并将三者平均以获得最终预测，兼顾稳定性与表达能力。
- **噪声鲁棒训练策略**：结合标签平滑、强弱增强一致性约束以及多种损失函数，提升带噪标签场景下的鲁棒性。

仓库同时提供：

- WebFG 等细粒度数据集的数据加载与增强管线。
- 训练日志、模型保存与实验配置的完整闭环。
- 推理脚本 `newwrite.py`，用于批量生成预测 CSV（可选带置信度）。
- `compare.py` 与 `csv数据比较器.py` 两个 CSV 校验工具，方便对比不同提交或模型输出。

## 目录结构
```
.
├── betterPNP.py              # 主训练脚本，包含模型定义与训练流程
├── newwrite.py               # 推理脚本，批量生成预测 CSV
├── compare.py                # 终端版 CSV 差异验证器
├── csv数据比较器.py          # 带可视化界面的 CSV 差异验证器
├── utils/                    # 数据增强、日志、损失等辅助模块
├── data/                     # 自定义数据集定义（IndexedImageFolder）
├── requirements.txt          # 依赖列表
└── README.md
```

## 环境准备
1. **创建 Python 环境**（推荐 Python 3.9+）：
   ```bash
   conda create -n finegrain python=3.9 -y
   conda activate finegrain
   ```

2. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```
   > 注意：`torch==2.7.1+cu126` 为 GPU 版本，需匹配本地 CUDA 12.6 运行环境，可在 [PyTorch 官网](https://pytorch.org/) 按照提示安装对应的 WHL 包。

## 数据准备
训练脚本默认使用 `Datasets/<dataset_name>` 作为数据根目录，结构需满足：
```
Datasets/
└── web-400/
    ├── train/
    │   ├── 000/
    │   │   ├── xxx.jpg
    │   │   └── ...
    │   └── ...
    └── val/
        ├── 000/
        │   ├── yyy.jpg
        │   └── ...
        └── ...
```
- 目录名 `train` 与 `val` 固定，用于训练与验证集。
- 子目录使用三位或四位编号，对应类别标签（`IndexedImageFolder` 会按照文件夹顺序自动编号）。
- 若需要使用其它数据集，可在 `betterPNP.py` 的 `Config` 中修改 `dataset`、`database` 与 `n_classes`。

## 训练模型
训练入口为 `betterPNP.py`，该文件末尾 `Config` 类给出了默认实验配置，可直接运行：
```bash
python betterPNP.py
```

关键参数说明：
- `dataset`：数据集文件夹名称，需以 `web-` 前缀开头。
- `database`：数据集父目录，默认 `Datasets`。
- `n_classes`：类别数。
- `batch_size`、`epochs`、`lr`、`opt` 等：常规训练超参。
- `rescale_size`、`crop_size`：输入尺寸；`build_transform` 内还包含 RandAugment 与强弱增强策略。
- `save_model`：是否保存每个 epoch 的权重及最佳模型。
- `use_fp16`、`use_grad_accumulate`：混合精度与梯度累积开关。

训练过程中会在 `Results/<dataset>/Better_PNP_efficientnet_v2L/<project>/<log>-时间戳/` 下保存：
- `log.txt`：逐 epoch 指标（训练/验证损失与精度）。
- `epoch_*.pth`：按 epoch 保存的模型权重。
- `best_epoch.pth`：验证精度最高的模型。
- `params.json`、`network.txt`：完整超参与网络结构备份。

### 自定义配置
如需快速尝试不同配置，可直接修改 `Config` 类中的属性；若想以命令行传参方式运行，可仿照 `Config` 构建 argparse，并将参数传入 `main` 函数。

## 推理与生成 CSV
推理脚本 `newwrite.py` 支持批量图像预测，并生成两份 CSV：
- `pred_results_web400.csv`：仅包含文件名与四位类别编号。
- `pred_results_web400_with_confidence.csv`：额外输出模型置信度。

### 使用步骤
1. 将待预测图片放入 `INPUT_DIR` 指定的目录（支持多种图像格式）。
2. 修改脚本顶部配置：
   - `INPUT_DIR`：测试图片根目录。
   - `CKPT_PATH`：训练得到的权重路径（`.pth`）。
   - `NUM_CLASSES`：类别数需与训练保持一致。
   - `GPU`：使用的 GPU 编号；若在 CPU 上推理，可设为 `None` 并关闭 `USE_AMP`。
   - `OUTPUT`：自定义输出 CSV 路径（可留空使用默认文件名）。
3. 运行脚本：
   ```bash
   python newwrite.py
   ```

脚本将自动执行：
- 读取 EfficientNet-V2-L 模型并加载权重。
- 对每张图片应用 5 种测试时增强（原图、水平翻转、亮度、对比度、多尺度），按设定权重融合 logits。
- 写出两份 CSV 文件，格式示例：
  ```
  filename,prediction
  I0001.jpg,0005
  I0002.jpg,0123
  ```
  带置信度版本会额外追加 `confidence` 列。

## CSV 结果校验
### 终端版差异验证器
使用 `compare.py` 快速比较两份预测 CSV：
```bash
python compare.py --csv1 pred_results_web400.csv --csv2 vote400.csv
```
脚本会输出差异数量、重合度百分比，并列出标签不同的文件。

### 可视化差异验证器
`csv数据比较器.py` 在终端比较的基础上提供 OpenCV 图形界面，可同时浏览两个标签下的样本，支持点击放大查看：
```bash
python csv数据比较器.py \
    --csv1 submission_a.csv \
    --csv2 submission_b.csv \
    --train_data_path Datasets/web-400/train \
    --test_images_path Datasets/web-400/test
```
界面操作说明：
- 左右箭头切换差异样本。
- `A`/`D` 键切换当前标签下的多张图片。
- 鼠标左键点击任意图片可弹出放大窗口。

## 常见问题
- **RandAugment 未找到**：请确认已安装 `randaugment` 库，或将自定义的 RandAugment 实现放入 Python 路径。
- **CUDA 版本不匹配**：若本地 CUDA < 12.6，请安装与之对应的 PyTorch 版本，并在 `requirements.txt` 中调整 `torch` 与 `torchvision`。
- **CSV 标签格式**：默认使用四位数字（如 `0005`）。如需其它格式，可修改 `newwrite.py` 中写入逻辑。

## 引用
如需在论文或项目中引用本仓库，请参考：
> Fine-Grained Recognition with Progressive Noise-robust PNP (2025).
