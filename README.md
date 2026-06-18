# Weather Image Classification

> 智海算法调优 · CAIP 强脑赛道 · 智能算法赛项
>
> 基于深度学习的天气图片四分类 —— 多云 / 雨天 / 雪天 / 晴天

---

## 目录

- [赛事背景](#赛事背景)
- [任务定义](#任务定义)
- [评分规则](#评分规则)
- [项目结构](#项目结构)
- [环境搭建](#环境搭建)
- [完整工作流](#完整工作流)
- [配置系统](#配置系统)
- [可选模型](#可选模型)
- [模块详解](#模块详解)
- [提交指南](#提交指南)
- [开发指南](#开发指南)
- [常见问题](#常见问题)
- [赛事规则摘要](#赛事规则摘要)

---

## 赛事背景

**智海算法调优** 是 CAIP 强脑赛道下的智能算法赛项。选手需要实现一个天气图片分类模型，对互联网来源的普通天气照片进行自动分类。

**核心挑战**：手机拍摄的天气图片尺寸不统一、拍摄视角多样、天气类型之间存在视觉混淆（如多云 vs 晴天、雨天 vs 雪天），同时最终模型必须在 **CPU 环境** 下 **70 分钟内** 完成评分集推理。

---

## 任务定义

### 分类任务

对输入天气图片进行**四分类**：

| 类别 | 英文 | 典型特征 |
|------|------|----------|
| 多云 | cloudy | 大片云层、天空灰白 |
| 雨天 | rainy | 雨滴、路面湿润、灰暗 |
| 雪天 | snowy | 积雪、白色覆盖、明亮 |
| 晴天 | sunny | 蓝天、阳光、高对比度 |

### 数据特点

| 属性 | 说明 |
|------|------|
| 图片类型 | 类似手机拍摄的普通天气照片（非卫星云图） |
| 图片来源 | 互联网，包含多种拍摄视角和场景 |
| 图片尺寸 | **不统一**，需要预处理 |
| 图片格式 | jpg / jpeg / png / bmp 等 |
| 数据划分 | 训练集（选手可用） → 测试集（验证效果） → 评分集（隐藏，最终评分） |

### 数据集选用

本项目训练数据来自三个公开数据集的合并，均为真实户外拍摄的天气照片（非卫星图、非合成图）：

| 数据集 | 来源 | 规模 | 选用类别 |
|--------|------|------|----------|
| **Image2Weather** | [NCKU 学术项目](http://mmcv.csie.ncku.edu.tw/~wtchu/projects/Weather/) | 183,798 张（选用 18,621 张） | cloudy / rainy / snowy / sunny |
| **Kaggle Weather Detection** | [tamimresearch](https://www.kaggle.com/datasets/tamimresearch/weather-detection-image-dataset) | 7,700 张（选用 rain + snow） | rain → rainy, snow → snowy |
| **Kaggle Multi-class Weather** | [pratik2901](https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset) | 1,125 张（全用） | Cloudy / Rain / Shine+Sunrise → sunny |

**Image2Weather** 是主要数据源，图片来自 Flickr 用户拍摄的真实户外照片，通过 GPS+时间戳与气象站数据匹配标注。该数据集标签来自气象站记录而非视觉判断，存在一定标签噪声，但更贴近比赛「手机随手拍」的真实分布。

**图片样式**：全部为互联网来源的真实户外照片，手机/数码相机拍摄，非卫星云图，含多种场景（街道、建筑、自然风景等），与比赛描述高度一致。

#### 最终数据分布

| 类别 | 训练集 | 验证集 | 合计 |
|------|--------|--------|------|
| cloudy | 6,640 | 1,660 | 8,300 |
| rainy | 1,828 | 456 | 2,284 |
| snowy | 1,562 | 390 | 1,952 |
| sunny | 6,888 | 1,722 | 8,610 |
| **总计** | **16,918** | **4,228** | **21,146** |

> **类别不平衡比 4.41:1** — rainy 和 snowy 是少数类。训练时建议使用项目中实现的 FocalLoss (`--training.loss.name focal`) 来缓解此问题。

#### 复现数据

```bash
# 下载数据集（需要 Kaggle 账号）
source venv/bin/activate
pip install kagglehub

python -c "
import kagglehub
kagglehub.dataset_download('tamimresearch/weather-detection-image-dataset')
kagglehub.dataset_download('pratik2901/multiclass-weather-dataset')
"

# 下载 Image2Weather（约 6GB）
wget http://mmcv.csie.ncku.edu.tw/~wtchu/projects/Weather/Image.zip -O data/raw/Image2Weather.zip

# 整理合并
python scripts/organize_datasets.py
```

---

## 评分规则

### 核心公式

```
最终得分 = Macro F1 × 100
```

### 评分优先级

| 优先级 | 指标 | 说明 |
|--------|------|------|
| **1st** | **Macro F1** | 四个类别 F1 的算术平均 — 每个类别同等重要 |
| 2nd | CPU 推理时间 | 同分时按推理时间排序（短 → 长） |
| 3rd | 资源效率 | 内存占用、CPU 利用率 |
| 4th | 代码规范 | 注释完整性、结构清晰度 |

### ⚠️ 同分排序规则

如果最终得分相同，依次比较：
1. 模型推理时间（由短到长）
2. 内存占用率、CPU/GPU 利用率
3. 代码规范性（注释完整性、代码结构清晰度）

---

## 项目结构

```
weather-classification/
│
├── 📂 configs/                     # 实验配置
│   ├── default.yaml                #   默认配置（所有模型继承）
│   └── models/                     #   各 backbone 专属配置
│       ├── resnet18.yaml           #   ResNet-18 ~11.7M
│       ├── resnet34.yaml           #   ResNet-34 ~21.8M
│       ├── efficientnet_b0.yaml    #   EfficientNet-B0 ~5.3M
│       ├── mobilenetv3_small.yaml  #   MobileNetV3-S ~2.5M
│       └── convnext_tiny.yaml      #   ConvNeXt-T ~28.6M
│
├── 📂 data/                        # 数据处理模块
│   ├── dataset.py                  #   WeatherDataset — 支持多格式、灰度/RGBA 转换、损坏检测
│   ├── transforms.py               #   训练/验证/测试 transforms — 保守增强策略
│   ├── dataset_report.py           #   DatasetAnalyzer — 自动生成分析报告
│   └── label_mapping.py            #   LabelMapper — 从目录自动检测类别，双向映射
│
├── 📂 models/                      # 模型定义
│   ├── base.py                     #   WeatherClassifier — backbone → head 统一封装
│   └── model_factory.py            #   MODEL_REGISTRY (12个 backbone) + create_model()
│
├── 📂 training/                    # 训练模块
│   ├── trainer.py                  #   Trainer.fit() — 完整训练循环 + AMP + 梯度累积
│   ├── metrics.py                  #   compute_macro_f1 / per-class F1 / 混淆矩阵
│   ├── losses.py                   #   CrossEntropy / LabelSmoothing / FocalLoss
│   └── callbacks.py                #   EarlyStopping / ModelCheckpoint / TrainingLogger
│
├── 📂 inference/                   # 推理 & 提交模块
│   ├── predictor.py                #   WeatherPredictor — 批量 CPU 推理 + 时间预估
│   ├── benchmark.py                #   CpuBenchmark — 延迟/吞吐/内存 全面评测
│   └── submit_checker.py           #   SubmitChecker — 12 项提交前自动检查
│
├── 📂 scripts/                     # 可执行脚本
│   ├── analyze_data.py             #   ① 数据集分析 → reports/dataset_report.md
│   ├── train.py                    #   ② 训练入口 — 支持命令行覆盖配置
│   ├── evaluate.py                 #   ③ 模型评估 → 混淆矩阵 + per-class F1
│   ├── inference.py                #   ④ 批量推理 → CSV 输出
│   ├── benchmark_cpu.py            #   ⑤ CPU 性能评测 → 70分钟可行性验证
│   └── prepare_submission.py       #   ⑥ 打包提交 + 12 项自动检查
│
├── 📂 tests/                       # 单元测试 (38 tests)
│   ├── test_dataset.py             #   数据集加载 / 标签映射 / 数据增强
│   ├── test_models.py              #   模型创建 / 前向传播 / 参数统计
│   ├── test_inference.py           #   推理链路 / CPU 基准 / 提交检查
│   └── test_submission.py          #   提交约束验证 / 确定性输出 / CPU 依赖
│
├── 📂 submit/                      # 提交目录（prepare_submission.py 生成）
├── 📂 weights/                     # 模型权重存储
├── 📂 outputs/                     # 训练输出（日志、checkpoint、历史记录）
├── 📂 reports/                     # 分析报告（数据集报告、混淆矩阵等）
├── 📂 experiments/                 # 实验记录
│
├── CLAUDE.md                       # AI Agent 工作指引
├── README.md                       # 本文件
├── requirements.txt                # Python 依赖
└── .gitignore                      # Git 忽略规则
```

---

## 环境搭建

### 比赛运行环境约束

根据比赛方 Q&A A63，提交侧运行环境需要按以下资源约束设计：

| 项目 | 约束 |
| --- | --- |
| PyTorch | 最高 `torch 2.1.7` |
| 设备 | CPU only |
| CPU | 2 核 |
| 内存 | 8 GiB |
| 推理时限 | 评分集总推理时间 ≤ 70 分钟 |

因此，提交依赖必须限制在平台可用版本内。本仓库的 `requirements.txt` 和 `scripts/prepare_submission.py` 生成的 `submit/requirements.txt` 均应保持：

```text
torch>=2.0.0,<=2.1.7
torchvision>=0.15.0,<0.17.0
```

注意：本地开发环境可以更高，但最终提交前必须在不超过 `torch 2.1.7` 的环境中做 smoke test。2 核 CPU / 8 GiB 内存意味着最终模型需要优先考虑 CPU 推理速度、batch size 和峰值内存，不能只看验证集 F1。

### 1. 创建虚拟环境

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate          # Linux / macOS
# 或
venv\Scripts\activate             # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "import torch; import torchvision; print(f'PyTorch {torch.__version__} ready, CUDA: {torch.cuda.is_available()}')"
```

### 4. 运行测试

```bash
pytest tests/ -v
```

预期输出：`38 passed`

---

## 完整工作流

### 第零步：了解你的数据

```bash
python scripts/analyze_data.py --data_dir /path/to/training_data
```

输出文件：
- `reports/dataset_report.md` — 完整分析报告
- `reports/class_distribution.csv` — 类别分布
- `reports/bad_images.txt` — 损坏图片列表
- `reports/label_mapping.json` — 类别映射（**提交时必须使用**）

### 第一步：训练基线模型

```bash
# 使用 ResNet-18 作为基线
python scripts/train.py \
    --config configs/models/resnet18.yaml \
    --data_dir /path/to/training_data \
    --output_dir outputs/resnet18_baseline
```

训练输出（在 `outputs/resnet18_baseline/` 下）：
- `best_model.pth` — 最佳 macro F1 权重
- `checkpoints/` — Top-K 检查点
- `training_history.csv` — 每轮指标记录
- `training_log.jsonl` — 详细日志
- `confusion_matrix_final.png` — 混淆矩阵

### 第二步：评估模型

```bash
python scripts/evaluate.py \
    --weights outputs/resnet18_baseline/best_model.pth \
    --model resnet18 \
    --data_dir /path/to/test_data \
    --label_mapping reports/label_mapping.json
```

### 第三步：CPU 性能评测 ⚠️ 关键步骤

```bash
python scripts/benchmark_cpu.py \
    --weights outputs/resnet18_baseline/best_model.pth \
    --model resnet18
```

**必须确认输出中的 `within_70min: ✅ YES`**，否则需要优化或更换更小的模型。

### 第四步：尝试不同模型

```bash
# 快速模型（推荐 CPU 推理）
python scripts/train.py --config configs/models/mobilenetv3_small.yaml --data_dir /path/to/training_data
python scripts/benchmark_cpu.py --weights outputs/.../best_model.pth --model mobilenetv3_small

# 精度优先模型（需验证 CPU 速度）
python scripts/train.py --config configs/models/efficientnet_b0.yaml --data_dir /path/to/training_data
python scripts/benchmark_cpu.py --weights outputs/.../best_model.pth --model efficientnet_b0

# 更强模型（小心 CPU 超时）
python scripts/train.py --config configs/models/convnext_tiny.yaml --data_dir /path/to/training_data
python scripts/benchmark_cpu.py --weights outputs/.../best_model.pth --model convnext_tiny
```

### 第五步：选择最佳模型并打包提交

```bash
python scripts/prepare_submission.py \
    --weights outputs/resnet18_baseline/best_model.pth \
    --model resnet18 \
    --label_mapping reports/label_mapping.json
```

此脚本会自动：
1. 生成独立的 `submit/inference.py`（无需本项目其他模块）
2. 复制模型权重
3. 创建 `submit/requirements.txt`
4. **运行 12 项提交前检查**
5. 打包为 `submit_<model>.zip`

---

## 配置系统

本项目使用层次化 YAML 配置。所有模型配置继承 `configs/default.yaml`。

### 配置层次

```
configs/default.yaml          ← 基础配置
    └── configs/models/*.yaml ← 模型专属覆盖
        └── 命令行 --key value ← 运行时覆盖
```

### 命令行覆盖

```bash
# 修改训练轮数
python scripts/train.py --config configs/models/resnet18.yaml \
    --training.epochs 100

# 修改图像尺寸
python scripts/train.py --config configs/models/resnet18.yaml \
    --data.image_size 256

# 修改批大小和学习率
python scripts/train.py --config configs/models/resnet18.yaml \
    --training.batch_size 32 --training.optimizer.lr 5e-5

# 使用 Focal Loss 处理类别不平衡
python scripts/train.py --config configs/models/resnet18.yaml \
    --training.loss.name focal --training.loss.focal_gamma 2.0
```

### 关键配置项

| 配置路径 | 说明 | 默认值 |
|----------|------|--------|
| `data.image_size` | 输入图片尺寸 | 224 |
| `data.augmentation.color_jitter.brightness` | 亮度增强幅度 | 0.15 |
| `model.dropout` | Dropout 比例 | 0.3 |
| `model.freeze_backbone` | 冻结 backbone | false |
| `training.batch_size` | 批大小 | 64 |
| `training.optimizer.lr` | 学习率 | 1e-4 |
| `training.loss.name` | 损失函数 | cross_entropy |
| `training.early_stopping.patience` | 早停耐心值 | 10 |
| `inference.batch_size` | 推理批大小 | 32 |

---

## 可选模型

| 模型 | 参数量 | 权重 ~MB | CPU 延迟估算 | 适用场景 |
|------|--------|----------|-------------|----------|
| `mobilenetv3_small` | 2.5M | ~10 | 5-10ms | 🚀 CPU 推理首选 |
| `shufflenet_v2_x1_0` | 2.3M | ~9 | 5-12ms | 🚀 轻量级备选 |
| `efficientnet_b0` | 5.3M | ~20 | 10-20ms | ⭐ 速度/精度均衡 |
| `efficientnet_b1` | 7.8M | ~30 | 15-30ms | 略强于 B0 |
| `resnet18` | 11.7M | ~45 | 15-25ms | 📌 稳定基线 |
| `resnet34` | 21.8M | ~83 | 30-45ms | 精度提升 |
| `resnet50` | 25.6M | ~98 | 45-70ms | 可能过拟合 |
| `densenet121` | 8.0M | ~30 | 40-60ms | Dense 连接 |
| `mobilenetv3_large` | 5.5M | ~22 | 10-18ms | 速度精度折中 |
| `convnext_tiny` | 28.6M | ~110 | 60-100ms | ⚠️ 需验证 CPU 速度 |
| `efficientnet_b2` | 9.1M | ~35 | 25-45ms | ⚠️ 需验证 CPU 速度 |
| `squeezenet1_0` | 1.2M | ~5 | 5-10ms | 精度有限 |

> **推荐策略**：从 `resnet18` 建立基线 → 测试 `mobilenetv3_small` 的速度 → 对比 `efficientnet_b0` 的精度 → 选择 Macro F1 与 CPU 推理时间的最佳平衡点。

---

## 模块详解

### data/ — 数据处理

#### WeatherDataset (`dataset.py`)
- 支持 jpg / jpeg / png / bmp / webp / tiff 格式
- 自动处理灰度图（L → RGB）、RGBA（→ RGB 白底）、调色板模式（P → RGB）
- 损坏图片自动跳过并记录到 `bad_images` 列表
- 支持目录结构或 CSV 标注文件两种加载方式

#### 数据增强 (`transforms.py`)
- **保守策略**：天气分类依赖颜色、光照和天空纹理
- `ColorJitter` 幅度限制在 0.15（brightness/contrast/saturation）
- `RandomRotation` 限制在 ±10°（极端旋转会破坏天气语义）
- `RandomHorizontalFlip` 安全用于所有天气类型
- 建议的混淆类别「多云↔晴天」「雨天↔雪天」需谨慎增强

#### LabelMapper (`label_mapping.py`)
- **绝不硬编码类别顺序** — 从数据目录自动检测
- 双向映射：class_name ↔ index
- JSON 序列化，确保训练和推理阶段一致

### models/ — 模型

#### WeatherClassifier (`base.py`)
```
backbone → AdaptiveAvgPool2d → Flatten → Dropout → Linear → logits
```
- 支持冻结 backbone、ONNX 导出
- `predict()` 返回概率和预测索引
- `get_param_count()` 统计可训练/冻结参数量

#### ModelFactory (`model_factory.py`)
- 注册 12 种 backbone，统一 `create_model(name)` 接口
- 自动移除原始分类头，替换为适配 4 分类的新头

### training/ — 训练

#### Trainer (`trainer.py`)
- 自动混合精度（AMP）训练
- 梯度累积支持大 batch
- 梯度裁剪防止爆炸
- 每轮验证并记录 macro F1 / per-class F1 / 混淆矩阵
- 自动保存最佳 macro F1 模型

#### 损失函数 (`losses.py`)
- `CrossEntropyLoss` — 标准分类损失
- `LabelSmoothingCrossEntropy` — 防止过拟合，缓解天气类别间的混淆
- `FocalLoss` — 处理类别不平衡，聚焦难分类样本

### inference/ — 推理

#### CpuBenchmark (`benchmark.py`)
测试多个 batch size 下的延迟（mean/median/p95/p99）、吞吐量、内存占用，并估算 3000 张评分图片的总推理时间。

#### SubmitChecker (`submit_checker.py`)
12 项自动检查：

| # | 检查项 | 说明 |
|---|--------|------|
| 1 | 权重文件存在 | 验证 .pth 文件可被 torch 加载 |
| 2 | 推理脚本语法 | Python 编译检查 |
| 3 | 无 CUDA 依赖 | 检测 `.cuda()` / `device='cuda'` |
| 4 | 依赖可安装 | 验证 requirements.txt |
| 5 | 无硬编码路径 | 检测 `/home/`、`C:\` 等 |
| 6 | 无外部 API | 检测 `requests`、`openai` 等 |
| 7 | 标签映射正确 | 4 个类别、名称匹配 |
| 8 | 输出格式 | CSV 含 `filename` + `prediction` 列 |
| 9 | 冒烟测试 | 实际加载模型运行推理 |
| 10 | 推理速度 | 估算 3000 张是否 ≤ 70 分钟 |
| 11 | 权重大小 | ≤ 500 MB |
| 12 | 代码结构 | docstring、`__main__`、注释、参数解析 |

---

## 提交指南

### ⚠️ 重要提醒

- **正式提交只有一次机会** — 提交前务必运行完整的 smoke test
- 测试提交次数无限制，建议先多次测试
- 提交内容：推理代码 + 模型权重（不是只提交分数或代码）

### 提交前检查清单

```
□ 已运行 python scripts/analyze_data.py 了解数据分布
□ 已运行 python scripts/benchmark_cpu.py 确认 70 分钟内可完成推理
□ 已运行 pytest tests/ -v 所有测试通过
□ 已运行 python scripts/prepare_submission.py 生成提交包
□ 已检查 submit/inference.py 可独立运行
□ 已检查类别映射与官方 baseline 一致
□ 已检查推理代码不依赖本地绝对路径
□ 已检查提交包不包含训练代码、调试代码
□ 已检查 requirements.txt 完整且版本号正确
```

---

## 开发指南

### 运行测试

```bash
# 全部测试
pytest tests/ -v

# 指定模块
pytest tests/test_dataset.py -v
pytest tests/test_models.py -v

# 带覆盖率报告
pytest tests/ -v --cov=. --cov-report=html
```

### 添加新模型

1. 在 `models/model_factory.py` 的 `MODEL_REGISTRY` 中注册：

```python
"my_new_model": {
    "builder": models.my_new_model,
    "in_features": 512,
    "description": "My custom model",
},
```

2. 创建配置文件 `configs/models/my_new_model.yaml`
3. 训练并评测

### 代码风格

- 所有模块使用绝对导入
- 类和方法包含 docstring
- 关键决策处添加注释说明原因
- 不要硬编码路径或类别顺序

---

## 常见问题

<details>
<summary><b>Q: 类别映射顺序重要吗？</b></summary>

**非常重要。** 类别顺序必须与官方数据目录结构或官方 baseline 保持一致。本项目使用 `detect_label_mapping()` 从数据目录自动检测，避免手动硬编码。提交时使用 `reports/label_mapping.json` 确保一致性。
</details>

<details>
<summary><b>Q: ColorJitter 可以调大吗？</b></summary>

**不建议。** 天气分类高度依赖颜色、光照和天空纹理。过大的 ColorJitter 会破坏这些关键语义特征。默认的 0.15 已经是保守值，如果数据量较少，建议进一步降低或完全关闭 color jitter。
</details>

<details>
<summary><b>Q: 推理超时怎么办？</b></summary>

1. 减小模型 → `mobilenetv3_small` 或 `efficientnet_b0`
2. 减小输入尺寸 → `image_size: 160` 或 `192`
3. 增大推理 batch size → `batch_size: 64`
4. 使用 ONNX 导出加速 → `model.export_onnx()`
5. 减少 DataLoader workers → `num_workers: 1`（避免 CPU 竞争）
</details>

<details>
<summary><b>Q: 训练时 GPU 显存不够？</b></summary>

```bash
python scripts/train.py --config configs/models/resnet18.yaml \
    --training.batch_size 16 \
    --data.image_size 192
```
</details>

<details>
<summary><b>Q: Macro F1 和 Accuracy 哪个更重要？</b></summary>

**Macro F1。** 比赛评分为 Macro F1 × 100。Accuracy 高不代表 Macro F1 高——如果模型对少数类表现差，Macro F1 会被明显拉低。Trainer 默认以 val_macro_f1 作为模型选择的指标。
</details>

---

## 赛事规则摘要

> 详细规则参见 `weather_agent_useful_info.md`

### ✅ 允许

- 使用开源模型和预训练权重
- 安装第三方库
- 本地训练后上传权重
- 使用 AI 大模型辅助编程
- 测试提交次数无限制

### ❌ 禁止

- 在最终推理时调用外部大模型 API
- 访问未授权数据
- 将隐藏评分集用于训练
- 攻击系统或窃取数据
- 抄袭或影响比赛公平性

---

## License

MIT License — 本项目仅供学习和比赛使用。

---

<p align="center">
  <b>🏆 Good luck in the competition!</b>
</p>
