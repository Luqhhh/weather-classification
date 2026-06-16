# TODO — Weather Image Classification

> 智海算法调优 · CAIP 强脑赛道
> 
> 完成状态：⬜ 待做 | 🔄 进行中 | ✅ 已完成

---

## 1. 项目基础

| 状态 | 任务 | 说明 |
|:----:|------|------|
| ✅ | 项目结构搭建 | 完整模块化代码：data / models / training / inference / scripts / tests |
| ✅ | 数据集收集 | Image2Weather + Kaggle × 2 → 21,146 张四分类图片 |
| ✅ | 虚拟环境 | venv + 全部依赖安装 |
| ✅ | 测试通过 | 38 个单元测试全部通过 |
| ✅ | README 文档 | 含数据集说明、工作流、配置文档 |
| ✅ | Git 提交 | 代码 + 数据均已提交 |

---

## 2. 实验 — 模型训练与选型

| 状态 | 任务 | 命令 | 备注 |
|:----:|------|------|------|
| ⬜ | ResNet-18 基线 | `python scripts/train.py --config configs/models/resnet18.yaml --data_dir data/train` | 第一个基线，记录 macro F1 |
| ⬜ | ResNet-34 | `python scripts/train.py --config configs/models/resnet34.yaml` | 对比 ResNet-18 精度提升 |
| ⬜ | EfficientNet-B0 | `python scripts/train.py --config configs/models/efficientnet_b0.yaml` | 速度/精度均衡 |
| ⬜ | MobileNetV3-Small | `python scripts/train.py --config configs/models/mobilenetv3_small.yaml` | 最轻量，CPU 推理首选 |
| ⬜ | ConvNeXt-Tiny | `python scripts/train.py --config configs/models/convnext_tiny.yaml` | 精度可能更高，需验证 CPU 速度 |
| ⬜ | ResNet-18 + FocalLoss | 加 `--training.loss.name focal --training.loss.focal_gamma 2.0` | rainy/snowy 少样本补偿 |
| ⬜ | ResNet-18 + LabelSmoothing | 加 `--training.loss.name label_smoothing` | 防止过拟合 |

### 2.1 超参数调优

| 状态 | 任务 | 说明 |
|:----:|------|------|
| ⬜ | 学习率网格搜索 | 测试 lr: 1e-3, 5e-4, 1e-4, 5e-5 |
| ⬜ | 输入尺寸对比 | 测试 image_size: 224, 256, 320, 384 |
| ⬜ | Dropout 调优 | 测试 dropout: 0.2, 0.3, 0.5 |
| ⬜ | 增强策略对比 | 测试 ColorJitter 幅度、RandomRotation 角度、RandAugment |
| ⬜ | 类别权重 | 根据类别不平衡比设置 weights |

---

## 3. CPU 性能评测

| 状态 | 任务 | 命令 | 备注 |
|:----:|------|------|------|
| ⬜ | ResNet-18 CPU 评测 | `python scripts/benchmark_cpu.py --weights ... --model resnet18` | 必须 < 70 分钟 |
| ⬜ | EfficientNet-B0 CPU 评测 | 同上 | 预期较快 |
| ⬜ | MobileNetV3 CPU 评测 | 同上 | 预期最快 |
| ⬜ | ConvNeXt-T CPU 评测 | 同上 | 需重点关注是否超时 |
| ⬜ | 不同 batch size 对比 | 在 benchmark 中测试 bs: 8, 16, 32, 64 | 找到 CPU 最优 batch size |
| ⬜ | ONNX 导出测试 | `model.export_onnx('model.onnx')` | 可能加速 CPU 推理 20-30% |
| ⬜ | 3000 张图全量模拟 | 用 data/val 模拟评分集规模推理 | 实际计时 |

---

## 4. 模型评估与分析

| 状态 | 任务 | 说明 |
|:----:|------|------|
| ⬜ | 每个模型生成混淆矩阵 | `python scripts/evaluate.py` |
| ⬜ | 分析易混淆类别对 | 多云↔晴天、雨天↔多云、雪天↔晴天、雨天↔雪天 |
| ⬜ | 错误样本分析 | 找出验证集中预测错误的图片，分析原因 |
| ⬜ | 各类别 F1 对比表 | 汇总所有实验的 per-class F1 |
| ⬜ | 模型选型决策矩阵 | macro F1 × CPU 时间 × 模型大小 三维对比 |

---

## 5. 提交准备 ⚠️ 正式提交仅一次

| 状态 | 任务 | 说明 |
|:----:|------|------|
| ⬜ | 选定最终模型 | 综合 macro F1 + CPU 时间 + 模型大小 |
| ⬜ | 生成提交包 | `python scripts/prepare_submission.py --weights ... --model ...` |
| ⬜ | 12 项自动检查通过 | SubmitChecker 全部 ✅ |
| ⬜ | 独立环境冒烟测试 | 在新 venv 中测试 submit/inference.py 可独立运行 |
| ⬜ | 全量测试集推理验证 | 在平台上用测试集验证效果 |
| ⬜ | 人工确认类别映射 | 对比 reports/label_mapping.json 与官方 baseline |
| ⬜ | 人工确认输出格式 | CSV 列名、编码、换行符 |
| ⬜ | 压缩包最终检查 | 含 inference.py + 权重 + requirements.txt |
| ⬜ | 正式提交 | 🚀 |

---

## 6. 可选优化

| 状态 | 任务 | 说明 |
|:----:|------|------|
| ⬜ | 模型集成 | 2-3 个模型的 soft voting / hard voting |
| ⬜ | MixUp / CutMix 增强 | config 中启用 mixup_alpha / cutmix_alpha |
| ⬜ | 测试时增强 (TTA) | 推理时对图片做多次增强取平均 |
| ⬜ | 知识蒸馏 | 大模型教小模型，保持精度同时加速推理 |
| ⬜ | 渐进式图像尺寸训练 | 先小尺寸快速训练，再大尺寸微调 |

---

## 实验记录模板

每轮实验记录以下信息到 `experiments/`：

```yaml
experiment_id: exp_001
date: 2026-06-16
model: resnet18
config: configs/models/resnet18.yaml

# 训练参数
epochs: 50
batch_size: 64
learning_rate: 1e-4
loss: cross_entropy
augmentation: default (ColorJitter 0.15)

# 验证结果
val_macro_f1: 0.xxxx
val_accuracy: 0.xxxx
per_class_f1:
  cloudy: 0.xxxx
  rainy: 0.xxxx
  snowy: 0.xxxx
  sunny: 0.xxxx

# CPU 评测
cpu_batch_size: 32
avg_latency_ms: xx
estimated_70min: true/false
model_size_mb: xx

# 备注
notes: ""
```

---

## 进度总览

```
项目基础:    ████████████ 100% (6/6)
模型训练:    ░░░░░░░░░░░░   0% (0/7)
超参调优:    ░░░░░░░░░░░░   0% (0/5)
CPU 评测:    ░░░░░░░░░░░░   0% (0/7)
评估分析:    ░░░░░░░░░░░░   0% (0/5)
提交准备:    ░░░░░░░░░░░░   0% (0/9)
可选优化:    ░░░░░░░░░░░░   0% (0/5)
─────────────────────────────
总体进度:    ██░░░░░░░░░░  14%
```
