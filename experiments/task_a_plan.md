# Task A Plan — Backbone & Input Resolution

> Owner: A | 基于 [spec](task_a_spec.md) | 2026-06-18

---

## 总览

```
Round 1（Phase 1→2→3）: CE + 固定增强，纯结构对比
  Phase 1 — 6 个 backbone 初筛 (ResNet-34 / EfficientNet-B0/B1/B2 / MobileNetV3-Small / ConvNeXt-Tiny)
  Phase 2 — Top 2 × 4 种输入尺寸 (224 / 256 / 320 / 384)
  Phase 3 — 最优 combo × 3 个 dropout (0.2 / 0.3 / 0.5)
  → 产出 Top-K backbone 排名 + 最优组合

Round 2（Phase 4）: B 的最优 loss/aug 复跑 Top 2
  → 产出加入不平衡处理后的最终上限
```

---

## 开始前 — 环境

```bash
# 确保统一版本
pip install -r requirements-dev.txt

# 验证
python -c "import torch; assert torch.__version__[:6] == '2.1.2+'; print('OK', torch.__version__)"
```

---

## Phase 1：Backbone 初筛（6 个新实验）

### 1.1 实验清单

所有 config 均已存在，无需新建文件。EfficientNet B1/B2 的 config 为新创建。

| Exp ID | Backbone | Config | 备注 |
|--------|----------|--------|------|
| exp_001 | ResNet-18 | — | ✅ 已完成 |
| **exp_003** | ResNet-34 | `configs/models/resnet34.yaml` | ✅ 已完成 |
| **exp_004** | EfficientNet-B0 | `configs/models/efficientnet_b0.yaml` | ✅ 已完成 |
| **exp_005** | EfficientNet-B1 | `configs/models/efficientnet_b1.yaml` | ✅ 已完成 |
| **exp_006** | EfficientNet-B2 | `configs/models/efficientnet_b2.yaml` | ⏭️ 跳过 |
| **exp_007** | MobileNetV3-Small | `configs/models/mobilenetv3_small.yaml` | ❌ 淘汰 |
| **exp_008** | ConvNeXt-Tiny | `configs/models/convnext_tiny.yaml` | ✅ 已完成，F1 第一 |

### 1.2 执行

```bash
# exp_003 — ResNet-34 ✅ 已完成
# python scripts/train.py --config configs/models/resnet34.yaml \
#     --output_dir experiments/exp_003_resnet34

# exp_004 — EfficientNet-B0 ✅ 已完成
# python scripts/train.py --config configs/models/efficientnet_b0.yaml \
#     --output_dir experiments/exp_004_efficientnet_b0

# exp_005 — EfficientNet-B1（建议从这里开始，B0 已验证 EfficientNet 可行）
python scripts/train.py --config configs/models/efficientnet_b1.yaml \
    --output_dir experiments/exp_005_efficientnet_b1

# exp_006 — EfficientNet-B2
python scripts/train.py --config configs/models/efficientnet_b2.yaml \
    --output_dir experiments/exp_006_efficientnet_b2

# exp_007 — MobileNetV3-Small
python scripts/train.py --config configs/models/mobilenetv3_small.yaml \
    --output_dir experiments/exp_007_mobilenetv3_small

# exp_008 — ConvNeXt-Tiny（⚠️ 先降 batch_size 到 32）
python scripts/train.py --config configs/models/convnext_tiny.yaml \
    --training.batch_size 32 \
    --output_dir experiments/exp_008_convnext_tiny
```

每个训练完立刻跑评估（无需等全部跑完）：

```bash
# 评估（注意: val 在 train 目录内，由 create_dataloaders 自动 split）
python scripts/evaluate.py \
    --weights experiments/<exp_dir>/best_model.pth \
    --model <model_name> \
    --data_dir data/train

# CPU benchmark
python scripts/benchmark_cpu.py \
    --weights experiments/<exp_dir>/best_model.pth \
    --model <model_name>
```

### 1.3 评估矩阵（逐实验填）

| Exp | Backbone | val_f1 | rainy | snowy | cloudy | sunny | CPU(3000) | 权重 |
|-----|----------|--------|-------|-------|--------|-------|-----------|------|
| 001 | ResNet-18 | 0.8708 | 0.8240 | 0.8927 | 0.8677 | 0.8990 | 0.5 min | 42.6 MB |
| 003 | ResNet-34 | 0.9007 | 0.880 | — | — | — | 1.2 min | ~83 MB |
| 004 | EfficientNet-B0 | 0.8963 | 0.865 | — | — | — | 0.8 min | ~21 MB |
| 005 | EfficientNet-B1 | 0.9014 | 0.885 | 0.925 | 0.884 | 0.911 | 1.1 min | ~25 MB |
| 006 | EfficientNet-B2 | — | — | — | — | — | — | — |
| 007 | MobileNetV3-Small | 0.8173 ❌ | 0.752 | 0.787 | — | — | 0.3 min | ~4 MB |
| 008 | ConvNeXt-Tiny | **0.9071** 🥇 | 0.886 | 0.932 | 0.893 | 0.917 | 1.7 min | ~106 MB |

### 1.4 决策

```
F1 < 0.83         → ✗ 淘汰
CPU > 30 min      → ✗ 淘汰
rainy < 0.75      → ⚠️ 标注风险
其余按 F1 降序    → 取 Top 2 进入 Phase 2
```

---

## Phase 2：输入尺寸对比（6 个实验）

### 2.1 范围

只跑 Phase 1 选出的 **Top 2** backbone。

### 2.2 实验矩阵

| 大小 | 像素比 | 预期 |
|------|--------|------|
| 224 | 100% | 已跑，基线 |
| 256 | 130% | 细节提升，CPU 几乎不变 |
| 320 | 204% | 可能显著改善 rainy/snowy |
| 384 | 294% | 最大收益，但 CPU 陡增 |

### 2.3 执行

```bash
# 假设 Top 1 是 ResNet-34，Top 2 是 EfficientNet-B0（示例）

# Top 1 — ResNet-34 @ 256/320/384
python scripts/train.py --config configs/models/resnet34.yaml \
    --data.image_size 256 \
    --output_dir experiments/exp_009_resnet34_256

python scripts/train.py --config configs/models/resnet34.yaml \
    --data.image_size 320 \
    --output_dir experiments/exp_010_resnet34_320

python scripts/train.py --config configs/models/resnet34.yaml \
    --data.image_size 384 \
    --output_dir experiments/exp_011_resnet34_384

# Top 1 的 224 复用 exp_003 结果即可，不用重跑
```

每跑完一个立即 evaluate + benchmark。

### 2.4 决策

```
delta(256→224) < 0.003  → 维持 224（无实质收益）
delta(320→224) > 0.01   → 推荐 320
delta(384→320) < 0.005 且 CPU+50% → 否决 384
```

### 2.5 结果模板

| Exp | Backbone | Size | val_f1 | rainy | snowy | CPU(3000) | 推荐 |
|----|----------|------|--------|-------|-------|-----------|------|
| — | Top1 | 224 | — | — | — | — | — |
| — | Top1 | 256 | — | — | — | — | — |
| — | Top1 | 320 | — | — | — | — | — |
| — | Top1 | 384 | — | — | — | — | — |

---

## Phase 3：Dropout 调优（3 个实验）

### 3.1 范围

只跑 **最优 backbone + 最优 image_size**。

### 3.2 候选值

| dropout | 适用场景 |
|---------|----------|
| 0.2 | 欠拟合或刚好 |
| **0.3** | **当前默认** |
| 0.5 | train_f1 >> val_f1 时 |

### 3.3 执行

```bash
python scripts/train.py --config configs/models/<best>.yaml \
    --data.image_size <best_size> \
    --model.dropout 0.2 \
    --output_dir experiments/exp_015_dropout_02

python scripts/train.py --config configs/models/<best>.yaml \
    --data.image_size <best_size> \
    --model.dropout 0.5 \
    --output_dir experiments/exp_016_dropout_05

# 0.3 复用 Phase 2 最优尺寸那次的实验结果
```

### 3.4 决策

- train/val gap 最小 + val_f1 最高 → 选它
- val_f1 明显下降 → 不过拟合，回到低 dropout
- gap 大 + val_f1 还行 → 取更高 dropout

---

## Phase 4：用 B 的最优策略复跑 Top 2

> ⏳ 等 B 完成 loss/aug 筛选后执行

### 4.1 前提条件

- [ ] A 已完成 Phase 1~3
- [ ] B 已确定最优 loss 和最优 augmentation
- [ ] B 提供了 config override 的具体值

### 4.2 执行

```bash
# Top 1 + B 最优策略
python scripts/train.py --config configs/models/<top1>.yaml \
    --data.image_size <best_size> \
    --model.dropout <best_dropout> \
    --training.loss.name <B_best_loss> \
    [--training.loss.focal_gamma <value> 等 B 提供的参数...] \
    --output_dir experiments/exp_017_top1_blended

# Top 2 + B 最优策略
python scripts/train.py --config configs/models/<top2>.yaml \
    --data.image_size <best_size> \
    --model.dropout <best_dropout> \
    --training.loss.name <B_best_loss> \
    [--training.loss.focal_gamma <value> 等 B 提供的参数...] \
    --output_dir experiments/exp_018_top2_blended
```

### 4.3 观察重点

| 对比 | 来源 |
|------|------|
| val_f1 变化 | Phase 4 vs Phase 3 |
| rainy F1 变化 | 看 B 的不平衡补偿是否有效 |
| snowy F1 变化 | 同上 |
| cloudy/sunny 是否下降 | 补偿不能牺牲大类 |

---

## 每个实验必须记录

无论哪个 Phase，跑了就填：

```
- experiment_id
- date
- model
- image_size
- dropout
- batch_size             # 特别注意 ConvNeXt
- val_macro_f1
- cloudy_f1 / rainy_f1 / snowy_f1 / sunny_f1
- cpu_time_per_image (ms)
- cpu_3000_total (min)
- model_size_mb
- best_epoch
- notes                  # 如有 OOM、收敛异常等
```

以上字段同步到 `experiments/leaderboard.md`。

---

## 时间估算（现实版）

基于 exp_001 实测数据：ResNet-18 每 epoch ~6 min、early stop 在 epoch 5~7。

| Phase | 实验数 | 单次预估 | 小计 |
|-------|--------|----------|------|
| Phase 1 | 6 | 30~60 min/个 | 3~6 h |
| Phase 2 | 6 | 30~60 min/个 | 3~6 h |
| Phase 3 | 2 新 + 1 复用 | 30~60 min/个 | 1~2 h |
| **Round 1 合计** | | | **7~14 h** |
| Phase 4 | 2 | 30~60 min/个 | 1~2 h |
| **总计** | | | **8~16 h** |

> 实际时间取决于模型收敛速度和 early stop 触发时机。
> 大输入尺寸（320/384）每 epoch 会明显更慢。
> 可以同时开两个终端各跑一个实验（不同 GPU 或 batch 错开），缩短总耗时。

---

## 文件最终结构

```
experiments/
├── task_a_spec.md
├── task_a_plan.md
├── leaderboard.md              ← 持续更新
├── backbone_results.md         ← Phase 3 后产出
├── backbone_results.csv        ← Phase 3 后产出
│
├── exp_001_resnet18_ce_224/      ✅
├── exp_002_resnet18_noaug/       ✅
├── exp_003_resnet34/             ✅
├── exp_004_efficientnet_b0/      ✅
├── exp_005_efficientnet_b1/      ← Phase 1
├── exp_006_efficientnet_b2/      ← Phase 1
├── exp_007_mobilenetv3_small/    ← Phase 1
├── exp_008_convnext_tiny/        ← Phase 1
├── exp_009_top1_256/             ← Phase 2
├── exp_010_top1_320/             ← Phase 2
├── exp_011_top1_384/             ← Phase 2
├── exp_012_top2_256/             ← Phase 2 (Top 2)
├── exp_013_top2_320/             ← Phase 2 (Top 2)
├── exp_014_top2_384/             ← Phase 2 (Top 2)
├── exp_015_dropout_02/           ← Phase 3
├── exp_016_dropout_05/           ← Phase 3
├── exp_017_top1_blended/         ← Phase 4
├── exp_018_top2_blended/         ← Phase 4
```
