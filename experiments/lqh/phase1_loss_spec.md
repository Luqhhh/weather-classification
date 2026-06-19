# Phase 1 — Loss 对比实验 Spec

> 目标：在 baseline (ResNet-18, 224, 默认增强, seed=42) 上找出最优 loss。
> 基准线：exp_001: val macro F1=0.8708, rainy F1=0.8240, snowy F1=0.8927。
> 原则：每次只改 loss，不改其他参数。

## 实验矩阵

| # | ID | Loss | loss.name | 额外参数 | 预期效果 |
|---|-----|------|-----------|----------|----------|
| 1 | exp_ce_baseline | CE | cross_entropy | — | 同 exp_001，基线复核 |
| 2 | exp_focal | FocalLoss | focal | gamma=2.0 | rainy recall ↑ |
| 3 | exp_label_smoothing | LabelSmoothingCE | label_smoothing | smoothing=0.1 | cloudy/sunny 混淆 ↓ |
| 4 | exp_weighted_ce | Weighted CE | cross_entropy | sqrt weights=[0.80,1.52,1.65,0.78] | rainy/snowy F1 ↑ |
| 5 | exp_weighted_focal | Weighted FocalLoss | focal | gamma=2.0 + sqrt weights | 仅当 #2 或 #4 有收益时跑 |

## 类别权重推导

使用 **sqrt 权重**（比 balanced 更保守，对数据集分布差异更鲁棒）：

```
公式: w = sqrt(total / (n_classes × count_per_class))

总数: cloudy=6640, rainy=1828, snowy=1562, sunny=6888 → total=16918

cloudy: sqrt(16918 / (4 × 6640)) = sqrt(0.637) = 0.798 → 0.80
rainy:  sqrt(16918 / (4 × 1828)) = sqrt(2.314) = 1.521 → 1.52
snowy:  sqrt(16918 / (4 × 1562)) = sqrt(2.708) = 1.646 → 1.65
sunny:  sqrt(16918 / (4 × 6888)) = sqrt(0.614) = 0.784 → 0.78

class_weights = [0.80, 1.52, 1.65, 0.78]   # order: cloudy, rainy, snowy, sunny
```

> 为什么选 sqrt 而不是 balanced：balanced 权重极差 4.2×，在我们数据上可能过拟合分布特征。sqrt 极差仅 2.1×，保留了 rainy/snowy > cloudy/sunny 的方向性，但数值更保守，换到比赛数据泛化风险更低。

## 共享配置

所有实验使用相同 baseline（不改动）：

```yaml
model: resnet18
image_size: 224
batch_size: 64
lr: 0.0001
optimizer: adamw
weight_decay: 0.0001
scheduler: cosine (warmup 3 epochs)
epochs: 50
seed: 42
augmentation: default (ColorJitter 0.15, RandomResizedCrop, HFlip, RandomRotation 10)
```

---

## exp_ce_baseline — CE 基线复核

**目的**：确认代码改动（Phase 0 的 RandAugment/MixUp/CutMix 接入、loss 参数传递）没有影响 baseline 行为。

**条件**：如果上一次 exp_001 之后改了 `trainer.py`、`transforms.py`、`losses.py` 或 `train.py`，必须重跑。

```bash
python scripts/train.py \
  --config configs/models/resnet18.yaml \
  --data_dir data/train \
  --output_dir outputs \
  --notes "Phase 1 baseline: CE + default augmentation, confirming no regressions"
```

**预期结果**：
- val macro F1 ≈ 0.87
- rainy F1 ≈ 0.82
- 与 exp_001 偏差 < ±0.01

**通过标准**：
- [ ] val macro F1 ≥ 0.865（与 exp_001 可比）
- [ ] 无训练异常（loss 不震荡、不 NaN）

---

## exp_focal — FocalLoss (γ=2.0, 无类权重)

**目的**：FocalLoss 自动降低 easy sample 的 loss 权重，让模型关注 rainy/snowy 等难分样本。不加类权，纯测 focal 机制。

```bash
python scripts/train.py \
  --config configs/models/resnet18.yaml \
  --data_dir data/train \
  --output_dir outputs \
  --notes "Phase 1: FocalLoss gamma=2.0, no class weights" \
  -- \
  --training.loss.name focal \
  --training.loss.focal_gamma 2.0
```

**预期效果**：
- rainy recall 提升（hard negative 被 focus）
- 总体 macro F1 可能持平或小幅提升
- snowy 可能轻微受益

**通过标准**：
- [ ] rainy F1 > 0.8240（baseline）
- [ ] val macro F1 不明显下降（≥ 0.86）
- [ ] 训练稳定

---

## exp_label_smoothing — LabelSmoothingCE (ε=0.1)

**目的**：LabelSmoothing 防止模型对 cloudy/sunny 过度自信，减少两者之间的假阳性误判。

```bash
python scripts/train.py \
  --config configs/models/resnet18.yaml \
  --data_dir data/train \
  --output_dir outputs \
  --notes "Phase 1: LabelSmoothing epsilon=0.1" \
  -- \
  --training.loss.name label_smoothing \
  --training.loss.label_smoothing 0.1
```

**预期效果**：
- cloudy F1 可能提升（减少误判为 sunny）
- rainy / snowy F1 可能持平（smoothing 不直接帮助少数类）
- 整体 macro F1 可能持平

**通过标准**：
- [ ] val macro F1 ≥ 0.87
- [ ] cloudy F1 不下降
- [ ] 不会导致 rainy 进一步恶化

---

## exp_weighted_ce — Weighted CE

**目的**：直接用类别权重补偿 rainy/snowy 的样本不足问题。使用 sqrt 权重（极差 2.1×），比 balanced 更保守，泛化风险更低。

> ⚠ 先运行一次 `detect_label_mapping` 确认 label 顺序，否则权重对错类会有反效果。

```bash
# 先确认 label 顺序
python -c "from data.label_mapping import detect_label_mapping; m=detect_label_mapping('data/train'); print(m.labels)"

# 假设输出 ['cloudy', 'rainy', 'snowy', 'sunny']
python scripts/train.py \
  --config configs/models/resnet18.yaml \
  --data_dir data/train \
  --output_dir outputs \
  --notes "Phase 1: Weighted CE, sqrt weights [0.80,1.52,1.65,0.78]" \
  -- \
  --training.loss.name cross_entropy \
  --training.loss.class_weights [0.80,1.52,1.65,0.78]
```

**预期效果**：
- rainy F1 应提升（每个 rainy 样本 loss × 1.52）
- snowy F1 应提升（每个 snowy 样本 loss × 1.65）
- cloudy/sunny 轻微影响（权重 0.80/0.78，接近 1.0 不会过度牺牲）
- 整体 macro F1 可能小幅提升

**通过标准**：
- [ ] rainy F1 > 0.8240
- [ ] snowy F1 ≥ 0.8927（不下降）
- [ ] cloudy/sunny F1 ≥ 0.85（不过度牺牲）

---

## exp_weighted_focal — Weighted FocalLoss（条件执行）

**条件**：仅在 #2 (Focal) 或 #4 (Weighted CE) 有明显收益时跑。

**目的**：Focal 机制 + 类别权重的叠加效果。

```bash
python scripts/train.py \
  --config configs/models/resnet18.yaml \
  --data_dir data/train \
  --output_dir outputs \
  --notes "Phase 1: Weighted FocalLoss gamma=2.0, sqrt weights" \
  -- \
  --training.loss.name focal \
  --training.loss.focal_gamma 2.0 \
  --training.loss.class_weights [0.80,1.52,1.65,0.78]
```

**通过标准**：
- [ ] 至少在 rainy F1 或 macro F1 上超过 CE baseline
- [ ] 不与 Weighted CE 或 FocalLoss 其中之一退化

---

## Phase 1 结束判定

全部必做实验完成后，填写：

| 指标 | Baseline | 最优 Loss | Δ |
|------|:---:|:---:|:---:|
| val macro F1 | 0.8708 | ? | |
| cloudy F1 | 0.8677 | ? | |
| rainy F1 | 0.8240 | ? | |
| snowy F1 | 0.8927 | ? | |
| sunny F1 | 0.8990 | ? | |

- [ ] 最优 loss 的 macro F1 ≥ 0.8708
- [ ] rainy F1 有提升或至少持平
- [ ] 训练曲线稳定，无过拟合信号
- [ ] 选出 Phase 2 的固定 loss

## 实验执行顺序

```
1. exp_ce_baseline  ← 先复核，确认环境/代码无回归
2. exp_focal
3. exp_label_smoothing
4. exp_weighted_ce
5. exp_weighted_focal  ← 条件执行
```

每个实验完成后在个人 todo 中记录结果，再决定是否进入下一个。
