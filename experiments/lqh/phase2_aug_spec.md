# Phase 2 — Augmentation 对比实验 Spec

> 目标：固定 Phase 1 最优 loss，只比较增强策略对 macro F1 和少数类的贡献。
> 前提：Phase 1 已选出最优 loss（当前 FocalLoss γ=2.0 领先，0.8847 vs 0.8708）。
> 原则：每次只改 augmentation，不改 loss / model / image_size / seed。

## 为什么 Phase 2 单独跑

Loss 改的是「模型应该关注什么」，Augmentation 改的是「模型看到什么」。两者通过不同路径影响泛化：

| 维度 | Loss | Augmentation |
|---|---|---|
| 作用位置 | 梯度反向传播 | 数据输入层 |
| 解决的瓶颈 | 类不平衡、难易样本 | 训练分布多样性 |
| 对 rainy 的帮助 | 放大梯度信号 | 模拟更多 rainy 变体 |
| 泛化风险 | 低（不依赖具体像素） | 中（可能破坏天气语义） |

需要在固定 loss 的前提下隔离 augmentation 的效果，否则无法归因。

---

## 实验矩阵

| # | ID | Augmentation | 参数 | 预期效果 |
|---|-----|-------------|------|----------|
| 1 | exp_aug_baseline | Default | ColorJitter 0.15, Rotation 10, HFlip, RRC | Phase 1 对照（即 exp_focal） |
| 2 | exp_aug_none | No Aug | Resize only | 隔离 loss 贡献，测纯 loss vs loss+aug |
| 3 | exp_aug_light | Light ColorJitter | brightness 0.05, contrast 0.05, saturation 0.05, hue 0.02 | 更保守的颜色扰动 |
| 4 | exp_aug_medium | Medium ColorJitter | brightness 0.25, contrast 0.25, saturation 0.25, hue 0.08 | 更强的颜色扰动 |
| 5 | exp_aug_rot20 | Rotation 20° | RandomRotation 20（默认是 10） | 更大旋转角 |
| 6 | exp_aug_randaug | RandAugment | num_ops=2, magnitude=9 | 自动增强策略 |
| 7 | exp_aug_mixup | MixUp | α=0.2 | 样本间平滑插值 |
| 8 | exp_aug_cutmix | CutMix | α=1.0 | 空间局部替换 |

## 共享配置

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
loss: {Phase 1 最优}     # ← 从 Phase 1 结果填入
```

---

## exp_aug_baseline — 默认增强（Phase 1 对照）

**目的**：这就是 Phase 1 的 exp_focal，直接复用结果，不需要重跑。

**配置**：默认增强，即 ColorJitter(0.15), RandomRotation(10), RandomHorizontalFlip(0.5), RandomResizedCrop(0.7-1.0)

**通过标准**：作为其他实验的对照基线。

---

## exp_aug_none — 无增强

**目的**：隔离 loss 的贡献。如果 FocalLoss + NoAug 已经接近 FocalLoss + DefaultAug，说明当前增强对 rainy 帮助很小，可以往更激进方向试。

```bash
python3 scripts/train.py \
  --config configs/models/resnet18.yaml \
  --data_dir data/train \
  --output_dir outputs \
  --notes "Phase 2: {best_loss} + No augmentation" \
  -- \
  --data.augmentation.random_resized_crop.scale [1.0,1.0] \
  --data.augmentation.random_horizontal_flip.prob 0.0 \
  --data.augmentation.random_rotation.degrees 0 \
  --data.augmentation.color_jitter '{"brightness":0.0,"contrast":0.0,"saturation":0.0,"hue":0.0}'
```

**预期**：F1 ↓ 0.5-1.5%，过拟合更快出现。参考 exp_002（CE+NoAug 降了 0.9%）。

---

## exp_aug_light — Light ColorJitter

**目的**：天气分类依赖颜色/光照语义，默认 0.15 的 ColorJitter 是否已经过强？测一个更保守的变体。

```bash
python3 scripts/train.py \
  --config configs/models/resnet18.yaml \
  --data_dir data/train \
  --output_dir outputs \
  --notes "Phase 2: {best_loss} + Light ColorJitter" \
  -- \
  --data.augmentation.color_jitter.brightness 0.05 \
  --data.augmentation.color_jitter.contrast 0.05 \
  --data.augmentation.color_jitter.saturation 0.05 \
  --data.augmentation.color_jitter.hue 0.02
```

**预期**：cloudy/sunny 可能受益（颜色语义保留更好），rainy 可能持平。

---

## exp_aug_medium — Medium ColorJitter

**目的**：如果 light 没有退化，测更强的颜色扰动能否提升泛化。

```bash
python3 scripts/train.py \
  --config configs/models/resnet18.yaml \
  --data_dir data/train \
  --output_dir outputs \
  --notes "Phase 2: {best_loss} + Medium ColorJitter" \
  -- \
  --data.augmentation.color_jitter.brightness 0.25 \
  --data.augmentation.color_jitter.contrast 0.25 \
  --data.augmentation.color_jitter.saturation 0.25 \
  --data.augmentation.color_jitter.hue 0.08
```

**预期**：可能提升 rainy（不同光照下的 rainy 更丰富），但 cloudy/sunny 可能退化。

---

## exp_aug_rot20 — Rotation 20°

**目的**：天气照片可能有多种拍摄角度。默认 10° 可能不够，测 20°。

```bash
python3 scripts/train.py \
  --config configs/models/resnet18.yaml \
  --data_dir data/train \
  --output_dir outputs \
  --notes "Phase 2: {best_loss} + Rotation 20" \
  -- \
  --data.augmentation.random_rotation.degrees 20
```

**预期**：小幅提升或持平。极端旋转（>30°）会翻转天空/地面位置，破坏天气语义；20° 在安全范围内。

---

## exp_aug_randaug — RandAugment

**目的**：RandAugment 随机组合多种增强操作。优势是不需手调单项参数，让模型接触更丰富的变体。

```bash
python3 scripts/train.py \
  --config configs/models/resnet18.yaml \
  --data_dir data/train \
  --output_dir outputs \
  --notes "Phase 2: {best_loss} + RandAugment (2 ops, mag 9)" \
  -- \
  --data.augmentation.rand_augment '{"num_ops":2,"magnitude":9}'
```

**预期**：可能提升 rainy（更多变体 → 更难过拟合到少数类的特定模式），但 magnitude 9 可能对 cloudy/sunny 过强。如果退化，后续测 mag=5。

---

## exp_aug_mixup — MixUp (α=0.2)

**目的**：MixUp 在两个样本之间做像素插值，强迫模型学习平滑的类别边界。对 rainy-vs-cloudy 这类模糊边界可能有益。

```bash
python3 scripts/train.py \
  --config configs/models/resnet18.yaml \
  --data_dir data/train \
  --output_dir outputs \
  --notes "Phase 2: {best_loss} + MixUp alpha=0.2" \
  -- \
  --data.augmentation.mixup_alpha 0.2
```

**预期**：rainy 边界样本的校准可能改善（减少 rainy→cloudy 误判）。α=0.2 保守起步，避免过度平滑。

---

## exp_aug_cutmix — CutMix (α=1.0)

**目的**：CutMix 保留局部空间结构（不像 MixUp 全局混合），更贴合天气分类中「局部纹理」（雨滴、雪花、云层）的重要性。

```bash
python3 scripts/train.py \
  --config configs/models/resnet18.yaml \
  --data_dir data/train \
  --output_dir outputs \
  --notes "Phase 2: {best_loss} + CutMix alpha=1.0" \
  -- \
  --data.augmentation.cutmix_alpha 1.0
```

**预期**：对 rainy/snowy 可能更友好（保留了雨雪纹理的局部性），但需要验证是否会导致 cloudy/sunny 大面积区域被替换后语义破坏。

---

## 实验执行顺序

```
1. exp_aug_none      ← 先测 loss-only 的底，理解增强的边际贡献
2. exp_aug_light     ← 从保守往激进方向探索
3. exp_aug_medium
4. exp_aug_rot20
5. exp_aug_randaug   ← 自动化增强
6. exp_aug_mixup     ← 样本间增强（新的正则化维度）
7. exp_aug_cutmix
```

**exp_aug_baseline 不重跑，复用 Phase 1 最优 loss 的结果。**

---

## 实验记录模板

```markdown
## exp_aug_xxx: {best_loss} + {aug_name}

- date:
- branch:
- commit:
- command:
- loss: {from Phase 1}
- augmentation:

### Results

| Metric | Value | vs Baseline (exp_aug_baseline) |
| --- | --- | --- |
| val_macro_f1 | | |
| cloudy_f1 | | |
| rainy_f1 | | |
| snowy_f1 | | |
| sunny_f1 | | |
| best_epoch | | |

### Observations
- rainy gain / loss:
- cloudy/sunny tradeoff:
- overfitting signal (val loss trend):
- conclusion:
```

---

## Phase 2 结束判定

全部实验完成后填写：

| 指标 | Default Aug | 最优 Aug | Δ |
|------|:---:|:---:|:---:|
| val macro F1 | ? | ? | |
| cloudy F1 | ? | ? | |
| rainy F1 | ? | ? | |
| snowy F1 | ? | ? | |
| sunny F1 | ? | ? | |

- [ ] 增强是否提高 val macro F1
- [ ] rainy / snowy 是否提升，而不是只提升大类
- [ ] 训练/验证 loss 是否稳定
- [ ] cloudy / sunny 没有被明显牺牲
- [ ] 选出 Phase 3 的最优 aug 策略
