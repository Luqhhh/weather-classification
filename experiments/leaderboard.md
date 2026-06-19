# Experiments Leaderboard — Weather Image Classification

> 更新日期：2026-06-18 | 排序：val_macro_f1 ↓

## 排行榜

| # | Experiment | Model | Loss | Image Size | Aug | Val F1 | rainy F1 | Best Epoch | CPU Time | Weight |
|---|-----------|-------|------|------------|-----|--------|----------|------------|----------|--------|
| 1 | **exp_008** | ConvNeXt-Tiny | CE | 224 | ✅ | **0.9071** 🥇 | 0.886 | 6 | 1.7 min ✅ | 106.1 MB |
| 2 | **exp_005** | EfficientNet-B1 | CE | 224 | ✅ | **0.9014** 🥈 | 0.885 | 9 | 1.1 min ✅ | 24.9 MB |
| 3 | **exp_003** | ResNet-34 | CE | 224 | ✅ | **0.9007** | 0.880 | 40 | 1.2 min ✅ | 81.2 MB |
| 4 | **exp_004** | EfficientNet-B0 | CE | 224 | ✅ | **0.8963** | 0.865 | 8 | 0.8 min ✅ | 15.3 MB |
| 5 | **exp_009** | ResNet-50 | CE | 224 | ✅ | **0.8916** | 0.855 | 5 | 1.7 min ✅ | 89.7 MB |
| 6 | **exp_001** | ResNet-18 | CE | 224 | ✅ | **0.8708** | 0.8240 | 5 | 0.5 min ✅ | 42.6 MB |
| 7 | **exp_002** | ResNet-18 | CE | 224 | ❌ | **0.8618** | 0.8231 | 7 | — | 42.6 MB |
| 8 | **exp_007** | MobileNetV3-Small | CE | 224 | ✅ | **0.8173** ❌ | 0.752 | 1 | 0.3 min | 3.5 MB |

---

## exp_001: ResNet-18 + CE + 224 (Baseline)

### 训练配置

```yaml
experiment_id: exp_001
date: 2026-06-17
model: resnet18
config: configs/models/resnet18.yaml

# 训练参数
epochs: 5 (early stop at best)
batch_size: 64
learning_rate: 0.0001
optimizer: adamw
weight_decay: 0.0001
scheduler: cosine (warmup 3 epochs)
loss: cross_entropy
augmentation: default (ColorJitter 0.15, RandomResizedCrop, HFlip)
dropout: 0.3

# 数据
train_images: 13535
val_images: 3383
num_classes: 4 (cloudy, rainy, snowy, sunny)
seed: 42
```

### 训练曲线

| Epoch | Val F1 | Val Acc | Val Loss | Time | Best? |
|-------|--------|---------|----------|------|-------|
| 1 | 0.8442 | 0.8637 | 0.4676 | 360s | |
| 2 | 0.8245 | 0.8436 | 0.4758 | 359s | |
| 3 | 0.8624 | 0.8688 | 0.3854 | 358s | |
| 4 | 0.8726 | 0.8871 | 0.3496 | 351s | ⭐ (train) |
| **5** | **0.8735** | **0.8835** | **0.3507** | **348s** | **⭐ Best** |

### 验证集结果

```
Macro F1:  0.8708
Accuracy:  0.8784

Per-class:
  cloudy   Precision 0.8473  Recall 0.8892  F1 0.8677  (N=1660)
  rainy    Precision 0.9077  Recall 0.7544  F1 0.8240  (N=456)
  snowy    Precision 0.9118  Recall 0.8744  F1 0.8927  (N=390)
  sunny    Precision 0.8961  Recall 0.9019  F1 0.8990  (N=1722)

⚠ Weak classes: cloudy, rainy
```

### 混淆矩阵

![confusion_matrix](../reports/confusion_matrix.png)

### CPU 性能

| Batch Size | Per-Image (ms) | Throughput (imgs/s) |
|------------|---------------|---------------------|
| 1 | 8.91 | 112.2 |
| 4 | 6.38 | 156.8 |
| **8** | **5.38** | **185.7** ⭐ |
| 16 | 5.87 | 170.4 |
| 32 | 6.98 | 143.3 |
| 64 | 7.45 | 134.1 |

```
Model: resnet18 (11.18M params, 42.6 MB)
Optimal batch: 8
3000 images: 0.5 min ✅ (70 min limit)
```

### 关键发现

1. **rainy 是瓶颈**：F1 仅 0.82，召回率 0.75——最容易误分为 cloudy
2. **类别不平衡显著**：rainy (1828) / snowy (1562) vs cloudy (6640) / sunny (6888)
3. **5 个 epoch 即接近收敛**：F1 从 epoch 1→4 提升 0.03，之后趋于平稳
4. **CPU 推理无压力**：0.5 min / 3000 张，远超 70min 限制
5. **WSL2 训练不稳定**：num_workers=0 + OMP_NUM_THREADS 控制线程数是稳定训练的关键

---

## exp_002: ResNet-18 + CE + 224 (No Augmentation)

### 训练配置

```yaml
experiment_id: exp_002
date: 2026-06-17
model: resnet18
augmentation: none (scale=1.0, no flip, no rotation, no color jitter)
其余配置同 exp_001
```

### 训练曲线

| Epoch | Val F1 | Acc | Loss | Time | Best? |
|-------|--------|-----|------|------|-------|
| 1 | 0.8637 | 0.8767 | 0.3603 | 496s | |
| 2 | 0.8537 | 0.8682 | 0.4093 | 493s | |
| 3 | 0.8570 | 0.8696 | 0.5091 | 592s | |
| 4 | 0.8588 | 0.8682 | 0.5688 | 586s | |
| 5 | 0.8598 | 0.8699 | 0.6004 | 493s | |
| 6 | 0.8593 | 0.8714 | 0.6383 | 480s | |
| **7** | **0.8769** | **0.8850** | **0.6867** | **487s** | **⭐ Best** |
| 8 | 0.8637 | 0.8753 | 0.7303 | 482s | |
| 14 | 0.8726 | 0.8815 | 0.7953 | — | |
| 17 | 0.8689 | 0.8782 | 0.8736 | — | Early Stop |

### 验证集结果（evaluate.py, data/val）

```
Macro F1:  0.8618
Accuracy:  0.8690

Per-class:
  cloudy   Precision 0.8516  Recall 0.8608  F1 0.8562
  rainy    Precision 0.8704  Recall 0.7807  F1 0.8231
  snowy    Precision 0.8769  Recall 0.8769  F1 0.8769
  sunny    Precision 0.8835  Recall 0.8984  F1 0.8909
```

### 对比总结

| 指标 | Augmented (exp_001) | No-Aug (exp_002) | Δ |
|------|:---:|:---:|:---:|
| Val F1 | 0.8708 | 0.8618 | **+0.9%** |
| rainy F1 | 0.8240 | 0.8231 | +0.1% |
| Loss 轨迹 | 稳定 (0.35~0.47) | 持续攀升 (0.36→0.87) | 过拟合 |
| 收敛速度 | epoch 5 | epoch 7 | 慢 2 epoch |

### 结论

增强贡献约 **+0.9% F1**，不算大但显著。主要价值是**抑制过拟合**（loss 稳定 vs 持续攀升）和**加速收敛**（快 2 epoch）。rainy 的瓶颈不是增强能解决的——两个实验 rainy F1 几乎一样，必须靠 FocalLoss 或类别权重。

---
## Phase 1 总结 (2026-06-19)

**Top 2 进入 Phase 2：ConvNeXt-Tiny + EfficientNet-B1**

| Exp | Model | Val F1 | rainy | snowy | cloudy | sunny | CPU(3000) | Weight | Batch | Epoch | Notes |
|-----|-------|--------|-------|-------|--------|-------|-----------|--------|-------|-------|-------|
| 008 | ConvNeXt-Tiny | 0.9071 | 0.886 | 0.932 | 0.893 | 0.917 | 1.7 min | 106 MB | 32 | 6 | 严重过拟合,需提 dropout |
| 005 | EfficientNet-B1 | 0.9014 | 0.885 | 0.925 | 0.884 | 0.911 | 1.1 min | 25 MB | 16 | 9 | 64/32 均 OOM |
| 003 | ResNet-34 | 0.9007 | 0.880 | — | — | — | 1.2 min | 81 MB | 64 | 40 | |
| 004 | EfficientNet-B0 | 0.8963 | 0.865 | — | — | — | 0.8 min | 15 MB | 64 | 8 | |
| 001 | ResNet-18 | 0.8708 | 0.824 | 0.893 | 0.868 | 0.899 | 0.5 min | 43 MB | 64 | 5 | Baseline |
| 007 | MobileNetV3-Small | 0.8173 | 0.752 | 0.787 | — | — | 0.3 min | 4 MB | 64 | 1 | ❌ 淘汰; segfault |
| — | EfficientNet-B2 | — | — | — | — | — | — | — | — | — | ⏭️ 跳过 |

### 关键发现

1. **ConvNeXt-Tiny 最强但过拟合严重** — loss 0.38→0.78，Phase 3 优先试 dropout 0.5
2. **EfficientNet-B1 性价比最优** — F1 差 0.006，权重小 4×，CPU 快 35%
3. **B0→B1 提升显著** — +0.005 F1, rainy +0.02
4. **MobileNetV3 不可用** — F1<0.83 + segfault 崩溃

### 下一步

- [ ] Phase 2: Top 2 × 4 种输入尺寸
- [ ] Phase 3: ConvNeXt dropout 优先 0.5
- [ ] 如有余力补跑 B2

---

## 实验模板 (Ctrl+C 复制)

```markdown
## exp_XXX: Model + Loss + ImageSize

### 训练配置
- date: YYYY-MM-DD
- model: xxx
- epochs: N
- batch_size: N
- lr: 0.xxxx
- loss: xxx
- augmentation: xxx

### 结果
- Val F1: 0.xxxx
- Per-class: cloudy x.xxx / rainy x.xxx / snowy x.xxx / sunny x.xxx
- CPU time (3000): x.x min

### 备注
- xxx
```
