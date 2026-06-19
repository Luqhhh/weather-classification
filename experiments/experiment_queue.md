# Experiment Queue — 剩余待做实验

> 更新：2026-06-19 | 16 个待做，分 A/B/C 三组

## 已完成（exp_001 ~ exp_012）

| ID | Owner | Category | 描述 | 结果 |
|----|-------|----------|------|------|
| exp_001 | A | Backbone | ResNet-18 + CE baseline | F1 0.8708 |
| exp_002 | A | Backbone | ResNet-18 + CE + NoAug | F1 0.8618 |
| exp_003 | A | Backbone | ResNet-34 + CE | F1 0.9007 |
| exp_004 | A | Backbone | EfficientNet-B0 + CE | F1 0.8963 |
| exp_005 | A | Backbone | EfficientNet-B1 + CE | F1 0.9014 |
| exp_007 | A | Backbone | MobileNetV3-Small + CE | F1 0.8173 ❌ |
| exp_008 | A | Backbone | ConvNeXt-Tiny + CE | F1 0.9071 |
| exp_009 | A | Backbone | ResNet-50 + CE | F1 0.8916 |
| exp_010 | B | Loss | LabelSmoothing ε=0.1 | F1 0.8966 ✅ 最优 |
| exp_011 | B | Loss | FocalLoss γ=2.0 | F1 0.8847 |
| exp_012 | B | Loss | Weighted CE (sqrt) | F1 0.8791 |
| exp_013 | B | Loss | Weighted CE (balanced) | F1 0.8841 |

## 运行中

_无_

---

## 共享凝固参数（三组通用）

```
seed: 42           optimizer: adamw         lr: 0.0001
weight_decay: 0.0001   scheduler: cosine (warmup 3)   epochs: 50
early_stop: patience=10, min_delta=0.001
```

---

## A 组 — Backbone 输入尺寸（exp_021 ~ exp_025，5 个）

| 参数 | 值 |
|------|-----|
| Config | `configs/models/efficientnet_b1.yaml` 或 `configs/models/convnext_tiny.yaml` |
| Loss | CE（cross_entropy） |
| Augmentation | 默认（ColorJitter 0.15, RRC 0.7-1.0, HFlip 0.5, Rotation 10°） |
| Batch size | B1=16 / CNX=32（显存限制） |
| Dropout | 0.3 |
| 对照 | exp_005 (B1-224), exp_008 (CNX-224) 复用不重跑 |

| ID | Config | Image Size | 预期 |
|----|--------|------------|------|
| **exp_021** | `efficientnet_b1` | 256 | 原生 240，256 提升感受野 |
| **exp_022** | `efficientnet_b1` | 320 | 更大输入，泛化可能提升 |
| **exp_023** | `efficientnet_b1` | 384 | 最大尺寸，关注 CPU 时间 |
| **exp_024** | `convnext_tiny` | 256 | 原生 224，细节提升 |
| **exp_025** | `convnext_tiny` | 320 | 更大感受野 |

Override 参数：`-- --data.image_size {256/320/384}`

---

## B 组 — Core Augmentation（exp_014 ~ exp_018，5 个）

| 参数 | 值 |
|------|-----|
| Config | `configs/models/resnet18.yaml` |
| Loss | **LabelSmoothing ε=0.1** |
| Model | ResNet-18, pretrained, dropout=0.3 |
| Batch size | 64 |
| 对照 | exp_010（默认增强）复用不重跑 |

> 详细：`experiments/lqh/phase2_aug_spec.md`

| ID | 实验 | 改动 | 预期 |
|----|------|------|------|
| **exp_014** | No Aug | 关闭所有增强：RRC scale=[1.0,1.0], HFlip prob=0, rotation=0, CJ all=0 | 测增强对 LabelSmoothing 的边际贡献 |
| **exp_015** | Light CJ | brightness 0.05, contrast 0.05, saturation 0.05, hue 0.02 | 保守，保留颜色语义 |
| **exp_016** | Medium CJ | 0.25 / 0.25 / 0.25 / 0.08 | 更强颜色扰动 |
| **exp_017** | Rotation 20° | rotation 10 → 20 | 更大角度旋转 |
| **exp_018** | RandAugment | `rand_augment: {num_ops: 2, magnitude: 9}` | 自动化增强策略 |

Override 参数：

| ID | 参数 |
|----|------|
| exp_014 | `--data.augmentation.random_resized_crop.scale [1.0,1.0] --data.augmentation.random_horizontal_flip.prob 0.0 --data.augmentation.random_rotation.degrees 0 --data.augmentation.color_jitter '{"brightness":0.0,"contrast":0.0,"saturation":0.0,"hue":0.0}'` |
| exp_015 | `--data.augmentation.color_jitter.brightness 0.05 --data.augmentation.color_jitter.contrast 0.05 --data.augmentation.color_jitter.saturation 0.05 --data.augmentation.color_jitter.hue 0.02` |
| exp_016 | `--data.augmentation.color_jitter.brightness 0.25 --data.augmentation.color_jitter.contrast 0.25 --data.augmentation.color_jitter.saturation 0.25 --data.augmentation.color_jitter.hue 0.08` |
| exp_017 | `--data.augmentation.random_rotation.degrees 20` |
| exp_018 | `--data.augmentation.rand_augment '{"num_ops":2,"magnitude":9}'` |

所有 B 组实验共用：`-- --training.loss.name label_smoothing --training.loss.label_smoothing 0.1` + 上表参数

---

## C 组 — 高级增强 + Backbone 精调（exp_019, 020, 026~029，6 个）

| 参数 | 值 |
|------|-----|
| Loss (019/020) | **LabelSmoothing ε=0.1** + ResNet-18 |
| Loss (026~029) | CE + ConvNeXt-Tiny（A 组结果确定最优 size 后） |
| Aug (019/020) | 默认增强 + MixUp/CutMix |
| 对照 (019/020) | exp_010（默认增强无 MixUp） |
| 对照 (026~029) | A 组最优 ConvNeXt + d=0.3 |

| ID | 实验 | 关键参数 | 预期 |
|----|------|----------|------|
| **exp_019** | MixUp | `mixup_alpha: 0.2` | 样本间平滑插值，改善 rainy 边界 |
| **exp_020** | CutMix | `cutmix_alpha: 1.0` | 空间局部替换，保留纹理 |
| **exp_026** | CNX + 384 | ConvNeXt-Tiny, image_size=384, dropout=0.3 | 最大输入，过拟合风险 |
| **exp_027** | Dropout 0.2 | ConvNeXt + best_size, dropout=0.2 | 轻正则 |
| **exp_028** | Dropout 0.4 | ConvNeXt + best_size, dropout=0.4 | 中等正则 |
| **exp_029** | Dropout 0.5 | ConvNeXt + best_size, dropout=0.5 | 重正则，针对过拟合 |

Override 参数：

| ID | 参数 |
|----|------|
| exp_019 | `-- --training.loss.name label_smoothing --training.loss.label_smoothing 0.1 --data.augmentation.mixup_alpha 0.2` |
| exp_020 | `-- --training.loss.name label_smoothing --training.loss.label_smoothing 0.1 --data.augmentation.cutmix_alpha 1.0` |
| exp_026 | `--training.batch_size 32 -- --data.image_size 384` |
| exp_027-029 | `--training.batch_size 32 -- --data.image_size {best_size} --model.dropout {0.2/0.4/0.5}` |

---

## 分组汇总

| 组 | 内容 | 数量 | ID | 依赖 |
|----|------|------|-----|------|
| A | 输入尺寸 (B1×3 + CNX×2) | 5 | exp_021~025 | 无 |
| B | Core Augmentation | 5 | exp_014~018 | 无 |
| C | 高级增强 (2) + CNX 精调 (4) | 6 | exp_019,020,026~029 | 026~029 依赖 A 结果 |
| **合计** | | **16** | | |

## 执行顺序

```
阶段 1 — 全部可并行（13 个，无依赖）
  B: exp_014-018  ← 5 augmentation（ResNet-18 + LabelSmoothing）
  C: exp_019-020  ← MixUp + CutMix（ResNet-18 + LabelSmoothing）
  A: exp_021-026  ← 6 输入尺寸（B1×3 + CNX×3）

阶段 2 — 依赖 A 阶段 1 结果（3 个）
  C: exp_027-029  ← ConvNeXt dropout 0.2 / 0.4 / 0.5
```
