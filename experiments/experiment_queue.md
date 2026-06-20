# Experiment Queue — 剩余待做实验

> 更新：2026-06-20 | 5 个待做（A 组），C 组全部完成✅

## 已完成（exp_001 ~ exp_018）

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
| exp_025 | A | Size | ConvNeXt-Tiny 320 + CE | F1 0.9106 |
| exp_010 | B | Loss | LabelSmoothing ε=0.1 | F1 0.8966 ✅ 最优 |
| exp_011 | B | Loss | FocalLoss γ=2.0 | F1 0.8847 |
| exp_012 | B | Loss | Weighted CE (sqrt) | F1 0.8791 |
| exp_013 | B | Loss | Weighted CE (balanced) | F1 0.8841 |
| exp_014 | B | Aug | No Aug (LabelSmoothing) | F1 0.8948 |
| exp_015 | B | Aug | Light CJ (LabelSmoothing) | F1 0.8923 |
| exp_016 | B | Aug | Medium CJ (LabelSmoothing) | F1 0.8890 |
| exp_017 | B | Aug | Rotation 20° (LabelSmoothing) | F1 0.8864 |
| exp_018 | B | Aug | RandAugment (LabelSmoothing) | F1 0.8841 |
| exp_019 | C | Aug | MixUp α=0.2 + LabelSmoothing | F1 0.8912 |
| exp_020 | C | Aug | CutMix α=1.0 + LabelSmoothing | F1 0.8968 |
| exp_026 | C | Size | ConvNeXt-Tiny 384 + CE | F1 0.9000 |
| exp_027 | C | Dropout | CNX-Tiny 320 + CE + d=0.2 | F1 0.9097 |
| exp_028 | C | Dropout | CNX-Tiny 320 + CE + d=0.4 | F1 0.9035 |
| exp_029 | C | Dropout | CNX-Tiny 320 + CE + d=0.5 | F1 0.9075 |

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

## A 组 — Backbone 输入尺寸（exp_021 ~ exp_025，5 个，exp_025 已完成✅）

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
| **exp_025** | `convnext_tiny` | 320 | ✅ F1 0.9106, epoch 43, 116 min |

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

### B 组追加 — ConvNeXt LabelSmoothing（exp_030 ~ exp_031，2 个）

> 固定：ConvNeXt-Tiny + image_size=320 + dropout=0.3 + 默认增强；对照：exp_025（CE, d=0.3, size=320）

| ID | Loss | Dropout | 预期 |
|----|------|---------|------|
| **exp_030** | LabelSmoothing ε=0.05 | 0.3 | 温和抑制过拟合 |
| **exp_031** | LabelSmoothing ε=0.1 | 0.3 | 对照 ResNet 最优设置 |

Override 参数：

| ID | 参数 |
|----|------|
| exp_030 | `-- --data.image_size 320 --training.batch_size 32 --training.loss.name label_smoothing --training.loss.label_smoothing 0.05 --model.dropout 0.3` |
| exp_031 | `-- --data.image_size 320 --training.batch_size 32 --training.loss.name label_smoothing --training.loss.label_smoothing 0.1 --model.dropout 0.3` |

### B 组结果总结 (2026-06-20)

> 固定：ResNet-18 + LabelSmoothing ε=0.1 | 对照：exp_010（默认增强）F1 0.8966

| ID | Augmentation | Val F1 | rainy F1 | Δ vs exp_010 | Best Epoch | 结论 |
|----|-------------|--------|----------|--------------|------------|------|
| exp_010 | Default (CJ 0.15, Rot 10°) | **0.8966** 🥇 | 0.8649 | — | 6 | 最优：保守增强最适配天气任务 |
| exp_014 | No Aug | 0.8948 | 0.8585 | -0.2% | 14 | 增强贡献边际，主要用于防止过拟合 |
| exp_015 | Light CJ | 0.8923 | 0.8628 | -0.4% | 6 | 太保守，rainy 略高但整体差 |
| exp_016 | Medium CJ | 0.8890 | 0.8678 | -0.8% | 6 | 颜色扰动过强损害天气特征 |
| exp_017 | Rotation 20° | 0.8864 | 0.8484 | -1.0% | 5 | 旋转过强破坏方向性天气特征 |
| exp_018 | RandAugment | 0.8841 | 0.8510 | -1.3% | 4 | 自动化增强不如手工保守增强 |

**关键发现**：
1. **默认增强最优** — ColorJitter 0.15 + Rotation 10° 达到最佳平衡
2. **增强虽贡献<1%** — 相比 No Aug (exp_014: 0.8948)，但显著抑制过拟合（No Aug 需 14 epochs 才收敛 vs 默认 6 epochs）
3. **增强过强反而有害** — Rotation 20° (-1.0%) 和 RandAugment (-1.3%) 损害天气语义
4. **B 组结论**：保持默认增强参数，不建议调大 Rotation 或引入 RandAugment

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

### C 组结果总结 (2026-06-20)

> exp_019/020：ResNet-18 + LabelSmoothing ε=0.1 | 对照：exp_010（默认增强）F1 0.8966
> exp_026：ConvNeXt-Tiny 384 + CE | 对照：exp_008（CNX-224）F1 0.9071

| ID | Aug/Size | Val F1 | rainy F1 | Δ vs 对照 | Best Epoch | 结论 |
|----|----------|--------|----------|-----------|------------|------|
| exp_010 | Default (对照) | **0.8966** | 0.8649 | — | 6 | B 组最优 baseline |
| exp_019 | MixUp α=0.2 | 0.8912 | 0.858 | -0.54% | 23 | ❌ MixUp 反而降分，不利天气任务 |
| exp_020 | CutMix α=1.0 | **0.8968** 🥇 | 0.863 | +0.02% | 23 | ✅ 追平 baseline，保留局部纹理有效 |
| exp_008 | CNX-224 (对照) | **0.9071** | 0.886 | — | 6 | A 组 CNX 基线 |
| exp_026 | CNX-384 | 0.9000 | 0.869 | -0.71% | 8 | ❌ 384 不如 224，过拟合更严重 |

**关键发现**：
1. **CutMix > MixUp** — CutMix α=1.0 追平 baseline（+0.02%），MixUp 反而 -0.54%。CutMix 保留局部纹理有利于天气特征，MixUp 的全局插值模糊了天气视觉线索
2. **ConvNeXt-Tiny 384×384 不如 224×224** — 差距 -0.71%，更大输入导致更严重的过拟合（val loss 从 0.31 飙到 1.17）
3. **bs=32 在 384×384 会爆 8GB 显存** — 96% VRAM 导致训练质量劣化，bs=16 才正常（epoch 1 F1: 0.8818 vs 0.8670）
4. **C 组结论**：不推荐 MixUp；CutMix 无伤害但无显著收益；384 大输入无益。后续 exp_027~029 仍须等 A 组 best_size

### C 组 Dropout 结果 (2026-06-20)

> 固定：ConvNeXt-Tiny + CE + image_size=320 + batch_size=32 | 对照：exp_025 (d=0.3) F1 0.9106

| ID | Dropout | Val F1 | rainy F1 | Δ vs d=0.3 | Best Epoch | Train Time | 结论 |
|----|---------|--------|----------|------------|------------|------------|------|
| exp_025 | **0.3** | **0.9106** 🥇 | — | — | — | — | ✅ 最优，原始设置即最佳 |
| exp_027 | 0.2 | 0.9097 | 0.895 | -0.09% | 19 | 79.8 min | 接近但未超越，欠正则不致命 |
| exp_029 | 0.5 | 0.9075 | 0.895 | -0.31% | 9 | ~53 min | 强正则，rainy 追平 d=0.2 |
| exp_028 | 0.4 | 0.9035 | 0.881 | -0.71% | 5 | ~33 min | ❌ 过度正则化，收敛太快 |

**关键发现**：
1. **d=0.3 最优** — 原始设置就是 best，无需调整
2. **d=0.2 几乎持平** — 差仅 0.09%，320×320 下轻微欠正则也可接受
3. **d=0.4 最差** — F1 骤降 0.7%，epoch 5 即收敛，明显欠拟合
4. **d=0.5 意外回升** — 比 d=0.4 好，但不如 d=0.3；rainy 0.895 优秀
5. **C 组 dropout 结论**：保持 d=0.3，不建议调整

---

## 分组汇总

| 组 | 内容 | 数量 | ID | 状态 |
|----|------|------|-----|------|
| A | 输入尺寸 (B1×3 + CNX×3) | 5 | exp_021~025 | 🔶 exp_025 完成, 021~024 待做 |
| B | Core Augmentation | 5 | exp_014~018 | ✅ 完成 |
| C | 高级增强 (2) + CNX 精调 (4) | 6 | exp_019,020,026~029 | ✅ 全部完成 |
| **合计** | | **4 待做** | | |

## 执行顺序

```
阶段 1 — A 组待做（5 个）
  A: exp_021-025  ← 5 输入尺寸（B1×3 + CNX×2）

B 组 ✅ 全部完成（exp_014~018）
C 组 ✅ 全部完成（exp_019/020/026~029）
```
