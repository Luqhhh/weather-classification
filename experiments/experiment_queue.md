# Experiment Queue — ConvNeXt Generalization Plan

> 更新：2026-06-21 | 目标：确认 Top 配置稳定性，并继续优化泛化与 rainy 类表现

## 当前结论

| 配置 | 关键结果 | 判断 |
|------|----------|------|
| exp_025: ConvNeXt-Tiny 320 + CE + d=0.3 | macro F1 0.9106 | 当前第一，作为主 baseline |
| exp_027: ConvNeXt-Tiny 320 + CE + d=0.2 | macro F1 0.9097, rainy F1 0.895 | 与 exp_025 差距很小，需要多 seed 验证 |
| exp_030: ConvNeXt-Tiny 320 + LabelSmoothing 0.05 + d=0.3 | macro F1 0.9087, rainy F1 0.8995 | 总分略低但 rainy 最强，值得组合验证 |
| exp_031: ConvNeXt-Tiny 320 + LabelSmoothing 0.1 + d=0.3 | macro F1 0.9087 | 未超过 0.05，暂不优先 |
| exp_023: EfficientNet-B1 384 + CE | macro F1 0.9045 | 可作为轻量 ensemble 互补模型 |
| exp_024: ConvNeXt-Tiny 256 + CE | macro F1 0.9048 | 低于 320，不继续投入 256 路线 |

注意：`configs/models/convnext_tiny.yaml` 当前 `weight_decay=0.05`。下面的 `5e-4` 实验是**大幅降低 weight decay**，不是加强 weight decay；目的是验证当前 ConvNeXt 是否被过强 weight decay 限制。

## 统一运行约束

| 项 | 值 |
|----|----|
| Base config | `configs/models/convnext_tiny.yaml` |
| Image size | 320 |
| Batch size | 32 |
| Augmentation | 默认增强，不再加 RandAugment / MixUp / 强 Rotation |
| Device | `auto`，最终仍需 CPU benchmark |
| Primary metric | validation macro F1 |
| Secondary metric | rainy F1、val loss、early stop epoch、CPU 推理时间 |

命令模板：

```bash
python3 scripts/train.py \
  --config configs/models/convnext_tiny.yaml \
  --output_dir outputs \
  --experiment_id EXP_ID \
  --notes "NOTES" \
  -- \
  --logging.experiment_name EXP_ID \
  --data.image_size 320 \
  --training.batch_size 32 \
  ...
```

`--experiment_id` 负责写入 tracking schema，`--logging.experiment_name` 负责固定输出目录为 `outputs/EXP_ID`。

---

## P1 — 多 seed 复验 Top 配置

目的：当前 Top 结果差距小于 0.2%，先确认排序不是随机波动。

执行策略：复用已有 seed 42 结果作为第一个 seed；新增 seed 7 和 seed 2026。每个候选配置最终用 3 个 seed 统计 mean / std。

| ID | 复验对象 | Seed | 参数 | 结论标准 |
|----|----------|------|------|----------|
| exp_032 | exp_025 | 7 | CE, d=0.3, wd=0.05 | 与 seed 42 共同计算均值 |
| exp_033 | exp_025 | 2026 | CE, d=0.3, wd=0.05 | 与 seed 42 共同计算均值 |
| exp_034 | exp_027 | 7 | CE, d=0.2, wd=0.05 | 检查 d=0.2 是否稳定超过 d=0.3 |
| exp_035 | exp_027 | 2026 | CE, d=0.2, wd=0.05 | 检查 d=0.2 是否稳定超过 d=0.3 |
| exp_036 | exp_030 | 7 | LabelSmoothing 0.05, d=0.3, wd=0.05 | 检查 rainy 优势是否稳定 |
| exp_037 | exp_030 | 2026 | LabelSmoothing 0.05, d=0.3, wd=0.05 | 检查 rainy 优势是否稳定 |

Override 参数：

| ID | 参数 |
|----|------|
| exp_032 | `--seed 7 --model.dropout 0.3 --training.loss.name cross_entropy --training.optimizer.weight_decay 0.05` |
| exp_033 | `--seed 2026 --model.dropout 0.3 --training.loss.name cross_entropy --training.optimizer.weight_decay 0.05` |
| exp_034 | `--seed 7 --model.dropout 0.2 --training.loss.name cross_entropy --training.optimizer.weight_decay 0.05` |
| exp_035 | `--seed 2026 --model.dropout 0.2 --training.loss.name cross_entropy --training.optimizer.weight_decay 0.05` |
| exp_036 | `--seed 7 --model.dropout 0.3 --training.loss.name label_smoothing --training.loss.label_smoothing 0.05 --training.optimizer.weight_decay 0.05` |
| exp_037 | `--seed 2026 --model.dropout 0.3 --training.loss.name label_smoothing --training.loss.label_smoothing 0.05 --training.optimizer.weight_decay 0.05` |

判定：

- 若某配置 mean F1 领先 >= 0.002，直接进入下一阶段主线。
- 若 mean F1 接近但 rainy F1 明显更高，保留作 ensemble 或 class-balanced 路线。
- 若 std >= 0.003，后续所有结论必须用多 seed，不再用单次 F1 排名。

---

## P2 — ConvNeXt 320 窄范围正则化

目的：在 `dropout=0.2/0.3` 附近做窄搜索，同时测试当前 `weight_decay=0.05` 是否过强。

| ID | 实验 | 参数 | 预期 |
|----|------|------|------|
| exp_038 | Dropout 中点 | d=0.25, wd=0.05, CE | 判断 0.2 与 0.3 之间是否有更稳点 |
| exp_039 | 降低 weight decay | d=0.3, wd=5e-4, CE | 测试当前 0.05 是否压制学习 |
| exp_040 | Dropout 中点 + 降低 weight decay | d=0.25, wd=5e-4, CE | 观察低 wd 是否需要更弱 dropout 配合 |

Override 参数：

| ID | 参数 |
|----|------|
| exp_038 | `--seed 42 --model.dropout 0.25 --training.loss.name cross_entropy --training.optimizer.weight_decay 0.05` |
| exp_039 | `--seed 42 --model.dropout 0.3 --training.loss.name cross_entropy --training.optimizer.weight_decay 0.0005` |
| exp_040 | `--seed 42 --model.dropout 0.25 --training.loss.name cross_entropy --training.optimizer.weight_decay 0.0005` |

判定：

- 若 `wd=5e-4` 的 val loss 明显下降且 F1 上升，后续以低 wd 为主线。
- 若 `wd=5e-4` 训练 loss 下降但 val F1 不升，说明 0.05 的强正则仍有价值。
- 若 `d=0.25` 超过 `d=0.2/0.3` 的 seed 均值，再进入多 seed 复验。

---

## P3 — 保 rainy 的 LabelSmoothing / Balanced 路线

目的：`exp_030` rainy F1 最高，但 macro F1 略低。尝试保住 rainy 增益，同时不牺牲整体排序。

| ID | 实验 | 参数 | 状态 |
|----|------|------|------|
| exp_041 | Dropout 0.2 + LabelSmoothing 0.05 | d=0.2, LS=0.05, wd=0.05 | 可直接运行 |
| exp_042 | CE + class-balanced sampler | d=0.3, CE, balanced sampler | 需先实现 sampler |
| exp_043 | LS 0.05 + class-balanced sampler | d=0.2, LS=0.05, balanced sampler | 需先实现 sampler |

Override 参数：

| ID | 参数 |
|----|------|
| exp_041 | `--seed 42 --model.dropout 0.2 --training.loss.name label_smoothing --training.loss.label_smoothing 0.05 --training.optimizer.weight_decay 0.05` |
| exp_042 | `--seed 42 --model.dropout 0.3 --training.loss.name cross_entropy --training.sampler.name class_balanced` |
| exp_043 | `--seed 42 --model.dropout 0.2 --training.loss.name label_smoothing --training.loss.label_smoothing 0.05 --training.sampler.name class_balanced` |

实现备注：

- 当前训练代码支持 `training.loss.class_weights`，但没有发现 `class-balanced sampler` 配置。
- 若短期不想改 dataloader，可先用 `--training.loss.class_weights [...]` 做 proxy；但它不等价于 sampler。
- sampler 路线只在 rainy recall / rainy F1 明显提升且 macro F1 不掉超过 0.001 时继续。

---

## P4 — EMA / SWA 泛化实验

目的：`exp_025` train loss 很低、val loss 很高，但 F1 仍第一。EMA / SWA 更适合先试，不再加重数据增强。

| ID | 实验 | 参数 | 状态 |
|----|------|------|------|
| infra_001 | 增加 EMA 支持 | `training.ema.enabled`, `training.ema.decay` | 需先实现 |
| exp_044 | EMA | 基于 exp_025，decay=0.999 | 依赖 infra_001 |
| exp_045 | SWA / checkpoint averaging | 平均后期或 top-k checkpoints | 可优先做离线脚本 |

建议实现顺序：

1. 先做离线 checkpoint averaging：读取 `outputs/exp_025/checkpoints/` 的 top-k checkpoint，平均权重后评估。
2. 如果 checkpoint averaging 有收益，再把 EMA 写进训练循环。
3. EMA/SWA 必须同时记录 macro F1、rainy F1、val loss 和 CPU 推理时间。

判定：

- F1 提升 >= 0.001 且 val loss 下降，进入多 seed 复验。
- 只降 val loss 但 F1 不升，保留为泛化备选，不替换主模型。

---

## P5 — 轻量 Ensemble

目的：在不继续训练大量模型的情况下，测试互补性和最终提交上限。

| ID | Ensemble | 组成 | 状态 |
|----|----------|------|------|
| infra_002 | 增加 ensemble evaluate 脚本 | 支持多个 checkpoint logits 平均 | 需先实现 |
| exp_046 | ConvNeXt CE + ConvNeXt LS | exp_025 + exp_030 | 依赖 infra_002 |
| exp_047 | ConvNeXt + EfficientNet-B1 | exp_025 + exp_023 | 依赖 infra_002 |

判定：

- ensemble F1 提升 >= 0.002 才值得进入最终方案。
- CPU 推理时间必须重新测；若超过预算，只保留为分析参考。
- 若 `exp_030` 只提升 rainy 但拖累其他类，可尝试按类别或按权重调整 logits：`0.7 * exp_025 + 0.3 * exp_030`。

---

## 执行顺序

```text
1. 先把 exp_024 / exp_030 / exp_031 结果纳入 leaderboard 和 results.csv
2. 跑 P1 多 seed 复验：exp_032~037
3. 若 P1 排名稳定，跑 P2：exp_038~040
4. 跑 P3 中可直接运行的 exp_041
5. 若仍有提升空间，再实现 sampler / EMA / ensemble：
   - infra_001 + exp_044/045
   - infra_002 + exp_046/047
```

## 当前待做清单

| 优先级 | ID | 状态 |
|--------|----|------|
| P0 | 汇总 exp_024 / exp_030 / exp_031 | 待做 |
| P1 | exp_032~037 | 待跑 |
| P2 | exp_038~040 | 待跑 |
| P3 | exp_041 | 待跑 |
| P3+ | exp_042~043 | 需实现 sampler |
| P4 | exp_044~045 | 需实现 EMA/SWA 或 checkpoint averaging |
| P5 | exp_046~047 | 需实现 ensemble evaluate |
