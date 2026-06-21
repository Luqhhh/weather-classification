# 实验发现与后续启示

> 更新日期：2026-06-21  
> 依据：`experiments/leaderboard.md`、`outputs/exp_*/training_log.jsonl`、`outputs/exp_*/results.json`

## 当前阶段结论

这一阶段的实验已经基本收敛。继续扩大 dropout、weight decay、增强策略或 backbone 搜索的边际收益不高。当前最重要的发现是：**模型上限的提升主要来自权重平滑和少量互补 ensemble，而不是继续训练更多普通单 checkpoint 模型**。

当前候选优先级：

| 优先级 | Experiment | 类型 | Macro F1 | rainy F1 | 判断 |
|--------|------------|------|---------:|---------:|------|
| 1 | exp_051 | 单权重 checkpoint averaging | 0.9158 | 0.8966 | 默认主候选 |
| 2 | exp_050 | EMA 单权重 | 0.9155 | 0.8952 | 与 exp_051 互为备选 |
| 3 | exp_052 | 双模型 ensemble | 0.9159 | 0.9056 | rainy 最强，但推理成本更高 |
| 4 | exp_025 | 原始 baseline | 0.9106 | 0.8937 | 保留对照 |

`exp_052` 的 macro F1 最高，但只比 `exp_051` 高 0.0001，主要收益在 rainy F1。因此最终若 CPU 推理预算紧张，应优先考虑 `exp_051` 或 `exp_050`；若官方数据更看重 rainy 或 CPU 预算足够，再考虑 `exp_052`。

## 关键发现

### 1. 低 weight decay 单模型不稳定，但更适合 EMA/SWA

`exp_039` 使用 `dropout=0.3, weight_decay=5e-4`，单次 seed42 达到 0.9089，但多 seed 复验后均值只有：

```text
exp_039/048/049 macro F1 mean = 0.9073
std = 0.0012
rainy F1 mean = 0.8818
```

这说明低 weight decay 本身没有稳定超过 `exp_025`。真正的提升来自同一路线上的权重平滑：

| 实验 | 方法 | Macro F1 | rainy F1 |
|------|------|---------:|---------:|
| exp_039 | 普通单 checkpoint | 0.9089 | 0.8879 |
| exp_050 | EMA | 0.9155 | 0.8952 |
| exp_051 | top-3 checkpoint averaging | 0.9158 | 0.8966 |

启发：`wd=5e-4` 可能让模型在训练后期进入一个更适合平均的权重区域。单点 checkpoint 会受波动影响，但 EMA/SWA 能把训练轨迹中的多个好状态平滑成更稳的解。

### 2. checkpoint averaging 不是通用必涨

对 `exp_025` 做同样的 top-3 checkpoint averaging 得到 `exp_045`：

| 实验 | 方法 | Macro F1 | rainy F1 |
|------|------|---------:|---------:|
| exp_025 | 原始 best checkpoint | 0.9106 | 0.8937 |
| exp_045 | exp_025 top-3 averaging | 0.9102 | 0.8956 |

`exp_045` 没有超过 `exp_025`，说明 averaging 的收益不是机械必然发生。它更依赖训练轨迹本身是否处在同一个可平均的 basin 内，以及不同 checkpoint 的错误是否互补。

### 3. exp_051 的 top-3 组合优于只平均后期 checkpoint

`exp_051` 平均了 `exp_039` 的 epoch 2/5/6；`exp_053` 只平均 epoch 5/6：

| 实验 | 组合 | Macro F1 | rainy F1 |
|------|------|---------:|---------:|
| exp_051 | epoch 2/5/6 | 0.9158 | 0.8966 |
| exp_053 | epoch 5/6 | 0.9120 | 0.8896 |

启发：`exp_051` 的收益不只是“越后期越好”，而是多个验证表现较好的 checkpoint 在类别边界上有互补性。epoch 2 虽然单点 F1 低，但加入平均后可能修正了后期 checkpoint 的某些偏差。

### 4. LabelSmoothing 对 rainy 有价值，但不适合作为主模型

`exp_030` 的 rainy F1 达到 0.8995，是单 checkpoint 中 rainy 表现最突出的路线之一。但多 seed 后表现不稳定：

```text
exp_030/036/037 macro F1 mean ≈ 0.9072
rainy F1 mean ≈ 0.8873
```

这说明 LabelSmoothing 更适合作为 rainy 互补模型，而不是直接替代 CE 主线。`exp_052 = exp_051 + exp_030` 就验证了这一点：

| 实验 | Macro F1 | rainy F1 |
|------|---------:|---------:|
| exp_051 | 0.9158 | 0.8966 |
| exp_052 | 0.9159 | 0.9056 |

`exp_052` 几乎不提高 macro F1，但显著提高 rainy F1。它适合作为“rainy 优先”的备选方案。

### 5. dropout / weight decay 窄搜索已经不值得继续扩大

已测结果显示：

| 实验 | 配置 | Macro F1 | 结论 |
|------|------|---------:|------|
| exp_025 | d=0.3, wd=0.05 | 0.9106 | 原 baseline |
| exp_034/035 | d=0.2, wd=0.05 多 seed | 0.9054 / 0.9033 | 复验后下降 |
| exp_039/048/049 | d=0.3, wd=5e-4 多 seed | 0.9089 / 0.9065 / 0.9064 | 单点接近，均值不超 baseline |
| exp_040 | d=0.25, wd=5e-4 | 0.9043 | 不值得继续 |

继续扫 `dropout=0.25/0.35` 或更多 weight decay 点，预计收益低于 EMA/SWA/官方数据重训带来的收益。

## 对官方数据集发布后的启示

官方数据集发布后，第一件事不应该重新扩大搜索，而是验证当前发现是否迁移。

### 必须先做的验证

1. **复现当前主候选**
   - 用官方训练集重训 `exp_039` 配置。
   - 同时产出普通 best checkpoint、EMA checkpoint、top-k checkpoint averaging。
   - 对比 `plain vs EMA vs SWA` 是否仍保持 `EMA/SWA > plain`。

2. **重新确认验证集切分**
   - 用官方数据重新做 stratified split。
   - 保持 dedup / leakage 检查。
   - 不要直接沿用当前 val 上调出来的结论当最终答案。

3. **CPU benchmark 和 submission check**
   - 对 `exp_051` 单权重候选做 CPU benchmark。
   - 对 `exp_052` 双模型 ensemble 做 CPU benchmark。
   - 如果 `exp_052` 推理成本超预算，就只作为分析参考，不进入最终提交。

4. **保留一个干净 holdout**
   - 当前很多选择已经在同一个 validation set 上做过比较，存在轻微贴合当前 val 的风险。
   - 官方数据到位后，应留一份最终 holdout，只用于最后确认 `exp_050/051/052` 的排序。

## 官方数据集后的微调方向

### P0 - 推荐默认路线

优先复现这条路线：

```text
ConvNeXt-Tiny
image_size = 320
loss = CrossEntropy
dropout = 0.3
weight_decay = 5e-4
seed = 42
EMA decay = 0.999
checkpoint averaging = top-k validation macro F1 checkpoints
```

产物至少包括：

- 普通 best checkpoint
- EMA best checkpoint
- top-3 checkpoint averaged model
- CPU benchmark
- validation error samples

### P1 - 多 seed 确认 EMA/SWA 稳定性

当前 `exp_050/051` 是在 `exp_039` seed42 路线上做的 EMA/SWA。官方数据集上建议对下列配置做 3 seed：

| 路线 | Seed | 目的 |
|------|------|------|
| exp_039 config + EMA | 42 / 7 / 2026 | 验证 EMA 是否稳定 |
| exp_039 config + SWA/top-k averaging | 42 / 7 / 2026 | 验证 averaging 是否稳定 |

如果 EMA/SWA 的 mean F1 仍领先 plain checkpoint >= 0.002，就把权重平滑固定进最终训练流程。

### P2 - rainy 方向微调

如果官方数据中 rainy 仍是弱类，优先考虑这些低成本方向：

1. **保留 LabelSmoothing 作为 ensemble 补充分支**
   - 类似 `exp_052`，用 CE/SWA 主模型 + LS rainy 模型做 logits average。
   - 不建议直接把 LS 作为唯一主模型。

2. **调 ensemble 权重**
   - 先试 `0.8 * exp_051 + 0.2 * LS`
   - 再试 `0.7 * exp_051 + 0.3 * LS`
   - 目标是 rainy F1 提升，同时 macro F1 不下降超过 0.001。

3. **class-balanced sampler 作为后备**
   - 只有当 rainy recall 明显偏低时再实现。
   - sampler 可能提高 rainy，但也可能伤害 cloudy/sunny，需要小心控制。

### P3 - 不优先做的方向

下列方向当前证据不足，官方数据出来前后都不应优先投入：

- 继续大范围 sweep dropout。
- 继续尝试强增强、MixUp、RandAugment、强 rotation。
- 新增大量 backbone。
- 盲目提高输入尺寸到 384。
- 在没有独立 holdout 的情况下继续围绕当前 validation set 调 ensemble。

## 当前阶段推荐行动

1. 暂停新增大规模训练实验。
2. 补齐 `exp_050/051/052` 的 CPU benchmark 和 submission check。
3. 保留 `exp_051` 作为默认单模型候选。
4. 保留 `exp_052` 作为 rainy 优先、预算允许时的 ensemble 候选。
5. 等官方数据集发布后，优先复验“低 weight decay + EMA/SWA”是否仍成立。

