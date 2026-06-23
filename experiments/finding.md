# 实验发现与后续启示

> 更新日期：2026-06-23
> 依据：`experiments/leaderboard.md`、`outputs/exp_*/training_log.jsonl`、`outputs/exp_*/results.json`

## 当前阶段结论

这一阶段的实验已经基本收敛。继续扩大 dropout、weight decay 或 backbone 搜索的边际收益不高。当前最重要的发现是：**模型上限的提升主要来自权重平滑，以及更贴合天气语义的保守增强，而不是继续训练更多普通单 checkpoint 模型**。最新结果显示，去掉 rotation、减轻 color jitter 后再加 EMA 的 `exp_054` 暂时是最强单模型。

当前候选优先级：

| 优先级 | Experiment | 类型                        | Macro F1 | rainy F1 | 判断                       |
| ------ | ---------- | --------------------------- | -------: | -------: | -------------------------- |
| 1      | exp_054    | EMA 单权重 + 天气敏感增强   |   0.9204 |   0.9025 | 默认主候选                 |
| 2      | exp_044    | EMA 单权重                  |   0.9182 |   0.8972 | 原主候选，保留对照         |
| 3      | exp_052    | 双模型 ensemble             |   0.9159 |   0.9056 | rainy 最强，但推理成本更高 |
| 4      | exp_051    | 单权重 checkpoint averaging |   0.9158 |   0.8966 | 单权重 averaging 备选      |
| 5      | exp_050    | EMA 单权重                  |   0.9155 |   0.8952 | 低 wd EMA 备选             |

`exp_054` 当前 macro F1 最高，而且仍是 1x ConvNeXt 推理成本，应作为默认单模型候选。`exp_052` 的主要价值在 rainy F1，比 `exp_054` 高 0.0031，但 macro F1 低 0.0045 且需要约 2x ConvNeXt 推理成本；若官方数据更看重 rainy 或 CPU 预算足够，再考虑 ensemble。

## 关键发现

### 1. 权重平滑是主要收益来源，EMA 暂时优于继续普通单点搜索

`exp_025/032/033` 复验了原始 `dropout=0.3, weight_decay=0.05` baseline，3 个 seed 的 macro F1 mean/std 为 0.9074/0.0028，rainy F1 mean 为 0.8821。`exp_039` 使用 `dropout=0.3, weight_decay=5e-4`，单次 seed42 达到 0.9089，但多 seed 复验后均值只有：

```text
exp_039/048/049 macro F1 mean = 0.9073
std = 0.0012
rainy F1 mean = 0.8818
```

这说明低 weight decay 本身没有稳定超过 `exp_025/032/033` 的原始 baseline 均值。真正的提升来自权重平滑，且目前 `exp_025` 原配置上的 EMA 表现最好：

| 实验    | 方法                             | Macro F1 | rainy F1 |
| ------- | -------------------------------- | -------: | -------: |
| exp_025 | 普通 best checkpoint             |   0.9106 |   0.8937 |
| exp_044 | exp_025 config + EMA             |   0.9182 |   0.8972 |
| exp_054 | exp_044 + no rotation / light CJ |   0.9204 |   0.9025 |
| exp_039 | 低 wd 普通 checkpoint            |   0.9089 |   0.8879 |
| exp_050 | 低 wd + EMA                      |   0.9155 |   0.8952 |
| exp_051 | 低 wd top-3 checkpoint averaging |   0.9158 |   0.8966 |

启发：单点 checkpoint 会受训练后期波动影响，EMA/SWA 能把训练轨迹中的多个好状态平滑成更稳的解。但当前证据并不支持直接把低 weight decay 作为默认训练配置；`wd=0.05 + EMA + 天气敏感增强` 的 `exp_054` 更适合作为下一轮复验起点。

### 2. checkpoint averaging 不是通用必涨

对 `exp_025` 做同样的 top-3 checkpoint averaging 得到 `exp_045`：

| 实验    | 方法                    | Macro F1 | rainy F1 |
| ------- | ----------------------- | -------: | -------: |
| exp_025 | 原始 best checkpoint    |   0.9106 |   0.8937 |
| exp_045 | exp_025 top-3 averaging |   0.9102 |   0.8956 |

`exp_045` 没有超过 `exp_025`，说明 averaging 的收益不是机械必然发生。它更依赖训练轨迹本身是否处在同一个可平均的 basin 内，以及不同 checkpoint 的错误是否互补。

### 3. exp_051 的 top-3 组合优于只平均后期 checkpoint

`exp_051` 平均了 `exp_039` 的 epoch 2/5/6；`exp_053` 只平均 epoch 5/6：

| 实验    | 组合        | Macro F1 | rainy F1 |
| ------- | ----------- | -------: | -------: |
| exp_051 | epoch 2/5/6 |   0.9158 |   0.8966 |
| exp_053 | epoch 5/6   |   0.9120 |   0.8896 |

启发：`exp_051` 的收益不只是“越后期越好”，而是多个验证表现较好的 checkpoint 在类别边界上有互补性。epoch 2 虽然单点 F1 低，但加入平均后可能修正了后期 checkpoint 的某些偏差。

### 4. LabelSmoothing 对 rainy 有价值，但不适合作为主模型

`exp_030` 的 rainy F1 达到 0.8995，是单 checkpoint 中 rainy 表现最突出的路线之一。但多 seed 后表现不稳定：

```text
exp_030/036/037 macro F1 mean ≈ 0.9072
rainy F1 mean ≈ 0.8873
```

这说明 LabelSmoothing 更适合作为 rainy 互补模型，而不是直接替代 CE 主线。`exp_052 = exp_051 + exp_030` 就验证了这一点：

| 实验    | Macro F1 | rainy F1 |
| ------- | -------: | -------: |
| exp_051 |   0.9158 |   0.8966 |
| exp_052 |   0.9159 |   0.9056 |

`exp_052` 几乎不提高 macro F1，但显著提高 rainy F1。它适合作为“rainy 优先”的备选方案。

### 5. dropout / weight decay 窄搜索已经不值得继续扩大

已测结果显示：

| 实验            | 配置                   |                 Macro F1 | 结论                        |
| --------------- | ---------------------- | -----------------------: | --------------------------- |
| exp_025         | d=0.3, wd=0.05         |                   0.9106 | 原 baseline                 |
| exp_032/033     | d=0.3, wd=0.05 多 seed |          0.9053 / 0.9064 | seed42 高点未稳定复现       |
| exp_034/035     | d=0.2, wd=0.05 多 seed |          0.9054 / 0.9033 | 复验后下降                  |
| exp_038         | d=0.25, wd=0.05        |                   0.9046 | 不优于 d=0.2 或 d=0.3 seed42 |
| exp_039/048/049 | d=0.3, wd=5e-4 多 seed | 0.9089 / 0.9065 / 0.9064 | 单点接近，均值不超 baseline |
| exp_040         | d=0.25, wd=5e-4        |                   0.9043 | 不值得继续                  |

继续扫 `dropout=0.25/0.35` 或更多 weight decay 点，预计收益低于 EMA/SWA/官方数据重训带来的收益。

### 6. 天气敏感增强是新增正结果，cutout / FixRes / TTA 暂不进主线

外部天气识别经验提示强几何增强可能破坏天气语义。本轮补充实验直接验证了这一点：在 ConvNeXt 320 + CE + EMA 主线里去掉 rotation，并把 color jitter 从 0.15 级别降到 0.08 / hue 0.03 后，`exp_054` 超过了原 EMA 主候选。

| 实验    | 配置                         | Macro F1 | rainy F1 | 结论 |
| ------- | ---------------------------- | -------: | -------: | ---- |
| exp_044 | 默认增强 + EMA               |   0.9182 |   0.8972 | 原单模型第一 |
| exp_054 | no rotation + light CJ + EMA |   0.9204 |   0.9025 | 当前单模型第一 |
| exp_055 | exp_054 + small cutout       |   0.9173 |   0.9013 | cutout 未带来收益 |
| exp_056 | exp_054 -> 384 FixRes 微调   |   0.9170 |   0.9012 | 384 head/norm 微调未补足收益 |
| exp_057 | exp_054 单模型 CPU TTA       |   0.8982 |   0.8666 | center-crop TTA 明显负收益 |

启发：

- 去掉 rotation 是当前少数明确的正向增强改动，应保留到官方数据复验。
- 小块 cutout 虽然没有明显破坏 rainy，但拖累 overall，不进入主线。
- FixRes 只微调 head / norm layers 不足以利用 384 分辨率，且 val loss 较高；不优先继续 384。
- 当前 center-crop + horizontal flip TTA 改变了验证预处理分布，macro F1 掉到 0.8982，即使 CPU 约 93.1 ms/image，也没有提交价值。

### 7. confidence / 手工特征诊断说明 ensemble 主要影响低置信度边界样本

`exp_058` 对比了 `exp_051` 单权重 averaging 和 `exp_052 = exp_051 + exp_030` ensemble。结果显示 ensemble 并不是无条件修正错误：

```text
base wrong -> ensemble correct: 71 samples, base confidence mean = 0.651
base correct -> ensemble wrong: 80 samples, base confidence mean = 0.665
both wrong: 293 samples, base confidence mean = 0.813
```

这说明 ensemble 的主要影响区间确实是低/中置信度边界样本，但当前组合修正和回退数量接近，净准确率没有提升。`exp_052` 的价值仍主要是 rainy F1，而不是 overall macro F1。

`exp_059` 的手工特征统计也符合直觉：snowy 样本平均亮度更高、饱和度更低、白色比例更高；sunny 样本饱和度最高。正确和错误样本在这些粗特征上的均值差异很小，说明当前错误不太可能只靠简单亮度/白色比例规则修正。

## 对官方数据集发布后的启示

官方数据集发布后，第一件事不应该重新扩大搜索，而是验证当前发现是否迁移。

### 必须先做的验证

1. **复现当前主候选**
   - 用官方训练集重训 `exp_054` 配置。
   - 同时产出普通 best checkpoint、EMA checkpoint、top-k checkpoint averaging。
   - 对比 `plain vs EMA vs SWA` 是否仍保持 `EMA/SWA > plain`。

2. **重新确认验证集切分**
   - 用官方数据重新做 stratified split。
   - 保持 dedup / leakage 检查。
   - 不要直接沿用当前 val 上调出来的结论当最终答案。

3. **CPU benchmark 和 submission check**
   - 对 `exp_054` 默认单权重候选做 CPU benchmark。
   - 对 `exp_044` 原 EMA 候选保留对照 benchmark。
   - 对 `exp_050/051` 单权重备选做 CPU benchmark。
   - 对 `exp_052` 双模型 ensemble 做 CPU benchmark。
   - 如果 `exp_052` 推理成本超预算，就只作为分析参考，不进入最终提交。

4. **保留一个干净 holdout**
   - 当前很多选择已经在同一个 validation set 上做过比较，存在轻微贴合当前 val 的风险。
   - 官方数据到位后，应留一份最终 holdout，只用于最后确认 `exp_054/044/050/051/052` 的排序。

## 官方数据集后的微调方向

### P0 - 推荐默认路线

优先复现这条路线：

```text
ConvNeXt-Tiny
image_size = 320
loss = CrossEntropy
dropout = 0.3
weight_decay = 0.05
seed = 42
EMA decay = 0.999
augmentation = no rotation + light color jitter
```

产物至少包括：

- 普通 best checkpoint
- EMA best checkpoint
- top-3 checkpoint averaged model（作为对照，不默认替代 EMA）
- CPU benchmark
- validation error samples

### P1 - 多 seed 确认 EMA/SWA 稳定性

当前 `exp_054` 是 seed42 上的单次结果；`exp_050/051` 是在 `exp_039` seed42 路线上做的 EMA/SWA。官方数据集上建议对下列配置做 3 seed：

| 路线                                 | Seed          | 目的                     |
| ------------------------------------ | ------------- | ------------------------ |
| exp_054 config + EMA                 | 42 / 7 / 2026 | 验证当前默认候选是否稳定 |
| exp_044 config + EMA                 | 42 / 7 / 2026 | 验证去 rotation / light CJ 的收益是否稳定 |
| exp_039 config + EMA                 | 42 / 7 / 2026 | 验证 EMA 是否稳定        |
| exp_039 config + SWA/top-k averaging | 42 / 7 / 2026 | 验证 averaging 是否稳定  |

如果 EMA/SWA 的 mean F1 仍领先 plain checkpoint >= 0.002，就把权重平滑固定进最终训练流程；若 `wd=0.05 + EMA + no rotation/light CJ` 仍领先低 wd 路线，就不再把低 weight decay 作为默认方案。

### P2 - rainy 方向微调

如果官方数据中 rainy 仍是弱类，优先考虑这些低成本方向：

1. **保留 LabelSmoothing 作为 ensemble 补充分支**
   - 类似 `exp_052`，用 CE/EMA 主模型 + LS rainy 模型做 logits average。
   - 不建议直接把 LS 作为唯一主模型。

2. **调 ensemble 权重**
   - 先试 `0.8 * exp_054 + 0.2 * LS`
   - 再试 `0.7 * exp_054 + 0.3 * LS`
   - 保留 `exp_052 = exp_051 + exp_030` 作为已验证 rainy 对照。
   - 目标是 rainy F1 提升，同时 macro F1 不下降超过 0.001。

3. **class-balanced sampler 作为后备**
   - `exp_042` 已实现并完成训练，macro F1 0.9059，rainy F1 0.8918。
   - 结果没有超过 `exp_025` 或 `exp_030`，暂不继续 `exp_043`。
   - 只有当官方数据中 rainy recall 明显偏低时，再重新考虑 sampler 或采样权重。

### P3 - 不优先做的方向

下列方向当前证据不足，官方数据出来前后都不应优先投入：

- 继续大范围 sweep dropout。
- 继续尝试强增强、MixUp、RandAugment、强 rotation。
- 新增大量 backbone。
- 盲目提高输入尺寸到 384。
- 继续当前 center-crop TTA 路线。
- 在没有独立 holdout 的情况下继续围绕当前 validation set 调 ensemble。

## 当前阶段推荐行动

1. 暂停新增大规模训练实验。
2. 补齐 `exp_054/044/050/051/052` 的 CPU benchmark 和 submission check。
3. 保留 `exp_054` 作为默认单模型候选。
4. 保留 `exp_052` 作为 rainy 优先、预算允许时的 ensemble 候选。
5. 等官方数据集发布后，优先复验“天气敏感增强 + EMA/SWA 权重平滑”是否仍成立，并比较 `wd=0.05` 与 `wd=5e-4` 两条路线。
