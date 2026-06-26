# Official Leaderboard — Weather Image Classification

> 数据来源：`experiments/officialTestScore.md` 为平台 test ledger；本地 holdout 来自 `outputs/<exp>/eval_holdout/results.json` 或已记录实验结果。
> 更新日期：2026-06-26。平台已测表按 official test macro F1 排序；pending 实验不混入平台排名。

## 平台 Test 已测总榜

| # | Experiment | 方向 | Local holdout | Official test | Test - Holdout | 推理成本 | 判断 |
|---|------------|------|--------------:|--------------:|---------------:|----------|------|
| 1 | official_013 | `0.7*009 + 0.3*004` logits ensemble | 0.9524 | **0.9456** | -0.0068 | 2x ConvNeXt | 当前平台第一 |
| 2 | official_018 | 224 warm-start from `exp_044`, EMA | 0.9423 | **0.9446** | +0.0023 | 1x ConvNeXt | 当前最强单模型，和 013 很接近 |
| 3 | official_012 | `0.8*009 + 0.2*004` logits ensemble | 0.9499 | **0.9424** | -0.0075 | 2x ConvNeXt | ensemble 有效，但低于 013 |
| 4 | official_010 | official_009 top-3 checkpoint averaging | 0.9412 | **0.9387** | -0.0025 | 1x ConvNeXt | 平台小幅高于 009 |
| 5 | official_009 | 224 warm-start from `exp_054`, seed=42 | 0.9412 | **0.9373** | -0.0039 | 1x ConvNeXt | warm-start 基线 |
| 6 | official_019 | 256 warm-start from `exp_054` | 0.9540 | 0.9359 | -0.0181 | 1x ConvNeXt | 本地高分不迁移 |
| 7 | official_004 | official_002 top-3 checkpoint averaging | 0.9315 | 0.9315 | 0.0000 | 1x ConvNeXt | 单独不强，但对 013 有互补 |
| 8 | official_011 | official_007 top-3 checkpoint averaging | 0.9507 | 0.9280 | -0.0227 | 1x ConvNeXt | 本地 holdout 明显高估 |
| 9 | official_022 | progressive resize 256->320 | 0.9572 | 0.9253 | -0.0319 | 1x ConvNeXt | 本地最高但平台翻车 |
| 10 | official_007 | 224 ImageNet init, EMA | 0.9456 | 0.9252 | -0.0204 | 1x ConvNeXt | 非 warm-start 平台不稳 |
| 11 | official_008 | 320 ImageNet init, EMA | 0.9399 | 0.9196 | -0.0203 | 1x ConvNeXt | 320 未带来平台收益 |
| 12 | official_005 | 320, wd=5e-4, EMA | 0.9142 | 0.9188 | +0.0046 | 1x ConvNeXt | 分数低，不作主线 |
| 13 | official_003 | 320, standard aug, EMA | 0.9246 | 0.9158 | -0.0088 | 1x ConvNeXt | 标准增强不稳 |
| 14 | official_001 | 320 baseline, EMA | 0.9376 | 0.9147 | -0.0229 | 1x ConvNeXt | 平台弱 |
| 15 | official_002 | 320 baseline, no EMA | 0.9287 | 0.9059 | -0.0228 | 1x ConvNeXt | 不作主线 |
| 16 | official_006 | 320 low-wd, no EMA | 0.9291 | 0.9034 | -0.0257 | 1x ConvNeXt | 不作主线 |

## 平台待测优先级

| 提交优先级 | Experiment | 配置 | Local holdout | 推理成本 | 统一权重文件 | 判断 |
|------------|------------|------|--------------:|----------|--------------|------|
| P0 | official_030 | official_013 logits + cloudy `-0.5` class bias | **0.9541** | 2x ConvNeXt | `outputs/official_030/official_030_best_model.pth` | 唯一正向后处理信号；先测，但有 holdout 拟合风险 |
| P1 | official_025 | `0.7*official_018 + 0.3*official_004` | 0.9439 | 2x ConvNeXt | `outputs/official_025/official_025_best_model.pth` | 验证用平台更强的 018 替换 009 后，004 是否仍互补 |
| P2 | official_031 | official_018 original + horizontal flip TTA | 0.9431 | 2x TTA | `outputs/official_031/official_031_best_model.pth` | 轻量 TTA 对照，平台时间允许再测 |
| P3 | official_024 | official_018 top-3 checkpoint averaging | 0.9423 | 1x ConvNeXt | `outputs/official_024/official_024_best_model.pth` | 1x 成本，但本地未超过 018 |
| Skip | official_026 | `0.8*official_018 + 0.2*official_004` | 0.9423 | 2x ConvNeXt | `outputs/official_026/official_026_best_model.pth` | 被 025 支配 |
| Skip | official_027 | `0.5*018 + 0.35*009 + 0.15*004` | 0.9418 | 3x ConvNeXt | `outputs/official_027/official_027_best_model.pth` | 成本高，且本地低于 025/031 |
| Skip | official_029 | 027-style ensemble + temperature scaling | 0.9418 | 3x ConvNeXt | `outputs/official_029/official_029_best_model.pth` | temperature calibration 未带来本地收益 |

旧 pending 项：`official_014` 和 `official_023` 在 `officialTestScore.md` 里仍无平台分数；当前不优先于 P0/P1/P2。

## 本地 Holdout 参考榜

本表只用于筛选，不代表平台排序。`official_022/019/011` 已证明本地高分可能严重不迁移。

| # | Experiment | 方向 | Local holdout | Official test | 备注 |
|---|------------|------|--------------:|--------------:|------|
| 1 | official_022 | progressive resize 256->320 | **0.9572** | 0.9253 | 本地最高但平台翻车 |
| 2 | official_030 | class-bias calibrated official_013 | **0.9541** | pending | 本轮最值得平台小步验证 |
| 3 | official_019 | 256 warm-start | **0.9540** | 0.9359 | 本地高分不迁移 |
| 4 | official_013 | 009/004 logits ensemble | **0.9524** | **0.9456** | 当前平台第一 |
| 5 | official_011 | 007 top-3 averaging | **0.9507** | 0.9280 | holdout 高估 |
| 6 | official_012 | 009/004 logits ensemble | 0.9499 | 0.9424 | 低于 013 |
| 7 | official_020 | ConvNeXt-Small capacity probe | 0.9470 | — | 大模型未作为主线 |
| 8 | official_023 | size 288 probe | 0.9461 | pending | 过拟合，旧 pending |
| 9 | official_007 | 224 ImageNet init, EMA | 0.9456 | 0.9252 | 平台不稳 |
| 10 | official_025 | 018/004 logits ensemble | 0.9439 | pending | P1 平台待测 |
| 11 | official_031 | official_018 hflip TTA | 0.9431 | pending | P2 平台待测 |
| 12 | official_018 | 224 warm-start from `exp_044` | 0.9423 | 0.9446 | 最强单模型平台候选 |
| 13 | official_024 | official_018 top-3 averaging | 0.9423 | pending | averaging 未提升 |
| 14 | official_026 | 018/004 conservative ensemble | 0.9423 | pending | 被 025 支配 |
| 15 | official_027 | 018/009/004 ensemble | 0.9418 | pending | 3x 成本，不优先 |
| 16 | official_029 | temperature calibrated ensemble | 0.9418 | pending | 未提升 |
| 17 | official_009 | 224 warm-start from `exp_054` | 0.9412 | 0.9373 | warm-start 基线 |
| 18 | official_010 | official_009 top-3 averaging | 0.9412 | 0.9387 | 平台小幅高于 009 |
| 19 | official_014 | warm-start seed=7 | 0.9385 | pending | 低于 018/013 |
| 20 | official_017 | warm-start no EMA | 0.9316 | — | no EMA 不优先 |
| 21 | official_015 | warm-start seed=2026 | 0.9248 | — | seed 方差偏低 |
| 22 | official_016 | warm-start lr=1e-5 | 0.9179 | — | 低 lr 不如 3e-5 |
| 23 | official_017_avg | official_017 top-3 averaging | 0.9144 | — | averaging 翻车 |

## 子榜 A — 平台已验证主线

| # | Experiment | 方向 | Official test | Local holdout | 结论 |
|---|------------|------|--------------:|--------------:|------|
| 1 | official_013 | 009/004 logits ensemble | **0.9456** | 0.9524 | 平台最高，但 2x 成本 |
| 2 | official_018 | 224 warm-start from `exp_044` | **0.9446** | 0.9423 | 最强 1x 候选 |
| 3 | official_012 | 009/004 logits ensemble | **0.9424** | 0.9499 | ensemble 有效但低于 013 |
| 4 | official_010 | 009 top-3 averaging | **0.9387** | 0.9412 | averaging 在平台小幅超过 009 |
| 5 | official_009 | 224 warm-start from `exp_054` | **0.9373** | 0.9412 | 已被 010/012/013/018 超过 |

## 子榜 B — 新后处理与提交候选

| # | Experiment | 方向 | Local holdout | vs official_018 holdout | vs official_013 holdout | 平台动作 |
|---|------------|------|--------------:|------------------------:|------------------------:|----------|
| 1 | official_030 | class bias on official_013 | **0.9541** | +0.0118 | +0.0017 | P0，先测 |
| 2 | official_025 | `0.7*018 + 0.3*004` | 0.9439 | +0.0016 | -0.0085 | P1，可测 |
| 3 | official_031 | official_018 hflip TTA | 0.9431 | +0.0008 | -0.0093 | P2，可测 |
| 4 | official_024 | official_018 checkpoint averaging | 0.9423 | +0.0000 | -0.0101 | P3，低优先 |
| 5 | official_026 | `0.8*018 + 0.2*004` | 0.9423 | +0.0000 | -0.0101 | Skip |
| 6 | official_027 | `0.5*018 + 0.35*009 + 0.15*004` | 0.9418 | -0.0005 | -0.0106 | Skip |
| 7 | official_029 | temperature calibrated 027-style ensemble | 0.9418 | -0.0005 | -0.0106 | Skip |

## 子榜 C — Warm-start Seed / LR / EMA 对照

| Experiment | 配置 | Val F1 | Holdout F1 | Official test | 判断 |
|------------|------|-------:|-----------:|--------------:|------|
| official_009 | seed=42, EMA, lr=3e-5, `exp_054` warm-start | 0.9509 | 0.9412 | 0.9373 | 原 warm-start 基线 |
| official_014 | seed=7, EMA | 0.9473 | 0.9385 | pending | seed=7 稳定但低于 009/018 |
| official_015 | seed=2026, EMA | 0.9431 | 0.9248 | — | 不优先 |
| official_016 | lr=1e-5, EMA | 0.9399 | 0.9179 | — | lr=1e-5 不如 3e-5 |
| official_017 | no EMA | 0.9463 | 0.9316 | — | no EMA 不如 EMA |
| official_017_avg | no EMA top-3 averaging | — | 0.9144 | — | averaging 翻车 |
| official_018 | seed=42, EMA, lr=3e-5, `exp_044` warm-start | 0.9467 | 0.9423 | 0.9446 | 当前最强单模型 |

结论：继续随机 seed 碰运气价值低；`official_018` 的 warm-start 来源比 `official_009` 更适合作为单模型主线。

## 子榜 D — 分辨率 / Backbone 探索

| Experiment | 配置 | Val F1 | Holdout F1 | Official test | 判断 |
|------------|------|-------:|-----------:|--------------:|------|
| official_018 | 224 ConvNeXt-Tiny | 0.9467 | 0.9423 | **0.9446** | 平台最稳 |
| official_019 | 256 ConvNeXt-Tiny | 0.9543 | 0.9540 | 0.9359 | 本地不迁移 |
| official_020 | 224 ConvNeXt-Small | 0.9457 | 0.9470 | — | 更大 backbone 未显示必要性 |
| official_022 | 256->320 progressive resize | 0.9551 | 0.9572 | 0.9253 | 平台翻车，停止 resize 主线 |
| official_023 | 288 ConvNeXt-Tiny | 0.9568 | 0.9461 | pending | 旧 pending，不优先 |

结论：本阶段不继续扩大分辨率或 backbone；平台反馈更支持 224 warm-start 与轻量 ensemble。

## 子榜 E — Averaging / Ensemble

| Experiment | 类型 | 组成 | Local holdout | Official test | 结论 |
|------------|------|------|--------------:|--------------:|------|
| official_004 | checkpoint averaging | official_002 top-3 | 0.9315 | 0.9315 | 单独不强，但作为 013 互补分支有效 |
| official_010 | checkpoint averaging | official_009 top-3 | 0.9412 | 0.9387 | 平台小幅超过 009 |
| official_011 | checkpoint averaging | official_007 top-3 | 0.9507 | 0.9280 | holdout 高估 |
| official_012 | logits ensemble | `0.8*009 + 0.2*004` | 0.9499 | 0.9424 | 有效但低于 013 |
| official_013 | logits ensemble | `0.7*009 + 0.3*004` | 0.9524 | **0.9456** | 当前平台第一 |
| official_024 | checkpoint averaging | official_018 top-3 | 0.9423 | pending | 未提升 018 |
| official_025 | logits ensemble | `0.7*018 + 0.3*004` | 0.9439 | pending | 平台可测但本地低于 013 |
| official_030 | class-bias ensemble | official_013 + cloudy `-0.5` | 0.9541 | pending | 本地最高，需平台验证是否过拟合 |

## 当前结论

1. **平台正式分数第一仍是 `official_013`**：0.9456，成本是 2x ConvNeXt。
2. **`official_018` 是最强 1x 候选**：平台 0.9446，仅比 013 低约 0.0011。
3. **本地 holdout 不能单独排序决策**：`official_022/019/011` 本地很高，但平台明显掉分。
4. **本轮新增只建议优先测 `official_030`**：它是唯一超过 013 holdout 的后处理，但 bias 由 holdout 搜索得到，平台风险明确。
5. **`official_025/031/024` 是次级验证**：分别验证 018+004 互补、flip TTA、018 averaging；`026/027/029` 暂不提交。

## Artifact 说明

- 统一平台提交权重命名：`outputs/official_xxx/official_xxx_best_model.pth`。
- `.pth` 大权重被 `.gitignore` 忽略，不进入 Git；Git 中保留 `official_xxx_bundle.json`、`results.json`、`predictions.csv`、`error_samples.csv` 和混淆矩阵。
- 012/013/025/026/027/029/030/031 是 submission bundle 或推理时后处理，不是新训练出的单一 ConvNeXt `state_dict`。
