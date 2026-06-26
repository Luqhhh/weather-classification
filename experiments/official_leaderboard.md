# Official Leaderboard — Weather Image Classification

> 数据来源：`experiments/officialTestScore.md` 为平台 test ledger；本地 holdout 来自 `outputs/<exp>/eval_holdout/results.json` 或已记录实验结果。
> 更新日期：2026-06-26。平台已测表按 official test macro F1 排序；pending 实验不混入平台排名。

## 平台 Test 已测总榜

| # | Experiment | 方向 | Local holdout | Official test | Test - Holdout | 推理成本 | 判断 |
|---|------------|------|--------------:|--------------:|---------------:|----------|------|
| 1 | official_025 | `0.7*018 + 0.3*004` logits ensemble | 0.9439 | **0.9476** | +0.0037 | 2x ConvNeXt | 并列平台第一 |
| 2 | official_035 | `0.7*028 + 0.3*004` logits ensemble | 0.9439 | **0.9476** | +0.0037 | 2x ConvNeXt | 并列平台第一，未超过 025 |
| 3 | official_024 | official_018 top-3 checkpoint averaging | 0.9423 | **0.9464** | +0.0041 | 1x ConvNeXt | 当前最强 1x |
| 4 | official_031 | official_018 original + horizontal flip TTA | 0.9431 | **0.9459** | +0.0028 | 2x TTA | 略高于 013，但成本翻倍 |
| 5 | official_013 | `0.7*009 + 0.3*004` logits ensemble | 0.9524 | **0.9456** | -0.0068 | 2x ConvNeXt | 旧平台第一，仍是强基线 |
| 6 | official_030 | official_013 logits + cloudy `-0.5` class bias | 0.9541 | **0.9456** | -0.0085 | 2x ConvNeXt | 平台等于 013，class bias 无收益 |
| 7 | official_032 | train+val fixed-schedule retrain | — | 0.9455 | — | 1x ConvNeXt | 合并重训未超过 024/025 |
| 8 | official_018 | 224 warm-start from `exp_044`, EMA | 0.9423 | 0.9446 | +0.0023 | 1x ConvNeXt | 024/025 的核心分支 |
| 9 | official_028 | official_018 dense/SWA-style averaging | 0.9423 | 0.9446 | +0.0023 | 1x ConvNeXt | 等于 018，未超过 024 |
| 10 | official_012 | `0.8*009 + 0.2*004` logits ensemble | 0.9499 | 0.9424 | -0.0075 | 2x ConvNeXt | ensemble 有效，但低于 013/025 |
| 11 | official_023 | 288 ConvNeXt-Tiny | 0.9461 | 0.9420 | -0.0041 | 1x ConvNeXt | 分辨率增大未超过 224 averaging |
| 12 | official_010 | official_009 top-3 checkpoint averaging | 0.9412 | 0.9387 | -0.0025 | 1x ConvNeXt | averaging 在 009 上也有平台小收益 |
| 13 | official_009 | 224 warm-start from `exp_054`, seed=42 | 0.9412 | 0.9373 | -0.0039 | 1x ConvNeXt | 旧 warm-start 基线 |
| 14 | official_019 | 256 warm-start from `exp_054` | 0.9540 | 0.9359 | -0.0181 | 1x ConvNeXt | 本地高分不迁移 |
| 15 | official_004 | official_002 top-3 checkpoint averaging | 0.9315 | 0.9315 | 0.0000 | 1x ConvNeXt | 单独不强，但与 009/018 互补 |
| 16 | official_014 | warm-start seed=7 | 0.9385 | 0.9283 | -0.0102 | 1x ConvNeXt | seed=7 不如 018/024 |
| 17 | official_011 | official_007 top-3 checkpoint averaging | 0.9507 | 0.9280 | -0.0227 | 1x ConvNeXt | holdout 明显高估 |
| 18 | official_022 | progressive resize 256->320 | 0.9572 | 0.9253 | -0.0319 | 1x ConvNeXt | 本地最高但平台翻车 |
| 19 | official_007 | 224 ImageNet init, EMA | 0.9456 | 0.9252 | -0.0204 | 1x ConvNeXt | 非 warm-start 平台不稳 |
| 20 | official_008 | 320 ImageNet init, EMA | 0.9399 | 0.9196 | -0.0203 | 1x ConvNeXt | 320 未带来平台收益 |
| 21 | official_005 | 320, wd=5e-4, EMA | 0.9142 | 0.9188 | +0.0046 | 1x ConvNeXt | 分数低，不作主线 |
| 22 | official_003 | 320, standard aug, EMA | 0.9246 | 0.9158 | -0.0088 | 1x ConvNeXt | 标准增强不稳 |
| 23 | official_001 | 320 baseline, EMA | 0.9376 | 0.9147 | -0.0229 | 1x ConvNeXt | 平台弱 |
| 24 | official_002 | 320 baseline, no EMA | 0.9287 | 0.9059 | -0.0228 | 1x ConvNeXt | 不作主线 |
| 25 | official_006 | 320 low-wd, no EMA | 0.9291 | 0.9034 | -0.0257 | 1x ConvNeXt | 不作主线 |

## 平台待测优先级

| 提交优先级 | Experiment | 配置 | Local holdout | 推理成本 | 统一权重文件 | 判断 |
|------------|------------|------|--------------:|----------|--------------|------|
| — | — | — | — | — | — | 暂无 |

## 本地 Holdout 参考榜

本表只用于筛选，不代表平台排序。`official_022/019/011/030` 已证明本地高分可能严重不迁移。

| # | Experiment | 方向 | Local holdout | Official test | 备注 |
|---|------------|------|--------------:|--------------:|------|
| 1 | official_022 | progressive resize 256->320 | **0.9572** | 0.9253 | 本地最高但平台翻车 |
| 2 | official_030 | class-bias calibrated official_013 | **0.9541** | 0.9456 | class bias 平台无收益 |
| 3 | official_019 | 256 warm-start | **0.9540** | 0.9359 | 本地高分不迁移 |
| 4 | official_013 | 009/004 logits ensemble | **0.9524** | 0.9456 | 旧平台第一 |
| 5 | official_011 | 007 top-3 averaging | **0.9507** | 0.9280 | holdout 高估 |
| 6 | official_012 | 009/004 logits ensemble | 0.9499 | 0.9424 | 低于 013/025 |
| 7 | official_020 | ConvNeXt-Small capacity probe | 0.9470 | — | 大模型未作为主线 |
| 8 | official_023 | size 288 probe | 0.9461 | 0.9420 | 不如 024 |
| 9 | official_007 | 224 ImageNet init, EMA | 0.9456 | 0.9252 | 平台不稳 |
| 10 | official_025 | 018/004 logits ensemble | 0.9439 | **0.9476** | 并列平台第一 |
| 11 | official_035 | 028/004 logits ensemble | 0.9439 | **0.9476** | 并列平台第一，未超过 025 |
| 12 | official_031 | official_018 hflip TTA | 0.9431 | 0.9459 | 平台略高于 013 |
| 13 | official_018 | 224 warm-start from `exp_044` | 0.9423 | 0.9446 | 024/025 的核心分支 |
| 14 | official_024 | official_018 top-3 averaging | 0.9423 | **0.9464** | 当前最强 1x |
| 15 | official_026 | 018/004 conservative ensemble | 0.9423 | pending | 被 025 支配 |
| 16 | official_028 | official_018 dense/SWA-style averaging | 0.9423 | 0.9446 | 等于 018，未超过 024 |
| 17 | official_027 | 018/009/004 ensemble | 0.9418 | pending | 3x 成本，不优先 |
| 18 | official_029 | temperature calibrated ensemble | 0.9418 | pending | 未提升 |
| 19 | official_009 | 224 warm-start from `exp_054` | 0.9412 | 0.9373 | 旧 warm-start 基线 |
| 20 | official_010 | official_009 top-3 averaging | 0.9412 | 0.9387 | 平台小幅高于 009 |
| 21 | official_014 | warm-start seed=7 | 0.9385 | 0.9283 | 低于 018/024 |
| 22 | official_017 | warm-start no EMA | 0.9316 | — | no EMA 不优先 |
| 23 | official_036 | 033/004 logits ensemble | 0.9303 | pending | 不建议平台测 |
| 24 | official_015 | warm-start seed=2026 | 0.9248 | — | seed 方差偏低 |
| 25 | official_033 | 3-fold 224 logits ensemble | 0.9224 | pending | 本地显著偏低 |
| 26 | official_016 | warm-start lr=1e-5 | 0.9179 | — | 低 lr 不如 3e-5 |
| 27 | official_017_avg | official_017 top-3 averaging | 0.9144 | — | averaging 翻车 |
| — | official_032 | train+val fixed-schedule retrain | — | 0.9455 | 合并重训未超过 024/025 |

## 子榜 A — 1x 单模型 / Checkpoint Averaging

| # | Experiment | 类型 | Local holdout | Official test | 结论 |
|---|------------|------|--------------:|--------------:|------|
| 1 | official_024 | official_018 top-3 averaging | 0.9423 | **0.9464** | 当前最强 1x，值得作为最终低成本提交 |
| 2 | official_032 | train+val fixed-schedule retrain | — | 0.9455 | 合并重训有收益，但低于 024 |
| 3 | official_018 | 224 warm-start EMA | 0.9423 | 0.9446 | 单模型核心基线 |
| 4 | official_028 | dense/SWA-style averaging | 0.9423 | 0.9446 | 等于 018，未超过 024 |
| 5 | official_023 | 288 ConvNeXt-Tiny | 0.9461 | 0.9420 | 分辨率更大但平台不占优 |
| 6 | official_010 | official_009 top-3 averaging | 0.9412 | 0.9387 | averaging 有小收益，但低于 018/024 |

## 子榜 B — Ensemble / TTA / Calibration

| # | Experiment | 组成 | Local holdout | Official test | 结论 |
|---|------------|------|--------------:|--------------:|------|
| 1 | official_025 | `0.7*018 + 0.3*004` | 0.9439 | **0.9476** | 并列平台第一，004 与 018 互补成立 |
| 2 | official_035 | `0.7*028 + 0.3*004` | 0.9439 | **0.9476** | 并列平台第一，未超过 025 |
| 3 | official_031 | official_018 + hflip TTA | 0.9431 | 0.9459 | 有正收益，但 2x TTA 低于 025/035 |
| 4 | official_013 | `0.7*009 + 0.3*004` | 0.9524 | 0.9456 | 旧平台第一，仍是强基线 |
| 5 | official_030 | official_013 + class bias | 0.9541 | 0.9456 | holdout bias 过拟合，平台无收益 |
| 6 | official_012 | `0.8*009 + 0.2*004` | 0.9499 | 0.9424 | ensemble 有效但权重不如 013/025 |
| 7 | official_033 | 3-fold 224 logits ensemble | 0.9224 | pending | 本地显著异常，不建议扩展到 5-fold |
| 8 | official_036 | `0.7*033 + 0.3*004` | 0.9303 | pending | 004 未能拉回 033 |

## 子榜 C — Warm-start Seed / LR / EMA 对照

| Experiment | 配置 | Val F1 | Holdout F1 | Official test | 判断 |
|------------|------|-------:|-----------:|--------------:|------|
| official_018 | seed=42, EMA, lr=3e-5, `exp_044` warm-start | 0.9467 | 0.9423 | 0.9446 | 当前主线成员 |
| official_009 | seed=42, EMA, lr=3e-5, `exp_054` warm-start | 0.9509 | 0.9412 | 0.9373 | 旧 warm-start 基线 |
| official_014 | seed=7, EMA | 0.9473 | 0.9385 | 0.9283 | seed=7 平台不稳 |
| official_015 | seed=2026, EMA | 0.9431 | 0.9248 | — | 不优先 |
| official_016 | lr=1e-5, EMA | 0.9399 | 0.9179 | — | lr=1e-5 不如 3e-5 |
| official_017 | no EMA | 0.9463 | 0.9316 | — | no EMA 不如 EMA |

结论：继续随机 seed 碰运气价值低；`official_018` 系列和 004 互补比单纯换 seed 更有效。

## 子榜 D — 分辨率 / Backbone 探索

| Experiment | 配置 | Val F1 | Holdout F1 | Official test | 判断 |
|------------|------|-------:|-----------:|--------------:|------|
| official_018 | 224 ConvNeXt-Tiny | 0.9467 | 0.9423 | **0.9446** | 主线基座 |
| official_019 | 256 ConvNeXt-Tiny | 0.9543 | 0.9540 | 0.9359 | 本地不迁移 |
| official_020 | 224 ConvNeXt-Small | 0.9457 | 0.9470 | — | 更大 backbone 未显示必要性 |
| official_022 | 256->320 progressive resize | 0.9551 | 0.9572 | 0.9253 | 平台翻车，停止 resize 主线 |
| official_023 | 288 ConvNeXt-Tiny | 0.9568 | 0.9461 | 0.9420 | 不如 224 averaging |

结论：本阶段不继续扩大分辨率或 backbone；平台反馈更支持 224 warm-start、checkpoint averaging 与轻量 logits ensemble。

## 当前结论

1. **平台正式分数第一是 `official_025/035` 并列**：0.9476，成本都是 2x ConvNeXt。
2. **最强 1x 是 `official_024`**：0.9464，说明 018 top-3 checkpoint averaging 在平台上确实有效。
3. **`official_035` 没有超过 `official_025`**：028 分支替换 018 后平台持平，说明 025 已经吃到主要 ensemble 收益。
4. **`official_032` 合并 train+val 重训没有超过 024/025**：平台 0.9455，低于最强 1x 和最强 2x。
5. **`official_030` 证明 holdout class-bias 过拟合**：本地 0.9541，但平台等于 013，没有真实增益。
6. **`official_031` 有小正收益但性价比一般**：2x TTA 得到 0.9459，低于 1x 的 024 和 2x 的 025/035。
7. **3-fold 路线当前不成立**：`official_033` 本地 0.9224，`official_036` 本地 0.9303，不建议扩展到 5-fold。
8. **本阶段不建议再增加新实验**：最终提交默认选 025/035 二选一；如果推理成本受限，选 024。

## Artifact 说明

- 统一平台提交权重命名：`outputs/official_xxx/official_xxx_best_model.pth`。
- `.pth` 大权重被 `.gitignore` 忽略，不进入 Git；Git 中保留 `official_xxx_bundle.json`、`results.json`、`predictions.csv`、`error_samples.csv` 和混淆矩阵。
- 012/013/025/026/027/029/030/031/033/035/036 是 submission bundle 或推理时后处理，不是新训练出的单一 ConvNeXt `state_dict`。
