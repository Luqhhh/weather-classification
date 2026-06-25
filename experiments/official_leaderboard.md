# Official Leaderboard — Weather Image Classification

> 数据集：官方 70/15/15 split | 更新日期：2026-06-25 | 排序：holdout macro F1 ↓

## 总榜

| # | Experiment | Sz | Model | Macro F1 (holdout) | Macro F1 (val) | Gap | rainy | cloudy | snowy | sunny | Notes |
|---|------------|----|-------|-------------------:|---------------:|-----|------:|-------:|------:|------:|-------|
| 1 | official_013 | 224+320 | ConvNeXt-Tiny logits ensemble | **0.9524** | 0.9551 | -0.0027 | 0.9624 | 0.9482 | 0.9492 | 0.9497 | 0.7*009 + 0.3*004 |
| 2 | official_011 | 224 | ConvNeXt-Tiny (avg) | **0.9507** | 0.9347 | +0.0160 | 0.9692 | 0.9506 | 0.9298 | 0.9530 | top-3 averaging from 007 |
| 3 | official_012 | 224+320 | ConvNeXt-Tiny logits ensemble | **0.9499** | 0.9535 | -0.0036 | 0.9624 | 0.9433 | 0.9492 | 0.9446 | 0.8*009 + 0.2*004 |
| 4 | official_007 | 224 | ConvNeXt-Tiny | **0.9456** | 0.9380 | +0.0076 | 0.9474 | 0.9502 | 0.9298 | 0.9549 | EMA tight crop seed=42 |
| 5 | official_009 | 224 | ConvNeXt-Tiny | **0.9412** | 0.9509 | -0.0097 | 0.9474 | 0.9349 | 0.9412 | 0.9414 | warm-start from exp_054 |
| 6 | official_010 | 224 | ConvNeXt-Tiny (avg) | **0.9412** | 0.9484 | -0.0072 | 0.9474 | 0.9349 | 0.9412 | 0.9414 | top-3 averaging from 009 |
| 7 | official_008 | 320 | ConvNeXt-Tiny | **0.9399** | 0.9375 | +0.0024 | 0.9466 | 0.9423 | 0.9231 | 0.9477 | EMA tight crop seed=42 |
| 8 | official_001 | 320 | ConvNeXt-Tiny | **0.9376** | 0.9366 | +0.0010 | 0.9333 | 0.9467 | 0.9244 | 0.9459 | EMA d=0.3 wd=0.05 |
| 9 | official_004 | 320 | ConvNeXt-Tiny (avg) | **0.9315** | — | — | 0.9254 | 0.9392 | 0.9180 | 0.9432 | top-3 checkpoint averaging from 002 |
| 10 | official_006 | 320 | ConvNeXt-Tiny | **0.9291** | 0.9301 | -0.0010 | 0.9185 | 0.9342 | 0.9256 | 0.9381 | no EMA wd=5e-4 |
| 11 | official_002 | 320 | ConvNeXt-Tiny | **0.9287** | 0.9371 | -0.0084 | 0.9118 | 0.9362 | 0.9256 | 0.9412 | no EMA wd=0.05 |
| 12 | official_003 | 320 | ConvNeXt-Tiny | **0.9246** | 0.9450 | -0.0204 | 0.9051 | 0.9408 | 0.9076 | 0.9450 | EMA - severe overfit |
| 13 | official_005 | 320 | ConvNeXt-Tiny | **0.9142** | 0.9400 | -0.0258 | 0.8872 | 0.9318 | 0.8983 | 0.9396 | EMA wd=5e-4 - severe overfit |

## Val vs Holdout Gap 分析

| Experiment | Val F1 | Holdout F1 | Gap | 判断 |
|------------|-------:|-----------:|----:|------|
| official_011 | 0.9347 | 0.9507 | +0.0160 | 本地 holdout 最高单权重，val 明显低估 |
| official_007 | 0.9380 | 0.9456 | +0.0076 | 稳定泛化，仍是 011 的来源 |
| official_008 | 0.9375 | 0.9399 | +0.0024 | 稳定泛化 |
| official_001 | 0.9366 | 0.9376 | +0.0010 | 稳定 |
| official_006 | 0.9301 | 0.9291 | -0.0010 | 稳定 |
| official_013 | 0.9551 | 0.9524 | -0.0027 | ensemble 稳定，但 2x 推理成本 |
| official_012 | 0.9535 | 0.9499 | -0.0036 | ensemble 稳定，低于 013 |
| official_010 | 0.9484 | 0.9412 | -0.0072 | 未改善 009 holdout |
| official_002 | 0.9371 | 0.9287 | -0.0084 | 轻微过拟合 |
| official_009 | 0.9509 | 0.9412 | -0.0097 | 过拟合，但平台 test 当前最高 |
| official_003 | 0.9450 | 0.9246 | -0.0204 | 严重过拟合 |
| official_005 | 0.9400 | 0.9142 | -0.0258 | 严重过拟合 |

## 关键发现

1. **official_011 是当前本地最高单权重**：007 top-3 averaging 后 holdout 从 0.9456 提升到 0.9507，rainy F1 达 0.9692。
2. **official_013 是当前本地最高整体模型**：0.7*009 + 0.3*004 的 logits ensemble holdout 0.9524，高于 012 的 0.9499。
3. **official_010 没有改善 warm-start holdout**：009 top-3 averaging 后 holdout 仍为 0.9412，且 val 从 0.9509 降到 0.9484。
4. **本地和平台仍不完全一致**：009 本地 holdout 低于 007/011/013，但平台 test 当前最高，因此 011/013 必须平台复核。
5. **ensemble 有收益但有成本**：012/013 都需要 224+320 两个 ConvNeXt 推理，只有平台 test 提升足够时才值得作为最终提交。

## 提交建议

下一批平台 test 优先级：

1. **official_011**：最高本地单权重，推理成本仍是 1x ConvNeXt。
2. **official_013**：最高本地整体模型，但需要 2x 推理成本和 ensemble submission 支持。
3. **official_012**：作为 013 的低 004 权重对照；若平台预算紧张可跳过。
4. **official_010**：本地未超过 009，暂不优先提交。

---

## 子榜 A — 单模型 Top 5

| # | Experiment | Sz | Macro F1 | rainy | cloudy | snowy | sunny | 备注 |
|---|------------|----|---------:|------:|-------:|------:|------:|------|
| 1 | official_007 | 224 | 0.9456 | 0.9474 | 0.9502 | 0.9298 | 0.9549 | EMA tight crop |
| 2 | official_009 | 224 | 0.9412 | 0.9474 | 0.9349 | 0.9412 | 0.9414 | warm-start exp_054 |
| 3 | official_008 | 320 | 0.9399 | 0.9466 | 0.9423 | 0.9231 | 0.9477 | EMA tight crop |
| 4 | official_001 | 320 | 0.9376 | 0.9333 | 0.9467 | 0.9244 | 0.9459 | EMA baseline |
| 5 | official_006 | 320 | 0.9291 | 0.9185 | 0.9342 | 0.9256 | 0.9381 | no EMA wd=5e-4 |

## 子榜 B — Checkpoint Averaging

| # | Experiment | 源实验 | Top-K | Macro F1 | vs 源模型 | 备注 |
|---|------------|--------|-------|---------:|----------:|------|
| 1 | official_011 | 007 | 3 | 0.9507 | **+0.0051** | 最大提升，rainy 0.9692 |
| 2 | official_004 | 002 | 3 | 0.9315 | +0.0028 | 小幅提升 |
| 3 | official_010 | 009 | 3 | 0.9412 | 0.0000 | 未改善，源模型已足够稳定 |

结论：averaging 对 007 有效（+0.5），对 002 微有效（+0.3），对 009 无效（warm-start 已收敛）。

## 子榜 C — Ensemble

| # | Experiment | 组成 | Macro F1 | rainy | 推理成本 | 备注 |
|---|------------|------|---------:|------:|---------:|------|
| 1 | official_013 | 0.7×009 + 0.3×004 | 0.9524 | 0.9624 | 2× ConvNeXt | 最高本地，009+004 互补 |
| 2 | official_012 | 0.8×009 + 0.2×004 | 0.9499 | 0.9624 | 2× ConvNeXt | 013 低 004 权重的对照 |

> 012/013 无 .pth 文件：logits ensemble 不训练新模型，仅在推理时加权平均已有模型的输出。

## 子榜 D — EMA 消融

| # | Experiment | EMA | WD | Macro F1 | Gap | 判断 |
|---|------------|-----|-----|---------:|----:|------|
| 1 | official_001 | ✓ | 0.05 | 0.9376 | +0.0010 | 稳定 |
| 2 | official_002 | ✗ | 0.05 | 0.9287 | -0.0084 | 无 EMA，val→holdout 下降明显 |
| 3 | official_003 | ✓ | 0.05 | 0.9246 | -0.0204 | EMA 但严重过拟合 |
| 4 | official_006 | ✗ | 5e-4 | 0.9291 | -0.0010 | 无 EMA 但 Gap 小 |
| 5 | official_005 | ✓ | 5e-4 | 0.9142 | -0.0258 | EMA + 低 wd，严重过拟合 |

## 子榜 E — 输入尺寸 224 vs 320

| # | Experiment | Sz | Macro F1 | 推理时间 | 备注 |
|---|------------|----|---------:|---------:|------|
| 1 | official_007 | 224 | 0.9456 | ~2.5s | 单权重最高 |
| 2 | official_008 | 320 | 0.9399 | ~4.0s | 320 最高 |
| 3 | official_001 | 320 | 0.9376 | ~4.8s | 320 baseline |

224 全面优于 320：holdout 更高 + 推理快 ~40%。

## 子榜 F — 训练策略对比

| 策略 | Experiment | Macro F1 | 说明 |
|------|------------|---------:|------|
| 从头训练 (EMA) | official_007 | 0.9456 | EMA tight crop seed=42 |
| Warm-start | official_009 | 0.9412 | 从 exp_054 初始化 |
| Averaging | official_011 | 0.9507 | 007 top-3 平均 |
| Ensemble | official_013 | 0.9524 | 009 + 004 加权 |

Warm-start 对平台 test 最有效（009 = 0.9373），但本地 holdout 上 Averaging 和 Ensemble 更高。需平台复核。
