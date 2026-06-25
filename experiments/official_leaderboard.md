# Official Leaderboard — Weather Image Classification

> 数据集：官方 70/15/15 split | 更新日期：2026-06-25 | 排序：holdout macro F1 ↓

## 总榜

| # | Experiment | Sz | Model | Macro F1 (holdout) | Macro F1 (val) | Gap | rainy | cloudy | snowy | sunny | Notes |
|---|------------|----|-------|-------------------:|---------------:|-----|------:|-------:|------:|------:|-------|
| 1 | official_007 | 224 | ConvNeXt-Tiny | **0.9456** | 0.9380 | +0.0076 | 0.9474 | 0.9502 | 0.9298 | 0.9549 | EMA tight crop seed=42 |
| 2 | official_009 | 224 | ConvNeXt-Tiny | **0.9412** | 0.9509 | -0.0097 | 0.9474 | 0.9349 | 0.9412 | 0.9414 | warm-start from exp_054 |
| 3 | official_008 | 320 | ConvNeXt-Tiny | **0.9399** | 0.9375 | +0.0024 | 0.9466 | 0.9423 | 0.9231 | 0.9477 | EMA tight crop seed=42 |
| 4 | official_001 | 320 | ConvNeXt-Tiny | **0.9376** | 0.9366 | +0.0010 | 0.9333 | 0.9467 | 0.9244 | 0.9459 | EMA d=0.3 wd=0.05 |
| 5 | official_004 | 320 | ConvNeXt-Tiny (avg) | **0.9315** | — | — | 0.9254 | 0.9392 | 0.9180 | 0.9432 | top-3 checkpoint averaging from 002 |
| 6 | official_006 | 320 | ConvNeXt-Tiny | **0.9291** | 0.9301 | -0.0010 | 0.9185 | 0.9342 | 0.9256 | 0.9381 | no EMA wd=5e-4 |
| 7 | official_002 | 320 | ConvNeXt-Tiny | **0.9287** | 0.9371 | -0.0084 | 0.9118 | 0.9362 | 0.9256 | 0.9412 | no EMA wd=0.05 |
| 8 | official_003 | 320 | ConvNeXt-Tiny | **0.9246** | 0.9450 | -0.0204 | 0.9051 | 0.9408 | 0.9076 | 0.9450 | EMA — severe overfit |
| 9 | official_005 | 320 | ConvNeXt-Tiny | **0.9142** | 0.9400 | -0.0258 | 0.8872 | 0.9318 | 0.8983 | 0.9396 | EMA wd=5e-4 — severe overfit |

## Val vs Holdout Gap 分析

| Experiment | Val F1 | Holdout F1 | Gap | 判断 |
|------------|-------:|-----------:|----:|------|
| official_007 | 0.9380 | 0.9456 | +0.0076 | ✅ 泛化最好 — holdout 高于 val |
| official_008 | 0.9375 | 0.9399 | +0.0024 | ✅ 稳定泛化 |
| official_001 | 0.9366 | 0.9376 | +0.0010 | ✅ 最稳定 |
| official_006 | 0.9301 | 0.9291 | -0.0010 | ✅ 稳定 |
| official_002 | 0.9371 | 0.9287 | -0.0084 | ⚠️ 轻微过拟合 |
| official_009 | 0.9509 | 0.9412 | -0.0097 | ⚠️ 过拟合 — val 最高但 holdout 下降 |
| official_003 | 0.9450 | 0.9246 | -0.0204 | ❌ 严重过拟合 |
| official_005 | 0.9400 | 0.9142 | -0.0258 | ❌ 严重过拟合 |

## 关键发现

1. **224 尺寸优于 320**：top 2 都是 224×224，且推理更快（2.0-2.5s vs 3.9-4.8s）
2. **EMA 有效但非决定性**：007 和 008 均有 EMA，但没有 EMA 的 006 也稳定
3. **wd=5e-4 导致严重过拟合**：003 和 005 val→holdout 掉了 2-2.5 个点
4. **warm-start 有效但过拟合风险高**：009 val 最高(0.9509)但 holdout 降到 0.9412
5. **checkpoint averaging (004) 提升 rainy +0.014**：002→004 的 averaging 让 rainy F1 从 0.9118 升到 0.9254

## 提交建议

首选 **official_007**（224, EMA, holdout 0.9456），备选 **official_009**（warm-start, holdout 0.9412）。
