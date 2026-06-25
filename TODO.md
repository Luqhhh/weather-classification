# 官方数据阶段 TODO

> 当前目标：围绕平台 test 已验证的正信号继续小矩阵优化。不要再按本地 val/holdout 单独排序扩大搜索；本地指标只作为 guardrail，平台 test 反馈是当前最接近目标域的信号。

## 当前结论

### 已完成实验 001-013

| Experiment | 关键配置 | Local val | Local holdout | Official test | 判断 |
|------------|----------|----------:|--------------:|--------------:|------|
| official_009 | 224, tight crop, old `exp_054` warm-start, EMA, lr=3e-5 | 0.9509 | 0.9412 | **0.9373** | 当前主候选 |
| official_013 | `0.7 * official_009 + 0.3 * official_004` logits ensemble | 0.9551 | **0.9524** | 待提交 | 本地最高，2x 推理成本 |
| official_011 | official_007 top-3 checkpoint averaging | 0.9347 | **0.9507** | 待提交 | 本地最高单权重 |
| official_012 | `0.8 * official_009 + 0.2 * official_004` logits ensemble | 0.9535 | **0.9499** | 待提交 | 稳定但低于 013 |
| official_004 | official_002 top-3 checkpoint averaging | - | 0.9315 | **0.9315** | averaging 被本地 holdout 低估，值得继续 |
| official_007 | 224, tight crop, ImageNet init, EMA | 0.9380 | **0.9456** | 0.9252 | 本地最强但平台掉点 |
| official_010 | official_009 top-3 checkpoint averaging | 0.9484 | 0.9412 | 待提交 | 未提升 009 holdout |
| official_008 | 320, tight crop, ImageNet init, EMA | 0.9375 | 0.9399 | 0.9196 | 320 未超过 224 |
| official_005 | 320, wd=5e-4, EMA | 0.9400 | 0.9142 | 0.9188 | 低 wd + EMA 不稳 |
| official_003 | 320, 标准 rotation/CJ, EMA | 0.9450 | 0.9246 | 0.9158 | 标准增强不稳 |
| official_001 | 320, old exp054-style, EMA | 0.9366 | 0.9376 | 0.9147 | 平台 test 弱 |
| official_002 | 320, no EMA, wd=0.05 | 0.9371 | 0.9287 | 0.9059 | 不作主线 |
| official_006 | 320, no EMA, wd=5e-4 | 0.9301 | 0.9291 | 0.9034 | 不作主线 |

### 当前判断

- `official_009` 的旧数据 warm-start 是最大正信号，后续优先围绕它做 averaging、seed 和 lr 小矩阵。
- `official_004` 说明 checkpoint averaging 对平台 test 可能有帮助，不能只按本地 holdout 淘汰。
- `official_011` 是当前本地 holdout 最高的单权重模型；`official_013` 是当前本地 holdout 最高整体模型，但需要 2x 推理和平台 test 确认。
- `official_010` 没有改善 `official_009` 的 holdout，warm-start top-3 averaging 暂不作为主线。
- 224 + tight crop 已经足够；320 没有显示稳定收益，推理成本还更高。
- 标准 rotation / stronger color jitter、低 weight decay、384/512、TTA、cutout 暂不继续。
- rainy/snowy 当前不是明显弱类，暂不做 weighted CE / focal / sampler。

## P0 - 低成本后处理优先

这些实验优先级最高，因为不需要重训或训练成本低，且直接验证平台 test 上的两个正信号：warm-start 和 checkpoint averaging。

| 优先级 | 临时 ID | 实验 | 配置 | 目的 | 状态 |
|--------|---------|------|------|------|------|
| 1 | official_010 | warm-start top-3 averaging | 对 `official_009` 的 top-3 checkpoints 做 weight averaging | 验证 averaging 能否叠加到当前平台第一的 warm-start 路线 | done: val 0.9484 / holdout 0.9412 |
| 2 | official_011 | 224 tight crop top-3 averaging | 对 `official_007` 的 top-3 checkpoints 做 weight averaging | 得到一个非 warm-start 的 224 稳健备选 | done: val 0.9347 / holdout 0.9507 |
| 3 | official_012 | logits ensemble 009/004 A | `0.8 * official_009 + 0.2 * official_004` | 轻量融合 warm-start 与 averaging 分支，控制 004 权重 | done: val 0.9535 / holdout 0.9499 |
| 4 | official_013 | logits ensemble 009/004 B | `0.7 * official_009 + 0.3 * official_004` | 提高 004 互补分支占比，检查平台 test 上限 | done: val 0.9551 / holdout 0.9524 |

判断标准：

- `official_010` 若平台 test 超过 `official_009` 或接近且 holdout 更稳，作为下一版主候选。
- `official_012/013` 若平台 test 提升小于 0.002，默认不承担 2x 推理成本，只作为分析参考。
- logits ensemble 不要继续做大范围权重搜索；最多在 `0.8/0.2` 与 `0.7/0.3` 中选一个。

## P1 - warm-start / resize / 大模型小训练矩阵

这些实验围绕当前主候选 `official_009` 做受控变化，同时加入一个 256 sanity check 和两个更高容量 backbone 探索。大模型实验不展开矩阵，先各跑一条判断是否值得继续。

| 优先级 | 临时 ID | 实验 | 配置 | 目的 | 状态 |
|--------|---------|------|------|------|------|
| 1 | official_014 | warm-start seed 7 | `official_009` 配置，seed=7 | 判断 warm-start 是否稳定，不被 seed42 偶然性支配 | todo |
| 2 | official_015 | warm-start seed 2026 | `official_009` 配置，seed=2026 | 与 014 一起估计 seed 方差 | todo |
| 3 | official_016 | lower-lr warm-start | `official_009` 配置，lr=1e-5 | 检查 3e-5 是否过拟合，是否需要更保守微调 | todo |
| 4 | official_017 | warm-start no EMA + averaging | `official_009` 配置但 EMA off，然后 top-3 averaging | 对照 004 的平台收益是否来自 no-EMA checkpoints + averaging | todo |
| 5 | official_018 | exp044 warm-start | 从旧 `exp_044` 初始化，224 tight crop，lr=3e-5，EMA | 检查旧权重源是否只有 exp054 有效 | todo |
| 6 | official_019 | resize256 warm-start sanity check | `official_009` 配置不变，只改 `image_size=256` | 验证 256 是否在平台域有额外收益；若本地 holdout 不能接近 009 到 `<=0.003`，不提交平台 | todo |
| 7 | official_020 | ConvNeXt-Small capacity probe | ConvNeXt-Small, 224 tight crop, CE, wd=0.05, EMA, seed=42；需先注册 `convnext_small` | 评估更高容量同系列模型是否超过 ConvNeXt-Tiny | todo |
| 8 | official_021 | EfficientNetV2-S capacity probe | EfficientNetV2-S, 224 tight crop, CE, EMA, seed=42；需先注册 `efficientnet_v2_s` | 评估不同架构的更强/互补 backbone 是否值得继续 | todo |

判断标准：

- 多 seed 平均如果明显低于 `official_009`，只保留 009 单 seed，不做 seed ensemble。
- 若 `official_016` 平台 test 高于 009，后续 warm-start 默认 lr 降到 1e-5。
- 若 `official_017` 接近或超过 009，优先做 no-EMA checkpoint averaging，而不是继续强化 EMA。
- `official_020/021` 若本地 holdout 明显低于 009，不提交平台；若接近，再提交平台 test 判断是否进入主线或 ensemble 备选。

## P2 - 只在 P0/P1 受限时补充

| 临时 ID | 实验 | 触发条件 |
|---------|------|----------|
| official_022 | 009/014/015 seed logits ensemble | 只有当 014/015 单模型平台 test 接近 009 时做 |
| official_023 | warm-start LS=0.05 分支 | 只有当平台错误分析显示 rainy 明显偏弱时做 |
| official_024 | 009 + LS 分支 logits ensemble | 只有 official_023 的 rainy 互补性明确时做 |

## 暂时不要做

- 不继续 320 大矩阵。
- 不做 384/512。
- 不回到标准 rotation + stronger color jitter。
- 不继续低 weight decay 主线。
- 不优先 weighted CE / focal / sampler。
- 不做 cutout / 当前 center-crop TTA。
- 不在 `0.8/0.2`、`0.7/0.3` 之外继续大范围调 logits ensemble 权重。
- 不使用 `weights/convnext_tiny_best.pth` 做稳定比较；比较时使用 `outputs/<experiment_id>/best_model.pth` 或明确的 averaged artifact。

## 下一步执行顺序

1. 优先把 `official_011` 提交平台 test，验证本地最高单权重是否迁移。
2. 若平台推理时间允许，再提交 `official_013`；`official_012` 作为 013 的低 004 权重对照。
3. `official_010` 暂不优先提交，除非需要验证 warm-start averaging 的平台一致性。
4. 若 `official_011/013` 平台 test 没超过 `official_009`，再启动 `official_014/015/016`，并补 `official_019/020/021`。
5. 每批结束后同步 `experiments/official_leaderboard.md` 和 `experiments/officialTestScore.md`，不要只更新本文件。
