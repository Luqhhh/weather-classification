# 比赛数据集发布后优先实验 TODO

> 目标：先验证当前最强结论能否迁移到正式比赛数据集，再决定是否扩大搜索。不要一上来重扫 dropout、backbone 或强增强。

## P0 - 数据落地与验证集切分

> **本地已完成**: 旧数据已重切为 70/15/15 三层结构（`data/train` / `data/val` / `data/holdout`），
> SHA-256 全局去重，seed=42 分层抽样。Holdout 上 exp_054 vs exp_044 已验证排序和 val 一致（差 < 0.001）。
> 工具脚本：`scripts/split_data.py`（从任意目录重建切分）、`scripts/organize_datasets.py`（从 Kaggle 原始数据走全流程）。

- [ ] 导入官方训练/测试数据，确认目录结构、类别名、样本数和图像可读性。
- [ ] **用 `scripts/split_data.py` 重建三层切分**（`--source <官方数据目录> --output_dir data --seed 42`），自动完成 hash 去重、leakage 检查和 stratified split。
- [ ] 确认 `data/holdout/` 已生成且各类样本数合理（rainy/snowy 至少 > 200）。
- [ ] 记录每类样本数和 splits 各类 support，重点观察 `rainy` / `snowy` 是否仍是弱类。

完成标准：

- `data/train`、`data/val`、`data/holdout`、`data/test` 可被现有 dataloader 直接读取。
- 有明确的 class distribution、重复样本检查结论和 split seed。
- val 和 holdout 上模型排名一致（差 < 0.002）才进入后续实验。
- 不用当前旧 validation set 的排序直接决定最终提交。

## P1 - 第一批必须跑的主线实验

优先只跑 ConvNeXt-Tiny 320，因为当前本地实验显示普通单点搜索已经收敛，提升主要来自权重平滑。

| 优先级 | 临时 ID | 实验 | 配置 | 目的 |
|--------|---------|------|------|------|
| 1 | official_001 | 主候选复现 | ConvNeXt-Tiny 320, CE, dropout=0.3, wd=0.05, EMA=0.999, no rotation, light CJ, seed=42 | 验证 `exp_054` 结论是否迁移 |
| 2 | official_002 | 同配置 plain checkpoint | ConvNeXt-Tiny 320, CE, dropout=0.3, wd=0.05, no EMA, no rotation, light CJ, seed=42 | 量化 EMA 相对普通 checkpoint 的收益 |
| 3 | official_003 | EMA + 标准增强（exp_044 复现） | ConvNeXt-Tiny 320, CE, dropout=0.3, wd=0.05, EMA=0.999, **默认 rotation + 默认 CJ**, seed=42 | 量化保守增强 vs 标准增强的收益（对比 official_001） |
| 4 | official_004 | 同配置 top-k averaging | 基于 official_001/002 的 top-3 checkpoints | 检查 checkpoint averaging 是否仍有效 |
| 5 | official_005 | 低 wd + EMA | ConvNeXt-Tiny 320, CE, dropout=0.3, wd=5e-4, EMA=0.999, seed=42 | 验证 `exp_050/051` 低 wd 路线是否迁移 |
| 6 | official_006 | 低 wd plain 对照 | ConvNeXt-Tiny 320, CE, dropout=0.3, wd=5e-4, no EMA, seed=42 | 分清低 wd 本身收益和权重平滑收益 |

> **为什么加 official_003**：当前数据上 exp_044（EMA + 标准增强）= 0.9182 vs exp_054（EMA + 保守增强）= 0.9204，差了 0.002。需要在官方数据上确认保守增强的收益是否可复现，还是 EMA 本身贡献了大部分。

建议命令模板：

```bash
python3 scripts/train.py \
  --config configs/models/convnext_tiny.yaml \
  --output_dir outputs \
  --experiment_id official_001 \
  --notes "official data: ConvNeXt 320 CE d=0.3 wd=0.05 EMA=0.999 seed=42" \
  -- \
  --logging.experiment_name official_001 \
  --data.image_size 320 \
  --training.batch_size 32 \
  --seed 42 \
  --model.dropout 0.3 \
  --training.loss.name cross_entropy \
  --training.optimizer.weight_decay 0.05 \
  --training.ema.enabled true \
  --training.ema.decay 0.999 \
  --data.augmentation.random_rotation.degrees 0 \
  --data.augmentation.color_jitter '{"brightness":0.08,"contrast":0.08,"saturation":0.08,"hue":0.03}'
```

判断标准：

- 若 `official_001` 比 plain checkpoint 高 `>= 0.002` macro F1，优先固定 EMA 作为最终训练流程。
- 若 `wd=0.05 + EMA` 仍高于 `wd=5e-4 + EMA`，不要把低 weight decay 作为默认配置。
- 若 top-k averaging 不超过 EMA，不把 averaging 作为默认提交，只保留为备选。

## P2 - 多 seed 稳定性确认

第一批结果出来后，只对有希望进最终候选的路线做 3 seed。

| 路线 | Seeds | 触发条件 |
|------|-------|----------|
| `wd=0.05 + EMA` | 42 / 7 / 2026 | 默认必做，除非 official_001 明显失败 |
| `wd=5e-4 + EMA` | 42 / 7 / 2026 | 只有当 official_004 距离 official_001 在 `0.002` 以内时做 |
| `top-k checkpoint averaging` | 42 / 7 / 2026 | 只有当 seed42 上超过 plain 或接近 EMA 时做 |

统计方式：

- 汇总 mean / std macro F1。
- 同时看 rainy F1、snowy F1、val loss 和 early stop epoch。
- 若 std `>= 0.003`，不要按单次最高分做最终决策。

## P3 - Ensemble 和多模型集成

> 在 P2 多 seed 稳定性确认后，若多 seed 模型间方差 < 0.003，优先尝试以下集成方式。
> 本地实验 exp_052（双模型 ensemble）= 0.9159 macro F1 / 0.9056 rainy F1，rainy 互补性强但 macro F1 低于单 EMA。

| 优先级 | 临时 ID | 实验 | 配置 | 目的 |
|--------|---------|------|------|------|
| 1 | official_010 | 多 seed EMA 平均 ensemble | official_001 的 3 seed EMA 模型 logits 平均 | 最低成本的集成，不增加推理时间（权重可平均） |
| 2 | official_011 | 多 seed top-k averaging | 3 seed × top-3 checkpoints 权重平均 | 和 official_004 对照，看 seed 多样性 vs 单 seed top-k |
| 3 | official_012 | 双模型 logits ensemble | EMA 主模型 + 互补分支，权重搜索 0.5~0.9 | 仅当 rainy 需要额外提升时做，参考 exp_052 |

判断标准：
- 多 seed EMA 平均的 macro F1 超过单 seed 最高分 ≥ 0.002 才作为主候选。
- 权重平均不增加推理成本，优先于 logits ensemble。
- 双模型 ensemble 增加 ~2x 推理成本，只在 rainy F1 增益 ≥ 0.005 时考虑。

## P4 - rainy 互补路线

只有当官方验证集中 `rainy` 仍明显弱于其他类，才优先做下面实验。

| 优先级 | 临时 ID | 实验 | 配置 | 目的 |
|--------|---------|------|------|------|
| 1 | official_013 | LabelSmoothing rainy 分支 | ConvNeXt-Tiny 320, LS=0.05, d=0.3, wd=0.05, seed=42 | 复验 `exp_030` 的 rainy 互补性 |
| 2 | official_014 | EMA 主模型 + LS logits ensemble | `0.8 * EMA + 0.2 * LS` | 提升 rainy，同时控制 macro F1 损失 |
| 3 | official_015 | EMA 主模型 + LS logits ensemble | `0.7 * EMA + 0.3 * LS` | 检查 rainy 增益上限 |
| 4 | official_016 | class-balanced sampler | CE, d=0.3, sampler=class_balanced | 仅当 rainy recall 明显偏低时作为后备（⚠️ exp_042: 0.9059 vs baseline 0.9106，大概率无效） |

判断标准：

- ensemble 只有在 macro F1 不下降超过 `0.001`，且 rainy F1 明显提升时才进入最终候选。
- 如果 CPU 推理预算紧张，双模型 ensemble 只能作为分析参考。
- 不建议把 LabelSmoothing 直接作为唯一主模型；它更适合作为 rainy 补充分支。

## P5 - 最终候选必须补齐的检查

对进入最终候选的模型补齐下面项目：

- [ ] **Holdout 评估**：在 `data/holdout` 上跑 evaluate，确认 val→holdout 排名一致（差 < 0.002），不一致则重新审视模型选择。
- [ ] CPU benchmark：至少覆盖 `official_001`、最佳低 wd 备选、最佳 ensemble。
- [ ] submission check：确认模型加载、预测文件格式、类别映射和 CPU-only 环境都能跑。
- [ ] error samples：导出验证集错分样本，重点看 cloudy/rainy、sunny/cloudy 混淆。
- [ ] confusion matrix：保存每个最终候选的混淆矩阵。
- [ ] model size：记录单模型和 ensemble 的权重大小。
- [ ] 更新 `experiments/leaderboard.md`、`experiments/experiment_queue.md`、`experiments/finding.md`。

最终推荐优先级：

1. 最高优先：`ConvNeXt-Tiny 320 + CE + dropout=0.3 + wd=0.05 + EMA=0.999 + no rotation + light CJ`
2. 备选：`ConvNeXt-Tiny 320 + CE + dropout=0.3 + wd=5e-4 + EMA/SWA`
3. rainy 优先备选：`EMA 主模型 + LabelSmoothing 分支 logits ensemble`

## 384 分辨率对照

> 本地三轮实验一致显示 384 负于 320：
> - exp_056 FixRes 320→384: 0.9170 vs exp_054 320: 0.9204
> - exp_060 全训练 384: 0.9164 vs exp_054 320: 0.9204
> - 四个类别全线下滑，非波动。
>
> 新数据集上不再对 384 投入训练资源，只做一次 inference-only 对照。

- [ ] 新数据集发布后，用 `exp_054` (320) 和 `exp_060` (384) 的 checkpoint 各跑一次 evaluate，确认分辨率排序是否迁移。
- [ ] 若 384 在新测试集上翻转（概率低），再重新评估是否做 384 全训练；否则 384 方向画句号。

## 暂时不要做

- [ ] 不先做大范围 dropout sweep。
- [ ] 不先做更多 backbone。
- [ ] 不先上强 RandAugment、MixUp、CutMix 或强 rotation。
- [ ] 不盲目提高输入尺寸到 384。
- [ ] 不继续当前 center-crop TTA 路线。
- [ ] 不在没有最终 holdout 的情况下反复调 ensemble 权重。
- [ ] 不使用 `weights/convnext_tiny_best.pth` 作为稳定实验产物；比较时使用 `outputs/<experiment_id>/best_model.pth` 或对应 averaged/EMA artifact。
- [ ] 不把 loss 层面的不平衡策略（class_weights, FocalLoss, label_smoothing）作为第一批实验。当前数据集上它们在 convnext_tiny 上均不优于纯 CE（exp_011/030/042），但新数据集可能不同。若新数据 rainy/snowy 显著弱于当前数据，再作为第二批补做。
