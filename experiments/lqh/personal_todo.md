# Personal TODO - B: Loss & Augmentation

我的分工是 B，方向是 Loss & Augmentation。核心目标不是盲目堆实验，而是在公共 baseline 上提升泛化能力，重点提升 rainy / snowy 少数类 F1，同时不能明显牺牲 cloudy / sunny。

## 当前基准

- 数据不平衡明显：cloudy 6640，rainy 1828，snowy 1562，sunny 6888，最大/最小约 4.41:1。
- 现有 baseline：ResNet-18 + CE + 224 + 默认增强。
- 当前 baseline 指标：val macro F1 约 0.8708，rainy F1 约 0.8240，snowy F1 约 0.8927。
- 当前瓶颈：rainy 召回偏低，主要需要通过 loss、类别权重、采样或更有针对性的增强来修正。
- CPU 推理不是当前主要风险，ResNet-18 已经很快；B 的优先级应放在 macro F1 和少数类 F1。

## 工作原则

- 每次实验只改变一个主要变量，保证能归因。
- 不用 `B` 这种长期个人分支，按实验建分支，例如 `exp/loss-focal`、`exp/loss-weighted-ce`、`exp/aug-randaugment`。
- 每轮实验都记录：实验名、commit hash、配置、训练命令、best epoch、val macro F1、per-class F1、混淆矩阵观察、是否推荐进入下一阶段。
- 先比较 loss，再固定最优 loss 比较 augmentation，最后只组合 Top 2 策略。
- 不要一开始组合 `FocalLoss + LabelSmoothing + Weighted CE + RandAugment + MixUp + CutMix`，这种实验不可归因。


## Phase 1 - Loss 对比

目标：找出能提升 macro F1，尤其是 rainy / snowy F1 的 loss。

- [ ] CE baseline 复核：使用现有 `exp_001` 作为对照，不重复无意义训练，除非环境或代码变了。
- [ ] FocalLoss：重点观察 rainy recall 是否提升。
- [ ] LabelSmoothing：重点观察 cloudy/sunny 混淆是否下降，以及是否牺牲少数类。
- [ ] Weighted CE：使用类别权重补偿 rainy/snowy。
- [ ] Weighted FocalLoss：只在 Focal 或 Weighted CE 有明显收益后再试。

建议权重起点：

```text
class_weights = [0.64, 2.31, 2.71, 0.61]
order = [cloudy, rainy, snowy, sunny]

```

Phase 1 结束判断：

- 最优 loss 的 macro F1 是否高于 0.8708。
- rainy F1 是否高于 0.8240。
- snowy F1 是否不下降或有提升。
- cloudy / sunny 是否没有明显下降。
- 训练曲线是否稳定，没有明显过拟合或震荡。

## Phase 2 - Augmentation 对比

目标：固定 Phase 1 最优 loss，只比较增强策略。

- [ ] 弱增强 / no augmentation 对照：使用现有 `configs/models/resnet18_no_aug.yaml` 和 `exp_002` 结论。
- [ ] ColorJitter light：比默认更保守，避免破坏天气颜色语义。
- [ ] ColorJitter medium：验证颜色扰动是否能提高泛化。
- [ ] RandomRotation 10：默认对照。
- [ ] RandomRotation 20：验证是否过强。
- [ ] RandomErasing small：只作为可选项，观察是否伤害天气区域。
- [ ] RandAugment：只有确认实现后再跑。
- [ ] MixUp / CutMix：只有确认 trainer 实现后再跑，且单独记录。

增强实验顺序：

```text
最优 loss + 默认增强
最优 loss + light ColorJitter
最优 loss + medium ColorJitter
最优 loss + rotation 20
最优 loss + RandAugment
最优 loss + MixUp
最优 loss + CutMix
```

Phase 2 结束判断：

- 增强是否提高 val macro F1。
- rainy / snowy 是否提升，而不是只提升大类。
- 训练/验证 loss 是否更稳定。
- 是否引入过强增强导致 cloudy/sunny 或整体 F1 下降。

## Phase 3 - 小范围组合

只组合前两阶段表现最好的策略：

- [ ] 最优 loss + 最优增强。
- [ ] 最优 loss + 次优增强。
- [ ] 最优 loss + MixUp 或 CutMix 中表现更好的一个。
- [ ] 如果 rainy 仍是瓶颈，再尝试 Weighted FocalLoss。

组合实验不超过 3-4 个，避免实验量爆炸。

## 需要交付给团队的结果

最终输出到：

```text
experiments/loss_aug_results.md
experiments/loss_aug_results.csv
```

至少包含：

- 最优 loss 配置。
- 最优 augmentation 配置。
- rainy / snowy 相比 baseline 的提升。
- cloudy / sunny 是否被牺牲。
- 混淆矩阵中主要错误类型。
- 推荐是否进入最终候选模型。

## 实验记录模板

```markdown
## exp_xxx: ResNet-18 + Loss + Aug

- date:
- branch:
- commit:
- command:
- config:
- loss:
- augmentation:
- image_size: 224
- seed: 42

### Results

| Metric | Value |
| --- | --- |
| val_macro_f1 | |
| cloudy_f1 | |
| rainy_f1 | |
| snowy_f1 | |
| sunny_f1 | |
| best_epoch | |

### Compare to baseline

- macro F1 delta:
- rainy F1 delta:
- snowy F1 delta:
- cloudy/sunny tradeoff:
- conclusion:
```
