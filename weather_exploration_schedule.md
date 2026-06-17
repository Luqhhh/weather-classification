# 天气分类项目探索阶段时间安排

## 文档目的

本文档用于明确天气分类项目当前阶段的团队协作节奏。当前阶段属于**探索阶段**，核心目标不是直接产出最终模型，而是尽快建立统一基线、统一评测流程，并筛选出值得继续精调的候选模型。

当前阶段预期周期：

* 理想完成时间：3 天
* 稳妥完成时间：4 天
* 最多不建议超过：5 天

超过 5 天仍未形成 Top 候选模型时，应停止继续扩展实验，进入收敛调优阶段。

\---

## 阶段目标

当前探索阶段需要产出：

1. 统一的 ResNet-18 baseline；
2. 可复现的数据路径、类别映射和训练配置；
3. 可比较的评测结果，包括 macro F1、per-class F1、混淆矩阵；
4. 初步 backbone 排名；
5. 初步 loss 和 augmentation 候选；
6. CPU 推理时间基准；
7. Top 1 精度候选、Top 1 速度候选、Top 1 稳定候选；
8. 主提交模型和备用提交模型的初步方向。

\---

# 第 0.5 天：统一基线和环境


## 任务

1. 确认数据路径；
2. 确认类别映射；
3. 跑通 `ResNet-18 + CE + 224`；
4. 跑通 `evaluate.py`；
5. 跑通一次 CPU benchmark；
6. 建立 `experiments/leaderboard.md`。

## 推荐 baseline 配置

```text
model = ResNet-18
loss = CrossEntropyLoss
image\_size = 224
augmentation = standard
dropout = default 或 0.2
```

## 产出

`baseline/resnet18\_ce\_224` 需要产出：

1. macro F1；
2. per-class F1；
3. confusion matrix；
4. CPU 推理时间；
5. 模型大小；
6. 训练配置文件；
7. 评测结果文件。

## 注意事项

这一步很关键。没有统一 baseline，后面所有实验都不好比较。

\---

# 第 1\~2 天：A/B/C 并行探索

## A：Backbone 初筛

A 不要一开始就调很多超参，先固定配置跑模型。

### 建议先跑的模型

1. ResNet-18；
2. ResNet-34；
3. EfficientNet-B0；
4. MobileNetV3-Small；
5. ConvNeXt-Tiny。

### 统一配置

```text
image\_size = 224
loss = CE
augmentation = standard
dropout = default 或 0.2
```

### A 第 2 天结束前产出

1. backbone 初步排名；
2. Top 2 backbone；
3. 明显不值得继续调的模型；
4. 每个模型的 macro F1；
5. 每个模型的 rainy F1 / snowy F1；
6. 每个模型的初步 CPU 推理时间；
7. 每个模型的训练配置和结果记录。

### A 的实验记录建议

每个实验都应至少记录：

```text
experiment\_id:
branch:
commit\_hash:
model:
image\_size:
loss:
augmentation:
dropout:
macro\_f1:
cloudy\_f1:
rainy\_f1:
snowy\_f1:
sunny\_f1:
cpu\_time:
model\_size:
conclusion:
```

\---

## B：Loss 和 Augmentation 初筛

B 不要一开始组合增强。先基于 ResNet-18 跑 loss。

## Phase 1：Loss 对比

基于公共 ResNet-18 baseline，对比：

1. CE；
2. FocalLoss；
3. LabelSmoothing；
4. Weighted CE。

### 目标

观察不同 loss 是否能提升整体 macro F1，以及是否能改善 rainy / snowy 少数类表现。

\---

## Phase 2：Augmentation 对比

固定表现最好的 loss，再试增强策略。

建议对比：

1. 弱增强 baseline；
2. ColorJitter；
3. RandomRotation；
4. RandAugment；
5. MixUp；
6. CutMix。

### 注意事项

不要一开始做复杂组合，例如：

```text
FocalLoss + LabelSmoothing + Weighted CE + RandAugment + MixUp + CutMix
```

这种组合不好归因，也容易实验量爆炸。

建议顺序是：

```text
先比较 loss
再固定最优 loss 比较 augmentation
最后只组合 Top 2 策略
```

### B 第 2 天结束前产出

1. 最优 loss 候选；
2. 最优 augmentation 候选；
3. rainy / snowy 是否有提升；
4. 是否牺牲 cloudy / sunny；
5. 推荐进入下一阶段的训练策略。

\---

## C：评测和交付管线

C 这两天不等 A/B，直接做工具和评测闭环。

## 必须完成

1. `evaluate.py` 输出 macro F1；
2. `evaluate.py` 输出 per-class F1；
3. 生成 `confusion\_matrix.png`；
4. 生成 `wrong\_samples.csv`；
5. 完成 `benchmark\_cpu.py`；
6. 建立 `leaderboard.md` 或 `results.csv`；
7. 能统一汇总 A/B 的实验结果。

## 推荐输出文件

```text
experiments/leaderboard.md
experiments/results.csv
experiments/benchmark\_results.csv
experiments/exp\_xxx/result.yaml
experiments/exp\_xxx/confusion\_matrix.png
experiments/exp\_xxx/wrong\_samples.csv
```

## wrong\_samples.csv 建议字段

```csv
image\_path,true\_label,pred\_label,confidence
```

## C 第 2 天结束前产出

1. 统一 leaderboard；
2. 所有已完成实验的 macro F1 / per-class F1；
3. 初步 CPU benchmark；
4. 错误样本统计；
5. 混淆矩阵；
6. 对 A/B 下一步实验的建议。

\---

# 第 3 天：第一次收敛

第 3 天不要再盲目增加模型，而是整理结果。

## 需要选出的候选

你们要选出：

1. Top 1 精度候选；
2. Top 1 速度候选；
3. Top 1 稳定候选。

## 示例

```text
精度候选：ConvNeXt-Tiny / EfficientNet-B0
速度候选：MobileNetV3-Small / ResNet-18
稳定候选：ResNet-34 / EfficientNet-B0
```

## 小范围调参

只对 Top 2 做小范围调参。

建议范围：

```text
image\_size: 224 / 256 / 320
dropout: 0.2 / 0.3
loss: 当前最优 loss
augmentation: 当前最优增强
```

## 禁止事项

不要再跑完整网格，例如：

```text
5 个模型 × 4 个输入尺寸 × 3 个 dropout × 4 个 loss × 6 个增强
```

这种搜索空间太大，不适合当前阶段。

## 第 3 天结束前产出

1. Top 3 候选模型；
2. Top 2 精调方向；
3. 是否继续保留大模型；
4. 是否需要放弃某些低性价比模型；
5. 下一阶段精调计划。

\---

# 第 4 天：确定主模型和备用模型

第 4 天应该进入决策。

C 汇总最终决策矩阵，A/B 根据结果补充最后一轮小范围实验。

## 决策矩阵模板

|模型|Macro F1|rainy F1|snowy F1|CPU 时间|模型大小|结论|
|-|-:|-:|-:|-:|-:|-|
|EfficientNet-B0|高|高|中|中|小|主候选|
|ConvNeXt-Tiny|最高|高|高|慢|大|精度候选|
|MobileNetV3-Small|中|中|中|快|小|速度备用|
|ResNet-34|稳|中|中|中|中|稳定备用|

## 第 4 天结束前必须确定

1. 主提交模型；
2. 备用提交模型；
3. 最终训练配置；
4. 最终推理配置；
5. 最终提交路线；
6. 是否需要 ONNX；
7. 是否需要压缩或量化；
8. 是否需要重新训练最终模型。

\---

# 每日同步机制

建议每天至少同步一次，格式如下：

```text
日期：
负责人：
今日完成：
实验编号：
模型：
配置：
macro F1：
rainy F1：
snowy F1：
CPU 时间：
遇到的问题：
明日计划：
是否建议进入 leaderboard：
```

\---

# Leaderboard 模板

建议维护 `experiments/leaderboard.md`。

|Rank|Experiment|Model|Image Size|Loss|Augmentation|Macro F1|Rainy F1|Snowy F1|CPU Time|Model Size|Conclusion|
|-:|-|-|-:|-|-|-:|-:|-:|-:|-:|-|
|1|exp\_008|EfficientNet-B0|256|FocalLoss|RandAugment||||||主候选|
|2|exp\_011|ConvNeXt-Tiny|256|CE|RandAugment||||||精度高但可能慢|
|3|exp\_003|ResNet-34|224|CE|standard||||||稳定备用|

\---

# Branch 和 PR 建议

不要按人长期创建 branch，例如：

```text
A
B
C
```

建议按实验或功能创建 branch：

```text
exp/resnet18-baseline
exp/backbone-convnext-tiny
exp/backbone-efficientnet-b0
exp/loss-focal
exp/aug-randaugment
feat/confusion-matrix
feat/error-analysis
feat/cpu-benchmark
feat/submission-smoke-test
```

## PR 内容要求

每个 PR 至少包含：

1. 改动目的；
2. 对应实验编号；
3. 训练配置；
4. macro F1；
5. per-class F1；
6. CPU 时间；
7. 是否更新 leaderboard；
8. 是否建议 merge；
9. 结论。

## 哪些内容应该 merge 到 main

建议 merge：

1. 稳定代码；
2. 通用脚本；
3. 配置文件；
4. 实验结果摘要；
5. leaderboard；
6. 提交相关代码。

不建议 merge：

1. 明显失败的临时代码；
2. 大体积权重文件；
3. 未验证的路径修改；
4. 本地绝对路径；
5. 只在个人电脑能跑的代码。

\---

# 当前阶段完成标准

当满足以下条件时，可以认为探索阶段完成：

1. ResNet-18 baseline 已跑通；
2. 至少 4 个 backbone 已完成初筛；
3. 至少 3 种 loss 已完成对比；
4. 至少 3 种 augmentation 已完成对比；
5. `leaderboard.md` 已建立；
6. CPU benchmark 已完成；
7. Top 3 候选模型已确定；
8. 主模型和备用模型方向已确定。

\---

# 当前阶段之后的工作

探索阶段完成后，进入**收敛调优阶段**。

建议重新分工：

|人员|角色|任务|
|-|-|-|
|A|主模型精调|围绕 Top 1 模型调 learning rate、weight decay、epoch、scheduler、dropout、image size|
|B|错误修正与泛化|针对 rainy/snowy、cloudy/sunny 混淆做类别权重、重采样、增强策略修正|
|C|最终评测与提交|统一重测 Top 3、CPU benchmark、3000 张模拟推理、打包提交、冒烟测试|

\---

# 总结

当前阶段建议控制在 3\~4 天内完成。

核心原则：

```text
先统一 baseline，再并行探索；
先做可比较实验，再做局部精调；
先选 Top 候选，再进入最终收敛；
不要无止境增加实验组合。
```

最终目标不是跑最多实验，而是在有限时间内找到：

1. 一个主提交模型；
2. 一个备用提交模型；
3. 一套稳定可复现的训练和评测流程。

