# Weather Classification 团队协作方案

## 1. 项目目标

本项目面向天气图片四分类任务：

- cloudy
- rainy
- snowy
- sunny

比赛核心目标是提升分类效果，主要关注：

1. **Macro F1**：优先级最高，决定主要排名。
2. **CPU 推理时间**：同分时会影响排名。
3. **资源占用与代码效率**：包括模型大小、内存占用、CPU/GPU 利用率等。
4. **提交稳定性**：提交包必须能在平台环境中独立运行。

因此，团队协作不应该只围绕“谁训练出来的模型分数高”，而应该围绕：

```text
统一框架 + 统一验证集 + 统一实验记录 + 统一提交流程
```

---

## 2. 总体协作原则

### 2.1 main 分支原则

`main` 分支只保留稳定、可运行、可提交的版本。

所有人不要直接向 `main` 推送代码，应通过 feature/experiment branch 开发，再合并。

推荐流程：

```bash
git checkout main
git pull origin main
git checkout -b exp/resnet18-baseline
```

完成实验后：

```bash
git add .
git commit -m "exp: add resnet18 baseline result"
git push origin exp/resnet18-baseline
```

然后发起 PR 或由队内统一 review 后合并。

---

### 2.2 branch 命名规范

推荐按任务命名：

```text
exp/resnet18-baseline
exp/backbone-resnet34
exp/backbone-convnext-tiny
exp/backbone-efficientnet-b0
exp/loss-focal
exp/loss-label-smoothing
exp/aug-randaugment
feat/confusion-matrix
feat/error-analysis
feat/cpu-benchmark
feat/submission-smoke-test
fix/label-mapping
```

这样可以避免三个人各自开发成三个互相冲突的版本。

---

## 3. 三人分工总览

| 成员 | 方向                        | 核心职责                                  | 主要产出                                       |
| ---- | --------------------------- | ----------------------------------------- | ---------------------------------------------- |
| A    | Backbone & Input Resolution | 探索模型架构、输入尺寸、dropout           | backbone 排名、最优 image size、候选模型 Top 3 |
| B    | Loss & Augmentation         | 优化 loss、类别不平衡、数据增强策略       | 最优 loss、最优 augmentation、少数类提升情况   |
| C    | Evaluation & Submission     | 评测闭环、CPU benchmark、错误分析、提交包 | leaderboard、决策矩阵、最终提交包              |

整体逻辑：

```text
A 负责找模型上限
B 负责提高泛化和少数类 F1
C 负责保证实验可比较、模型可提交、推理够快
```

---

# 4. A — Backbone & Input Resolution

## 4.1 目标

找到最值得继续调优的 backbone，并确定合理的输入尺寸和 dropout。

A 的重点不是盲目堆大模型，而是找到：

```text
Macro F1 高 + CPU 推理能接受 + 模型稳定
```

的候选方案。

---

## 4.2 Phase 0：确认公共 baseline

先跑通公共基线：

```text
ResNet-18 + CrossEntropyLoss + image_size=224 + 标准增强
```

记录：

- macro F1
- per-class F1
- cloudy F1
- rainy F1
- snowy F1
- sunny F1
- CPU 推理时间
- 模型大小
- 训练配置
- commit hash

建议实验名：

```text
exp_001_resnet18_ce_224
```

---

## 4.3 Phase 1：backbone 初筛 ✅ 已完成

统一使用相同训练策略，已完成 backbone 筛选。

结果：**ConvNeXt-Tiny (0.9071) 与 EfficientNet-B1 (0.9014) 进入 Phase 2**。详见 `experiments/leaderboard.md` 子榜 A。

---

## 4.4 Phase 2：输入尺寸对比

只对 Phase 1 中表现最好的 Top 1 或 Top 2 backbone 做输入尺寸对比。

候选输入尺寸：

- 224
- 256
- 320
- 384

记录重点：

- Macro F1 是否明显提升
- rainy/snowy 是否提升
- CPU 推理时间是否明显变慢
- 显存占用是否明显增加

注意：

如果 `384` 只比 `320` 高很少，但推理慢很多，不一定值得选。

---

## 4.5 Phase 3：dropout 调优

只在最优 backbone + 最优输入尺寸上调 dropout。

候选值：

- 0.2
- 0.3
- 0.5

判断逻辑：

```text
如果训练集分数高、验证集分数低：适当增大 dropout
如果训练集和验证集都低：可能不是过拟合，增大 dropout 可能无效
如果验证集已经稳定：不必过度调 dropout
```

---

## 4.6 A 的最终产出

A 需要输出：

1. backbone 排名
2. 最优 image size
3. 最优 dropout
4. 候选模型 Top 3
5. 每个候选模型的 CPU 推理时间
6. 推荐最终是否采用该模型

建议结果文件：

```text
experiments/backbone_results.md
experiments/backbone_results.csv
```

---

# 5. B — Loss & Augmentation

## 5.1 目标

让模型更适配类别不平衡数据，并提升泛化能力。

B 的重点是：

```text
提升 rainy / snowy 少数类 F1，同时不明显牺牲 cloudy / sunny
```

---

## 5.2 Phase 1：loss 对比 ✅ 已完成

基于 ResNet-18 + 224 + 标准增强，已完成 4 种 loss 对比（exp_010~013）。

结果：**LabelSmoothing ε=0.1 最优**（F1 0.8966，rainy 0.8649）。详见 `experiments/leaderboard.md` 子榜 B。

| 排名 | Loss | F1 | rainy F1 |
|------|------|-----|----------|
| 1 | LabelSmoothing | 0.8966 | 0.8649 |
| 2 | FocalLoss γ=2.0 | 0.8847 | 0.8590 |
| 3 | Weighted CE balanced | 0.8841 | 0.8522 |
| 4 | Weighted CE sqrt | 0.8791 | 0.8409 |
| — | CE baseline | 0.8708 | 0.8240 |

---

## 5.3 Phase 2：增强策略对比

固定 Phase 1 中最优 loss，再比较增强策略。

候选增强：

- 弱增强 baseline
- ColorJitter light
- ColorJitter medium
- RandomRotation 10°
- RandomRotation 20°
- RandAugment
- MixUp
- CutMix

建议不要一次性叠加太多增强。

推荐实验顺序：

```text
弱增强 baseline
ColorJitter light
ColorJitter medium
RandomRotation 10°
RandomRotation 20°
RandAugment
MixUp
CutMix
```

---

## 5.4 Phase 3：组合实验

只组合前两阶段表现最好的策略。

推荐组合：

```text
最优 loss + 最优增强
最优 loss + 次优增强
最优 loss + MixUp/CutMix 中表现更好的一个
```

不要做无控制的大量组合，例如：

```text
FocalLoss + LabelSmoothing + 类别权重 + RandAugment + MixUp + CutMix
```

这种组合很难归因，也容易导致训练不稳定。

---

## 5.5 B 的最终产出

B 需要输出：

1. 最优 loss 配置
2. 最优 augmentation 配置
3. rainy / snowy 少数类提升情况
4. 是否牺牲 cloudy / sunny
5. 错误样本观察
6. 是否推荐进入最终候选模型

建议结果文件：

```text
experiments/loss_aug_results.md
experiments/loss_aug_results.csv
```

---

# 6. C — Evaluation & Submission

## 6.1 目标

让所有实验可比较，并保证最终提交稳定。

C 的重点是：

```text
实验记录统一 + 错误分析清楚 + CPU 性能可控 + 提交包能跑
```

---

## 6.2 Phase 1：评测基础设施

在 A/B 训练期间，C 可以立即并行做。

需要完善：

- `evaluate.py` 输出 macro F1
- `evaluate.py` 输出 per-class F1
- 输出 confusion matrix
- 保存错误样本列表
- 保存错误样本可视化结果
- 自动汇总 `experiments/` 下的实验结果
- 生成 `leaderboard.md` 或 `results.csv`

推荐输出字段：

```text
experiment_id
branch
commit_hash
model
image_size
loss
augmentation
dropout
batch_size
val_macro_f1
cloudy_f1
rainy_f1
snowy_f1
sunny_f1
cpu_time_per_image
model_size_mb
submit_check_passed
notes
```

---

## 6.3 Phase 2：CPU benchmark

等 A/B 产出模型后，C 统一做 CPU 性能评测。

评测内容：

- 单张平均推理时间
- batch size 对比：8 / 16 / 32 / 64
- 3000 张图全量模拟计时
- 模型大小统计
- 峰值内存统计，能做则做

重点判断：

```text
F1 高但 CPU 明显慢的模型，不一定适合作为最终提交
F1 接近时，优先选择 CPU 更快、提交风险更低的模型
```

---

## 6.4 Phase 3：提交可靠性

提交前必须检查：

- `prepare_submission.py` 能正常打包
- 权重文件存在
- 类别顺序正确
- 推理代码没有写死本地绝对路径
- 依赖没有缺失
- CPU-only 环境能运行
- 独立目录冒烟测试通过
- 预测输出格式符合平台要求
- 模型加载路径正确
- label mapping 与训练阶段一致
- 不包含无关大文件
- 最终 zip 能被重新解压并运行

---

## 6.5 C 的最终产出

C 需要输出：

1. 实验 leaderboard
2. 错误样本分析报告
3. CPU benchmark 报告
4. 最终决策矩阵
5. 候选提交包 Top 3
6. 正式提交包
7. 提交前 checklist

建议结果文件：

```text
experiments/leaderboard.md
reports/error_analysis.md
reports/cpu_benchmark.md
reports/final_decision.md
submit/final_submission.zip
```

---

# 7. 实验记录模板

每个实验建议记录为一个 `result.yaml` 或 `result.json`。

示例：

```yaml
experiment_id: exp_001_resnet18_ce_224
owner: A
branch: exp/resnet18-baseline
commit_hash: ""
model: resnet18
image_size: 224
loss: cross_entropy
augmentation: standard
dropout: 0.2
batch_size: 32
epochs: 30
optimizer: adamw
learning_rate: 0.0003

metrics:
  val_macro_f1: null
  cloudy_f1: null
  rainy_f1: null
  snowy_f1: null
  sunny_f1: null

performance:
  cpu_time_per_image_ms: null
  full_3000_images_time_s: null
  model_size_mb: null
  peak_memory_mb: null

submission:
  submit_check_passed: false
  smoke_test_passed: false

notes: ""
```

---

# 8. PR / 合并检查清单

每个实验分支合并前，需要满足：

- [ ] config 文件已提交
- [ ] result.yaml 或 result.json 已提交
- [ ] leaderboard 已更新
- [ ] 记录了 macro F1
- [ ] 记录了 per-class F1
- [ ] 记录了 CPU 推理时间，若还没测需注明
- [ ] 没有提交无关大文件
- [ ] 没有写死本地绝对路径
- [ ] 类别顺序没有改错
- [ ] 简短写明实验结论

PR 描述模板：

```text
实验名称：
负责人：
branch：
commit hash：

改动内容：

模型配置：
- model:
- image_size:
- loss:
- augmentation:
- dropout:

结果：
- val_macro_f1:
- cloudy_f1:
- rainy_f1:
- snowy_f1:
- sunny_f1:
- cpu_time_per_image:
- model_size:

结论：
是否推荐进入候选模型：是 / 否
```

---

# 9. 最终决策矩阵

最终不要只看 F1，需要综合：

- Macro F1
- rainy F1
- snowy F1
- CPU 推理时间
- 模型大小
- 提交稳定性
- 代码风险

推荐表格：

| 模型              | Macro F1 | cloudy F1 | rainy F1 | snowy F1 | sunny F1 | CPU 时间 | 模型大小 | 风险 | 结论     |
| ----------------- | -------: | --------: | -------: | -------: | -------: | -------: | -------: | ---- | -------- |
| ResNet-18         |          |           |          |          |          |          |          | 低   | 稳定基线 |
| ResNet-34         |          |           |          |          |          |          |          | 中   | 待比较   |
| EfficientNet-B0   |          |           |          |          |          |          |          | 中   | 待比较   |
| MobileNetV3-Small |          |           |          |          |          |          |          | 低   | 速度候选 |
| ConvNeXt-Tiny     |          |           |          |          |          |          |          | 高   | 精度候选 |

最终选择逻辑：

```text
如果某模型 F1 明显最高，且 CPU 时间可接受，选它。
如果 Top 2 F1 很接近，选 CPU 更快、提交风险更低的。
如果大模型只高 0.2~0.5 分，但慢很多，不建议选大模型。
如果 rainy/snowy 明显崩，不建议选，即使总体 F1 还行。
```

---

# 10. 当前进度与后续执行

> 详细实验队列：`experiments/experiment_queue.md`

## Phase 1 ✅ 已完成

- **A**：Backbone 初筛完成（exp_001~009），ConvNeXt-Tiny (0.9071) + EfficientNet-B1 (0.9014) 进入下一阶段
- **B**：Loss 对比完成（exp_010~013），LabelSmoothing ε=0.1 最优（0.8966）
- **C**：Leaderboard + experiment_queue 已建立

## Phase 2 🔜 当前 — 三组可并行

| 组 | 内容 | 数量 | 
|----|------|------|
| A | Top 2 backbone × 输入尺寸（256/320/384） | 5 |
| B | 固定 LabelSmoothing，对比 Core Augmentation | 5 |
| C | MixUp / CutMix + ConvNeXt 大尺寸 | 3 |

## Phase 3 🔜

| 组 | 内容 | 数量 |
|----|------|------|
| A | — | 0 |
| B | 最优 loss + 最优 augmentation 组合 | ≤3 |
| C | ConvNeXt dropout 调优（0.2/0.4/0.5） | 3 |

---

## Final Day

目标：确定最终提交。

- A：提供候选模型 Top 3
- B：提供最终训练策略建议
- C：完成 CPU benchmark、提交包检查、最终打包

产出：

```text
final_decision.md
final_submission.zip
backup_submission.zip
```

---

# 11. 最终结论

本项目推荐采用：

```text
A：Backbone & Input Resolution
B：Loss & Augmentation
C：Evaluation & Submission
```

这种分工可以让三个人同时推进，并且避免实验混乱。

最重要的是：

```text
不要只比较谁的平台提交分数高，必须用统一验证集、统一 leaderboard、统一 CPU benchmark 来选最终方案。
```

最终目标不是训练出一个“看起来最强”的模型，而是提交一个：

```text
F1 高、CPU 快、代码稳、平台能跑
```

的完整方案。
