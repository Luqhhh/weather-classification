# Weather Image Classification

本项目面向天气图片四分类任务，目标是把普通户外天气照片识别为 `cloudy`、`rainy`、`snowy`、`sunny` 四类。项目重点不只是训练一个高分模型，也包括验证集可靠性、少数类表现、泛化稳定性，以及最终在 CPU 推理约束下可提交。

## 项目简介

当前仓库包含一套完整的天气分类实验工程：

- 数据读取、标签映射、训练集和验证集构建。
- 常用视觉 backbone 的统一模型封装。
- 训练、评估、推理和提交检查流程。
- 面向 macro F1 的实验记录和对比。
- 适配本地 GPU 训练与官方 CPU 推理环境的工程约束。

这个项目已经进入官方数据收敛阶段：前期先在公开或本地数据上建立稳定实验流程，官方数据发布后围绕 warm-start、checkpoint averaging、轻量 logits ensemble、TTA、train+val 重训等方向做小范围验证，最终根据平台 test macro F1 和推理成本选择提交候选。

## 我们已经完成的工作

我们首先整理了数据和验证流程，重点检查了类别分布、重复图片、训练集和验证集泄漏风险，并让验证划分更稳定。这样后续实验比较不只是看单次分数，而是尽量减少数据问题带来的误判。

训练工程上，我们把模型、loss、优化器、数据增强、EMA、checkpoint averaging、评估输出等能力接入到统一配置和脚本中。这样每次实验可以留下可追踪的配置、日志、结果和错误样本，便于复盘。

模型优化上，我们围绕当前主力 backbone 做了多轮小范围实验，重点关注正则化强度、少数类表现、不同 seed 的稳定性，以及 EMA、SWA、checkpoint averaging、轻量 ensemble 这类更偏泛化的策略。整体结论是：继续盲目扩大模型或增强收益有限，更值得保留的是稳定单模型候选和少量低成本集成方案。

评估记录上，我们把平台正式 test 分数、实验队列、榜单、阶段性发现和分工文档放在 `experiments/` 下，便于团队同步当前结论和下一步优先级。`experiments/officialTestScore.md` 是平台分数 ledger，`experiments/official_leaderboard.md` 是当前官方阶段总结。

## 当前阶段

当前阶段已经从“大范围探索”收敛到最终候选选择。平台正式 test macro F1 的关键结果如下：

| 候选 | 方案 | 平台 test macro F1 | 推理成本 | 结论 |
|------|------|-------------------:|----------|------|
| `official_025` | `0.7*official_018 + 0.3*official_004` logits ensemble | **0.947611** | 2x ConvNeXt | 默认最终提交候选 |
| `official_035` | `0.7*official_028 + 0.3*official_004` logits ensemble | **0.947611** | 2x ConvNeXt | 与 025 持平，未带来新增益 |
| `official_024` | `official_018` top-3 checkpoint averaging | 0.946377 | 1x ConvNeXt | 最强低成本单模型 |
| `official_032` | train+val fixed-schedule retrain | 0.945458 | 1x ConvNeXt | 合并重训未超过 024/025 |

如果评分集没有严格推理时间限制，默认提交 `outputs/official_025/official_025_best_model.pth`。如果推理时间或成本更重要，提交 `outputs/official_024/official_024_best_model.pth`。

## 仓库内容

```text
configs/      模型和训练配置
data/         本地数据目录，不提交到 git
models/       模型封装和创建逻辑
training/     训练循环、loss、metrics、callbacks
inference/    推理和提交检查
scripts/      训练、评估、分析和提交相关脚本
experiments/  实验队列、榜单、发现和分工
outputs/      实验输出，权重文件通常不进 git
reports/      评估报告和分析结果
tests/        测试和冒烟检查
```

## 后续方向

当前没有必要继续增加新实验。已有验证显示：

- `official_035` 与 `official_025` 平台持平，说明 018/028 分支替换没有进一步收益。
- `official_032` 的 train+val 合并重训没有超过 024/025。
- `official_030` 的 class bias 在 holdout 上有效，但平台上没有收益。
- 3-fold / 5-fold 方向成本高，且已有 3-fold 本地结果不支持继续扩展。

后续只保留两类动作：确认最终提交模型的 CPU 推理时间，以及在平台规则变化或新增数据时重新同步 `experiments/officialTestScore.md` 和 `experiments/official_leaderboard.md`。
