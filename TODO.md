# 官方数据阶段 TODO

## 平台待测优先级与权重命名

以下为已经完成本地评估、但 `experiments/officialTestScore.md` 尚无平台 test 分数的新实验。权重统一使用 `outputs/official_xxx/official_xxx_best_model.pth`，对齐旧实验 `official_010/011/012/013` 的命名；ensemble、temperature、class bias、TTA，以及需要显式保存 `image_size` 的 224 单模型 `.pth`，都是 submission bundle，不是裸单模型 `state_dict`。

| 提交优先级 | ID | Local holdout | 推理成本 | 统一权重文件 | 平台动作 |
|------------|----|--------------:|----------|--------------|----------|
| P0 | official_030 | **0.9541** | 2x ConvNeXt | `outputs/official_030/official_030_best_model.pth` | 先测；唯一正向后处理信号，但有 holdout bias 拟合风险 |
| P1 | official_025 | 0.9439 | 2x ConvNeXt | `outputs/official_025/official_025_best_model.pth` | 可测；用平台强的 018 替换 013 中的 009，验证 004 互补是否仍成立 |
| P2 | official_031 | 0.9431 | 2x TTA | `outputs/official_031/official_031_best_model.pth` | 可测；018 flip TTA 小幅高于 018/024，若平台时间允许作为单模型 TTA 对照 |
| P3 | official_024 | 0.9423 | 1x ConvNeXt | `outputs/official_024/official_024_best_model.pth` | 低优先；1x 成本但本地没有超过 018，仅作 averaging sanity check |
| Skip | official_026 | 0.9423 | 2x ConvNeXt | `outputs/official_026/official_026_best_model.pth` | 暂不测；被 025 支配 |
| Skip | official_027 | 0.9418 | 3x ConvNeXt | `outputs/official_027/official_027_best_model.pth` | 暂不测；成本高且本地低于 025/031 |
| Skip | official_029 | 0.9418 | 3x ConvNeXt | `outputs/official_029/official_029_best_model.pth` | 暂不测；temperature calibration 未带来本地收益 |

`official_014/023` 是旧 pending 项，不属于本轮新增优先级。

## 未实现的实验

| 优先级 | ID | 方向 | 配置 / 做法 | 目的 | 状态 |
|--------|----|------|-------------|------|------|
| 1 | official_028 | SWA / EMA 后期平均 | 基于 `official_018`，尝试最后若干 epoch 的 SWA，或对后期 checkpoint 做比 top-3 更密集的权重平均 | 验证比 `official_024` 简单 top-3 averaging 更稳定的 1x 单模型候选 | todo |
| 2 | official_032 | train+val 合并重训最终候选 | 用 `official_018` 单模型配置，在 `train+val` 上固定 epoch/schedule 重训，保留 holdout 做 sanity check | 用更多官方训练数据提升最终单模型；避免依赖 val 早停 | conditional: 024/028 或平台单模型方向确认后做 |
| 3 | official_033 | 3-fold 稳定性训练 | 3-fold ConvNeXt-Tiny 224 warm-start；每折固定 schedule，最终做 fold logits ensemble | 系统降低 split 方差，作为最终冲榜前的中等成本 ensemble 验证 | conditional: 后处理收益耗尽后再做 |
| 4 | official_034 | 5-fold 稳定性训练 | 在 3-fold 有平台正收益时扩展到 5-fold ConvNeXt-Tiny 224 warm-start ensemble | 作为最终冲榜上限探索，换取更高训练和推理成本 | conditional: 033 平台收益明确后做 |

执行约束：

- `official_032` 不使用早停选 epoch；固定轮数参考 `official_018` 的 best epoch / early-stop 轨迹。
- `official_033` 和 `official_034` 不再混成一个实验；先验证 3-fold，再决定是否承担 5-fold 成本。
