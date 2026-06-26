# 官方数据阶段 TODO

## 平台待测优先级与权重命名

以下为已经完成本地评估、但 `experiments/officialTestScore.md` 尚无平台 test 分数的新实验。权重统一使用 `outputs/official_xxx/official_xxx_best_model.pth`，对齐旧实验 `official_010/011/012/013` 的命名；ensemble、temperature、class bias、TTA，以及需要显式保存 `image_size` 的 224 单模型 `.pth`，都是 submission bundle，不是裸单模型 `state_dict`。

| 提交优先级 | ID | Local holdout | 推理成本 | 统一权重文件 | 平台动作 |
|------------|----|--------------:|----------|--------------|----------|
| P0 | official_035 | 0.9439 | 2x ConvNeXt | `outputs/official_035/official_035_best_model.pth` | 可测；025 平台第一后，验证 028 分支替换 018 是否还有微小收益 |
| P1 | official_028 | 0.9423 | 1x ConvNeXt | `outputs/official_028/official_028_best_model.pth` | 可测；024 已证明 018 averaging 平台有效，028 是更密集平均分支 |


`official_024/025/030/031` 已经有平台正式分数；当前平台第一是 `official_025`，最强 1x 是 `official_024`。

## 未实现的实验

| 优先级 | ID | 方向 | 配置 / 做法 | 目的 | 状态 |
|--------|----|------|-------------|------|------|
| 1 | official_032 | train+val 合并重训最终候选 | 用 `official_024` / `official_025` 对应配置，在 `train+val` 上固定 epoch/schedule 重训，保留 holdout 做 sanity check | 用更多官方训练数据提升最终候选；避免依赖 val 早停 | conditional: 只有需要最终冲榜且能接受不可早停风险时做 |
| 2 | official_034 | 5-fold 稳定性训练 | 在 3-fold 有平台正收益时扩展到 5-fold ConvNeXt-Tiny 224 warm-start ensemble | 作为最终冲榜上限探索，换取更高训练和推理成本 | hold: 033 本地 0.9224，当前不建议启动 |

执行约束：

- `official_032` 不使用早停选 epoch；固定轮数参考 `official_024/025` 成员模型的 best epoch / early-stop 轨迹。
- `official_034` 只有在 `official_033` 平台意外转正时才值得继续；按当前本地结果不建议扩大到 5-fold。
