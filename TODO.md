# 官方数据阶段 TODO

## 平台待测优先级与权重命名

以下为已经完成本地评估、但 `experiments/officialTestScore.md` 尚无平台 test 分数的新实验。权重统一使用 `outputs/official_xxx/official_xxx_best_model.pth`，对齐旧实验 `official_010/011/012/013` 的命名；ensemble、temperature、class bias、TTA，以及需要显式保存 `image_size` 的 224 单模型 `.pth`，都是 submission bundle，不是裸单模型 `state_dict`。

| 提交优先级 | ID | Local holdout | 推理成本 | 统一权重文件 | 平台动作 |
|------------|----|--------------:|----------|--------------|----------|
| — | — | — | — | — | 暂无 |


`official_024/025/028/030/031/032/035` 已经有平台正式分数；当前平台第一是 `official_025/035`，最强 1x 是 `official_024`。

## 未实现的实验

| 优先级 | ID | 方向 | 配置 / 做法 | 目的 | 状态 |
|--------|----|------|-------------|------|------|
| — | — | — | — | — | 暂无 |

执行约束：

- `official_032` 已完成，不再作为 TODO。
- `official_034` 取消，不做 5-fold；`official_033` 本地 0.9224，不支持继续扩大 fold 成本。
