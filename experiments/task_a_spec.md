# Task A Spec — Backbone & Input Resolution

> Owner: A | 状态: 已定稿 | 2026-06-18

---

## 0. 前置背景

### 0.1 协作流程

```
Round 1（当前）:
  A → 用 CE + 固定增强初筛 backbone，变量干净，纯粹比较模型结构
  B → 并行找最优 loss／augmentation，重点提升 rainy／snowy 少数类

Round 2（后续）:
  B 的最优 loss/aug + A 的 Top 2 backbone → 复跑一轮
  先看 backbone 本身能力，再加不平衡补偿 → 归因清晰
```

### 0.2 评分公式

```
Per-class F1 = 2 × Precision × Recall / (Precision + Recall)
  Precision = 预测为该类的图片中，真正属于该类的比例
  Recall    = 真正属于该类的图片中，被模型找到的比例

Macro F1 = (F1_cloudy + F1_rainy + F1_snowy + F1_sunny) / 4
```

每个类别权重相等，rainy (~1800) 和 snowy (~1500) 远少于 cloudy/sunny (~6700/~6900)，少数类 F1 低会直接拉低总分。

### 0.3 环境

| 阶段 | 安装 | 版本 |
|------|------|------|
| 训练（GPU） | `pip install -r requirements-dev.txt` | `torch 2.1.2+cu121` |
| 推理（CPU） | `pip install torch==2.1.2 torchvision==0.16.2` | `torch 2.1.2` |

---

## 1. 问题定义

在统一训练条件下（loss、augmentation、optimizer、scheduler 全部相同），从 5 个候选 backbone 中筛选 **Macro F1 高 + CPU 推理可接受 + 训练稳定** 的 Top 2~3 模型，然后对 Top 2 做输入尺寸和 dropout 精调。

## 2. 评价标准（优先级降序）

| 优先级 | 指标 | 判据 |
|--------|------|------|
| **P0** | val_macro_f1 | 越高越好，<0.83 直接淘汰 |
| **P1** | rainy_f1 / snowy_f1 | 少数类不能崩，<0.75 标记风险 |
| **P2** | CPU 推理（3000 张） | 必须 ≤ 70 min；>30 min 淘汰 |
| **P3** | 模型大小 | 同等 F1 下越小越好 |
| **P4** | 训练稳定性 | loss 轨迹平滑、train/val gap 合理 |

### 2.1 平局裁决

当两个模型 Macro F1 差 < 0.005 时按以下顺序选：
1. CPU 推理更快的优先
2. 模型更小的优先
3. 训练更稳定（val loss variance 更小）的优先

## 3. 范围边界

**做：**
- 7 个 backbone（含 baseline 的 ResNet-18）在统一配置下跑完
- Top 2 backbone × 4 种输入尺寸（224 / 256 / 320 / 384）
- 最优组合 × 3 种 dropout（0.2 / 0.3 / 0.5）

**不做（这些是 B 的：）**
- CrossEntropy 以外的 loss（FocalLoss、Weighted CE、LabelSmoothing）
- 默认增强以外的 augmentation 变体
- 类别不平衡的任何处理
- lr / batch_size / optimizer 调优（先固定，让实验变量干净）

**不做（本次不考虑：）**
- ViT / Swin / DeiT — CPU 推理压力大，性价比未知
- 模型 ensemble
- 超参网格搜索

## 4. 候选模型

| # | 模型 | 参数量 | 预估权重 | 预估 CPU | GPU 风险 |
|---|------|--------|----------|-----------|----------|
| 0 | ResNet-18 | 11.7M | 42.6 MB | ~9ms | ✅ 低 |
| 1 | ResNet-34 | 21.8M | ~83 MB | ~18ms | ✅ 低 |
| 2 | EfficientNet-B0 | 5.3M | ~21 MB | ~20ms | ✅ 低 |
| 3 | EfficientNet-B1 | 7.8M | ~30 MB | ~25ms | ✅ 低 |
| 4 | EfficientNet-B2 | 9.1M | ~35 MB | ~35ms | ✅ 低 |
| 5 | MobileNetV3-Small | 2.5M | ~10 MB | ~6ms | ✅ 低 |
| 6 | ConvNeXt-Tiny | 28.6M | ~109 MB | ~45ms | ⚠️ batch_size 降到 32 |

> **关于 EfficientNet 系列**：B0/B1/B2 共享相同架构设计，逐步增大宽度/深度/分辨率。加入 B1/B2 可以观察 EfficientNet 在天气分类任务上的 scaling behavior——是否值得为 ~0.01 F1 提升付出 50%+ 的推理时间。
>
> **关于 batch_size**：ConvNeXt-Tiny 参数量大、中间特征图占显存多，64 可能 OOM；出现 OOM 则降为 `--training.batch_size 32`，在 leaderboard 注明即可。这不算改变实验变量——只是硬件适配，不涉及超参调优。

## 5. 统一实验配置

整个 Round 1 只允许变 **model name**、**image_size**、**dropout** 三个变量。

```yaml
# 凝固变量
image_size: 224          # Phase 2 才变
loss: cross_entropy      # 永远不变
augmentation: standard   # 永远不变 (ColorJitter 0.15, RRC, HFlip, Rotation 10°)
dropout: 0.3             # Phase 3 才变
batch_size: 64           # ConvNeXt 例外: 32
epochs: 50               # early stop patience=10 下实际通常 <15
optimizer: adamw
learning_rate: 0.0001
weight_decay: 0.0001
scheduler: cosine (warmup 3 epochs)
seed: 42
```

## 6. 各 Phase 成功标准

### Phase 1 — Backbone 初筛
- [ ] ResNet-34、EfficientNet-B0、EfficientNet-B1、EfficientNet-B2、MobileNetV3-Small、ConvNeXt-Tiny 共 6 个训练完成
- [ ] 每个模型的 evaluate + CPU benchmark 完成
- [ ] 评估矩阵填满，Top 2 明确
- [ ] 排除了 F1 < 0.83 或 CPU > 30min 的模型

### Phase 2 — 输入尺寸对比
- [ ] Top 2 backbone × 3 新尺寸（256/320/384）= 6 个实验跑完
- [ ] 每个 backbone 的最优 image_size 确定
- [ ] CPU 时间 vs F1 的 trade-off 明确（注在 leaderboard）

### Phase 3 — Dropout 调优
- [ ] 最优 backbone + 最优 image_size × 3 个 dropout 值
- [ ] train/val gap 最小的 dropout 确定为最终值
- [ ] backbone_results.md + backbone_results.csv 产出

## 7. 最终产出（给 B 和 C 的交付物）

| 文件 | 内容 |
|------|------|
| `experiments/backbone_results.md` | 排名 + 分析 + 推荐理由 |
| `experiments/backbone_results.csv` | 所有实验的结构化数据 |
| `experiments/leaderboard.md` | 更新所有实验汇总 |
| 推荐列表 | Top 3 候选模型，含完整 metrics |
