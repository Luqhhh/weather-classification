# 优化策略 Spec — 冲刺阶段新增方向

> Owner: A | 2026-06-26 | 四个低成本优化策略，无需重训或少量训练

---

## 策略 1 — 温度校准 3-fold Ensemble

> ✅ **已完成 (official_035)**

固定 T=2.0，三折 warm-start 模型 logits 除以 T 后 softmax，概率等权平均。
Holdout: 0.9283, rainy: 0.9231。

---

## 策略 2 — Multi-Scale 推理 Ensemble（待做）

### 原理
同一个 ConvNeXt-Tiny 模型在不同分辨率下关注不同尺度的纹理细节。用 224 / 256 / 288 三个分辨率分别推理，logits 等权平均，天然互补。

### 配置
- Base model: `official_009` 或 `official_018` best checkpoint
- 推理尺寸: 224 / 256 / 288
- 方法: logits 平均，不重训
- CPU 成本: 约 3× 单模型推理，预计 < 5 min / 3000 张，仍在 70min 内

### 实验设计
| ID | 实验 | 模型 | 尺寸 | 方法 |
|----|------|------|------|------|
| official_036 | 双尺度 | 009 或 018 | 224 + 256 | logits 平均 |
| official_037 | 三尺度 | 009 或 018 | 224 + 256 + 288 | logits 平均 |

### 判断
- 若 multi-scale holdout 超过单尺度 ≥ 0.002 则进入候选
- 优先双尺度（推理成本 2×）

### 实现要点
- 在 `evaluate.py` 或 `evaluate_ensemble.py` 层面实现，不改训练
- 也可同时加温度校准（如 T=2.0）

---

## 策略 3 — 两阶段渐进训练（待做）

### 原理
不用旧数据 checkpoint (exp_054/044) 做 warm-start，而是用 own best checkpoint 做第二阶段的精修 warm-start。第一阶段常规训练 → 第二阶段用第一阶段的最优权重，保持配置但 lr 降到 1e-6，跑少量 epoch。在 NLP fine-tuning 和 CV competition 中常见。

### 实验设计
| ID | 阶段 | 配置 | lr | epochs |
|----|------|------|----|--------|
| official_038_s1 | 第一阶段 | 009 配置 + ImageNet init, EMA, 224 | 3e-5 | 50（早停） |
| official_038_s2 | 第二阶段 | s1 best checkpoint warm-start, 其余不变 | 1e-6 | 10（固定） |

### 对比基线
- s1 自身的 best checkpoint
- s2 的 best checkpoint + EMA

### 判断
- s2 holdout 超过 s1 ≥ 0.002 则有效
- 若有效，可扩展到 multi-seed (seed 7/42/2026)

---

## 策略 4 — Confidence-based 困难样本重训（待做）

### 原理
用当前最优模型在 val 上预测，找出置信度最低的 20% 样本，在最终 retrain 时给这些样本 2× 采样权重。把算力集中在模型最不确定的边界样本上。

### 实验设计
| ID | 实验 | 配置 | 采样 |
|----|------|------|------|
| official_039 | 困难样本加权 | 009 配置 + ImageNet init, EMA | 低置信度 20% 样本 2× weight |

### 实现要点
- 需先跑一次 inference 获取 val 上每个样本的 max probability
- 修改 `data/dataset.py` 的 sampler 支持 confidence-based weighting
- 不要过度加权（max 2×），避免过拟合少数困难样本

### 判断
- holdout 超过官方 009 ≥ 0.002 则有效
- 只在 retrain_on_full 阶段使用

---

## 策略优先级

| 优先级 | 策略 | 成本 | 预期收益 | 风险 |
|--------|------|------|----------|------|
| P0 | ✅ 温度校准 ensemble | 零 | +0.022 vs 无校准 | 已完成 |
| P1 | Multi-scale 推理 | 零训练 | 0.002~0.005 | CPU 3× |
| P2 | 两阶段渐进训练 | ~15 min | 0.003~0.008 | 低 |
| P3 | 困难样本重训 | ~40 min | 0.001~0.005 | 需确认 low-confidence 样本比例 |

## 不做

- 不扩大搜索 backbone 或 dropout
- 不引入新数据增强
- 不做 5-fold ensemble（033 已证明 3-fold 不如单模型）
- 不做大幅 lr/ema/wd sweep
