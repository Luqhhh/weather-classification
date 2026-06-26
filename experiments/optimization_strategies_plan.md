# 优化策略 Plan — 冲刺阶段执行计划

> Owner: A | 基于 [spec](optimization_strategies_spec.md) | 2026-06-26

---

## 总览

```
策略 1 — 温度校准 ensemble  ✅ 已完成 (official_035)
策略 2 — Multi-scale 推理   🔜 待跑 (official_036, 037)
策略 3 — 两阶段渐进训练     🔜 待跑 (official_038)
策略 4 — 困难样本重训       🔜 待跑 (official_039)
```

---

## 策略 2 — Multi-Scale 推理 Ensemble

### 执行

使用 `scripts/evaluate_ensemble.py` 的 `--member` 参数传入同一模型的不同尺寸推理。

```bash
# official_036: 双尺度 (224 + 256)
python scripts/evaluate_ensemble.py \
  --member outputs/official_009/best_model.pth:outputs/official_009/config.yaml \
  --data_dir data/holdout \
  --output_dir outputs/official_036/eval_holdout

# 修改 evaluate_ensemble.py 支持 --image_size 参数，或手动改 config 跑两次推理后平均 logits
```

### 临时实现（若 evaluate_ensemble.py 不支持多尺寸）

```bash
# 写临时脚本 scripts/temp_multiscale.py
# 1. 加载 official_018 模型
# 2. 对每个样本分别在 224/256/288 下推理
# 3. logits 平均 → softmax → 预测
# 4. 输出 holdout F1
```

### 判断

- `official_036` (224+256) holdout > 0.928 则优先级提升
- `official_037` (224+256+288) 只在 036 有正收益时做

---

## 策略 3 — 两阶段渐进训练

### 实验 ID: official_038_s1, official_038_s2

**阶段 1 — 常规训练**

```bash
python scripts/train.py \
  --config outputs/official_009/config.yaml \
  --output_dir outputs \
  --experiment_id official_038_s1 \
  --notes "progressive training stage1: ImageNet init, EMA, 224" \
  -- \
  --logging.experiment_name official_038_s1 \
  --model.init_weights null \
  --model.pretrained true
```

**阶段 2 — lr=1e-6 精修**

```bash
# 先复制 s1 的 best_model 作为 s2 的 warm-start
mkdir -p outputs/official_038_s2
cp outputs/official_038_s1/best_model.pth outputs/official_038_s2/init.pth

python scripts/train.py \
  --config outputs/official_009/config.yaml \
  --output_dir outputs \
  --experiment_id official_038_s2 \
  --notes "progressive training stage2: lr=1e-6 warm-start from s1" \
  -- \
  --logging.experiment_name official_038_s2 \
  --model.init_weights outputs/official_038_s2/init.pth \
  --model.pretrained false \
  --training.optimizer.lr 0.000001 \
  --training.epochs 10 \
  --training.scheduler.warmup_epochs 0
```

### 评估

```bash
# s1 holdout
python scripts/evaluate.py --weights outputs/official_038_s1/best_model.pth --config outputs/official_009/config.yaml --data_dir data/holdout

# s2 holdout
python scripts/evaluate.py --weights outputs/official_038_s2/best_model.pth --config outputs/official_009/config.yaml --data_dir data/holdout
```

### 判断

- s2 holdout 超过 s1 ≥ 0.002 → 策略有效，可扩展到 multi-seed
- 否则放弃此方向

---

## 策略 4 — Confidence-based 困难样本重训

### 实验 ID: official_039

**步骤 1 — 找出低置信度样本**

```bash
# 用 018 模型在 val 上推理，记录每个样本的 max probability
python -c "
import torch
from data.dataset import WeatherDataset
from data.transforms import build_transforms
from models.model_factory import create_model
from torch.utils.data import DataLoader

model = create_model('convnext_tiny', num_classes=4, pretrained=False)
model.load_state_dict(torch.load('outputs/official_018/best_model.pth', map_location='cpu', weights_only=True))
model.cuda().eval()

transform = build_transforms(image_size=224)
dataset = WeatherDataset('data/val', transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

all_confidences = []
all_paths = []
with torch.no_grad():
    for imgs, targets in loader:
        logits = model(imgs.cuda())
        probs = torch.softmax(logits, dim=1)
        max_probs, _ = probs.max(dim=1)
        all_confidences.extend(max_probs.cpu().tolist())

# 排序，找出低 20% 置信度阈值
import numpy as np
threshold = np.percentile(all_confidences, 20)
# 保存低置信度样本路径列表
"
```

**步骤 2 — 修改 sampler 支持样本权重**

在 `data/dataset.py` 的 `_build_train_sampler()` 中新增 `confidence_based` 模式：
- 读取低置信度样本列表
- 这些样本的采样权重 × 2
- 其他样本权重 × 1

**步骤 3 — 训练**

```bash
python scripts/train.py \
  --config outputs/official_009/config.yaml \
  --output_dir outputs \
  --experiment_id official_039 \
  --notes "confidence-based hard sample retraining" \
  -- \
  --logging.experiment_name official_039 \
  --training.sampler.name confidence_based \
  --training.sampler.low_conf_threshold 0.2 \
  --training.sampler.low_conf_weight 2.0
```

### 判断

- holdout 超过官方 009 ≥ 0.002 → 策略有效
- 否则放弃此方向

---

## 文件最终结构

```
outputs/
├── official_035/                      ✅ 温度校准 ensemble
│   ├── results.json
│   └── official_035_bundle.json
├── official_036/                      ← 策略 2 双尺度
│   └── eval_holdout/results.json
├── official_037/                      ← 策略 2 三尺度
│   └── eval_holdout/results.json
├── official_038_s1/                   ← 策略 3 阶段1
│   ├── best_model.pth
│   └── results.json
├── official_038_s2/                   ← 策略 3 阶段2
│   ├── init.pth
│   └── results.json
└── official_039/                      ← 策略 4
    └── results.json
```

## 时间估算

| 策略 | 实验 | 预估时间 |
|------|------|----------|
| 2 | official_036 | ~5 min（推理） |
| 2 | official_037 | ~8 min（推理） |
| 3 | official_038_s1 | ~45 min（训练） |
| 3 | official_038_s2 | ~10 min（训练） |
| 4 | official_039 | ~40 min（训练） |
| **合计** | | **约 2 小时** |
