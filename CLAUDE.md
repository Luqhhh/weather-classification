# CLAUDE.md — Weather Image Classification

> 智海算法调优 / CAIP 强脑赛道 / 天气图片四分类

## Work Rules

1. **Training MUST use GPU** — 本机有 RTX 4070 (8GB, CUDA 12.6)，训练必须走 GPU，禁止 CPU 训练。
2. **Commit 必须经过用户允许** — 提交前先展示改动内容，得到确认后再 commit。不自动提交。
3. **禁止 Co-Authored-By** — commit message 中不要带 `Co-Authored-By: Claude <noreply@anthropic.com>`。

## Project Overview

Weather image classification competition. 4-class output (cloudy, rainy, snowy, sunny) from phone-style photos. Macro F1 score. CPU-only inference within 70 minutes.

## Quick Commands

```bash
# Data analysis (run FIRST)
python scripts/analyze_data.py --data_dir data/train

# Train
python scripts/train.py --config configs/models/resnet18.yaml --data_dir data/train

# Evaluate
python scripts/evaluate.py --weights weights/resnet18_best.pth --model resnet18 --data_dir data/test

# CPU benchmark
python scripts/benchmark_cpu.py --weights weights/resnet18_best.pth --model resnet18

# Batch inference
python scripts/inference.py --weights weights/resnet18_best.pth --model resnet18 --input_dir data/test --output predictions.csv

# Prepare submission package
python scripts/prepare_submission.py --weights weights/resnet18_best.pth --model resnet18 --label_mapping reports/label_mapping.json

# Run all tests
pytest tests/ -v
```

## Architecture

```
weather-classification/
├── configs/                    # YAML configs (default + per-model)
│   ├── default.yaml            # Base config — all models inherit from this
│   └── models/                 # Model-specific overrides
├── data/                       # Data pipeline
│   ├── dataset.py              # WeatherDataset + create_dataloaders
│   ├── transforms.py           # Train/val/test transforms (conservative augmentation)
│   ├── dataset_report.py       # DatasetAnalyzer + report generation
│   └── label_mapping.py        # LabelMapper — auto-detect from directory structure
├── models/                     # Model definitions
│   ├── base.py                 # WeatherClassifier wrapper (backbone → head)
│   └── model_factory.py        # MODEL_REGISTRY + create_model()
├── training/                   # Training infrastructure
│   ├── trainer.py              # Trainer.fit() — full training loop
│   ├── metrics.py              # compute_macro_f1, per-class metrics, confusion matrix
│   ├── losses.py               # CE, LabelSmoothing, FocalLoss
│   └── callbacks.py            # EarlyStopping, ModelCheckpoint, TrainingLogger
├── inference/                  # Inference & submission
│   ├── predictor.py            # WeatherPredictor — batch CPU inference
│   ├── benchmark.py            # CpuBenchmark — latency/throughput/memory
│   └── submit_checker.py       # SubmitChecker — pre-submission validation
├── scripts/                    # Executable entry points
│   ├── analyze_data.py         # Dataset analysis
│   ├── train.py                # Training
│   ├── evaluate.py             # Model evaluation
│   ├── inference.py            # Batch prediction
│   ├── benchmark_cpu.py        # CPU profiling
│   └── prepare_submission.py   # Submission package builder
├── submit/                     # Final submission directory (generated)
├── tests/                      # pytest tests
├── weights/                    # Saved model weights
├── outputs/                    # Training outputs
├── experiments/                # Experiment logs
└── reports/                    # Dataset reports & analysis
```

## Key Design Decisions

1. **Macro F1 > Accuracy** — The trainer optimizes for macro F1, not accuracy. Per-class metrics tracked.
2. **Conservative augmentation** — ColorJitter ranges kept small (0.15) because weather semantics depend on color/lighting. No extreme rotations.
3. **Auto-detect labels** — NEVER hardcode label order. `LabelMapper` reads the directory structure at runtime.
4. **CPU-first benchmarking** — Every model must pass CPU speed check before submission.
5. **Pre-submission validation** — `SubmitChecker` runs 12 checks before allowing submission.

## Competition Rules (from weather_agent_useful_info.md)

- **Metric**: Macro F1 × 100 (each class equally weighted)
- **Inference**: CPU only, total time ≤ 70 minutes for scoring set (~3000 images)
- **Submission**: Inference code + model weights (one formal submission only)
- **Allowed**: Open-source models, pretrained weights, any libraries, AI coding assistance
- **Forbidden**: External API calls during inference, accessing unauthorized data

## Common Pitfalls

- Hardcoding label order → use `detect_label_mapping()` instead
- Over-augmenting → destroys weather visual cues (color, lighting, sky texture)
- GPU-only model selection → always benchmark on CPU before committing
- 70-minute timeout → weight size, batch size, and image size all matter
- Single formal submission → run full smoke test first!
