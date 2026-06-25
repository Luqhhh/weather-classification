#!/usr/bin/env python3
"""
Model Evaluation Script

Evaluates a trained model on a validation or test dataset.
Generates metrics, confusion matrix, and per-class F1 scores.

Usage:
    python scripts/evaluate.py --weights weights/best.pth --config configs/models/resnet18.yaml
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import WeatherDataset
from data.label_mapping import load_label_mapping, detect_label_mapping
from data.transforms import build_transforms
from models.model_factory import create_model
from training.metrics import compute_metrics, plot_confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment_tracking.tracker import ExperimentTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained weather classification model"
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to model weights (.pth file)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to model config YAML"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Path to evaluation data directory"
    )
    parser.add_argument(
        "--label_mapping", type=str, default=None,
        help="Path to label mapping JSON"
    )
    parser.add_argument(
        "--output_dir", type=str, default="reports",
        help="Directory for output files"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device: 'cpu' or 'cuda'"
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="DataLoader workers for evaluation (defaults to config data.num_workers or 0)"
    )
    parser.add_argument(
        "--experiment_id", type=str, default=None,
        help="Experiment ID (e.g. exp_003). Auto-detected from output_dir name if omitted.",
    )
    parser.add_argument(
        "--notes", type=str, default="",
        help="Free-text notes for the experiment record",
    )
    parser.add_argument(
        "--save_results", action="store_true", default=True,
        help="Save structured results.json alongside other outputs (default: True)",
    )
    parser.add_argument(
        "--no_save_results", action="store_false", dest="save_results",
        help="Do NOT save results.json",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    import yaml
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    data_dir = args.data_dir or data_cfg.get("val_dir") or data_cfg.get("test_dir")

    if not data_dir:
        logger.error("No evaluation data directory specified")
        sys.exit(1)

    # Label mapping
    if args.label_mapping and Path(args.label_mapping).exists():
        label_mapper = load_label_mapping(args.label_mapping)
    else:
        label_mapper = detect_label_mapping(data_dir)

    logger.info(f"Classes: {label_mapper.labels}")

    # Create model
    model = create_model(
        name=model_cfg.get("name", "resnet18"),
        num_classes=label_mapper.num_classes,
        pretrained=False,
    )

    # Load weights
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]

    # Auto-detect FP16 weights → convert to FP32
    sample = next(iter(state.values()))
    if isinstance(sample, torch.Tensor) and sample.dtype == torch.float16:
        logger.info("Detected FP16 weights — converting to FP32")
        state = {k: v.float() if isinstance(v, torch.Tensor) else v
                 for k, v in state.items()}

    model.load_state_dict(state)
    model = model.to(args.device)
    model.eval()
    logger.info(f"Model loaded from {args.weights}")

    # Build transforms and dataset
    image_size = data_cfg.get("image_size", 224)
    mean = tuple(data_cfg.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(data_cfg.get("std", [0.229, 0.224, 0.225]))

    transform = build_transforms(mode="val", image_size=image_size, mean=mean, std=std)
    dataset = WeatherDataset(data_dir=data_dir, transform=transform, label_mapper=label_mapper)

    from torch.utils.data import DataLoader
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = data_cfg.get("num_workers", 0)
    mp_ctx = data_cfg.get("multiprocessing_context", "spawn") or None
    mp_ctx = mp_ctx if num_workers > 0 else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        multiprocessing_context=mp_ctx,
    )

    # Run evaluation
    logger.info(f"Evaluating on {len(dataset)} images...")
    all_preds = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        from tqdm import tqdm
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(args.device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            confs = probs[torch.arange(preds.size(0), device=preds.device), preds]
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_confidences.extend(confs.cpu().numpy().tolist())

    # Compute metrics
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    metrics = compute_metrics(y_true, y_pred, label_mapper.labels)

    # Safety check: all collected arrays must match dataset length
    n_images = len(dataset.images)
    if not (len(all_preds) == len(all_labels) == len(all_confidences) == n_images):
        raise RuntimeError(
            f"Length mismatch after evaluation: "
            f"dataset.images={n_images}, all_preds={len(all_preds)}, "
            f"all_labels={len(all_labels)}, all_confidences={len(all_confidences)}"
        )

    # Collect error samples (misclassified images)
    data_dir_path = Path(data_dir).resolve()
    error_samples = []
    predictions = []
    for (img_path, true_idx), pred_idx, conf in zip(
        dataset.images, all_preds, all_confidences
    ):
        try:
            rel_path = str(img_path.resolve().relative_to(data_dir_path))
        except ValueError:
            rel_path = img_path.name
        predictions.append({
            "filename": rel_path,
            "true_label": label_mapper.decode(true_idx),
            "predicted_label": label_mapper.decode(pred_idx),
            "confidence": round(conf, 6),
            "correct": int(true_idx == pred_idx),
        })
        if true_idx != pred_idx:
            error_samples.append({
                "filename": rel_path,
                "true_label": label_mapper.decode(true_idx),
                "predicted_label": label_mapper.decode(pred_idx),
                "confidence": round(conf, 4),
            })

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {model_cfg.get('name', 'unknown')}")
    print(f"Macro F1:  {metrics['macro_f1']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"\nPer-class metrics:")
    print(f"{'Class':>10s}  {'Precision':>9s}  {'Recall':>9s}  {'F1':>9s}  {'Support':>8s}")
    print("-" * 56)
    for cls_name, cls_metrics in metrics["per_class"].items():
        print(
            f"{cls_name:>10s}  {cls_metrics['precision']:>9.4f}  "
            f"{cls_metrics['recall']:>9.4f}  {cls_metrics['f1']:>9.4f}  "
            f"{cls_metrics['support']:>8d}"
        )
    if metrics["weak_classes"]:
        print(f"\n[WEAK] Weak classes (below avg F1): {', '.join(metrics['weak_classes'])}")
    print("=" * 60)

    # Save confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(
        cm,
        label_mapper.labels,
        save_path=str(output_dir / "confusion_matrix.png"),
        title=f"Confusion Matrix (Macro F1: {metrics['macro_f1']:.4f})",
    )
    logger.info(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")

    # Save error samples CSV
    error_csv_path = output_dir / "error_samples.csv"
    with open(error_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "true_label", "predicted_label", "confidence"])
        writer.writeheader()
        writer.writerows(error_samples)
    logger.info(f"Error samples saved to {error_csv_path} ({len(error_samples)} misclassified)")

    predictions_csv_path = output_dir / "predictions.csv"
    with open(predictions_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "true_label", "predicted_label", "confidence", "correct"],
        )
        writer.writeheader()
        writer.writerows(predictions)
    logger.info("Per-sample predictions saved to %s", predictions_csv_path)

    # --- Save structured results.json ---
    if args.save_results:
        tracker = ExperimentTracker(output_dir)
        result = tracker.build_result(
            config=config,
            evaluation_metrics=metrics,
            notes=args.notes,
            experiment_id=args.experiment_id,
        )
        tracker.save(result)
        logger.info(
            "Experiment result saved (id=%s, macro_f1=%s)",
            result.experiment_id, result.val_macro_f1,
        )


if __name__ == "__main__":
    main()
