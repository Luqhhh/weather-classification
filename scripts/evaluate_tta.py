#!/usr/bin/env python3
"""Evaluate a single model with deterministic center-crop + horizontal-flip TTA."""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import WeatherDataset
from data.label_mapping import detect_label_mapping, load_label_mapping
from experiment_tracking.tracker import ExperimentTracker
from models.model_factory import create_model
from training.metrics import compute_metrics, plot_confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class TtaDataset(Dataset):
    """Return stacked deterministic TTA views for each image."""

    def __init__(self, base: WeatherDataset, views: list[transforms.Compose]):
        self.base = base
        self.views = views
        self.images = base.images

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label_idx = self.base.images[idx]
        image = self.base._safe_load_image(img_path)
        if image is None:
            logger.error("Failed to load image at runtime: %s", img_path)
            image = Image.new("RGB", (224, 224), (128, 128, 128))
        stacked = torch.stack([view(image) for view in self.views], dim=0)
        return stacked, label_idx


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_state(path: Path) -> dict:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    return state


def _build_tta_views(image_size: int, mean: tuple, std: tuple, crop_pct: float) -> list:
    resize_size = max(image_size, int(round(image_size / crop_pct)))
    base = [
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
    ]
    tail = [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return [
        transforms.Compose(base + tail),
        transforms.Compose(base + [transforms.RandomHorizontalFlip(p=1.0)] + tail),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate single-model TTA")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--label_mapping", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--crop_pct", type=float, default=0.875)
    parser.add_argument("--experiment_id", type=str, default=None)
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(Path(args.config))
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    data_dir = args.data_dir or data_cfg.get("val_dir") or data_cfg.get("test_dir")
    if not data_dir:
        raise ValueError("No evaluation data_dir specified")

    if args.label_mapping and Path(args.label_mapping).exists():
        label_mapper = load_label_mapping(args.label_mapping)
    else:
        label_mapper = detect_label_mapping(data_dir)

    model = create_model(
        name=model_cfg.get("name", "resnet18"),
        num_classes=label_mapper.num_classes,
        pretrained=False,
        dropout=float(model_cfg.get("dropout", 0.3)),
    )
    model.load_state_dict(_load_state(Path(args.weights)))
    model = model.to(args.device).eval()

    image_size = int(data_cfg.get("image_size", 224))
    mean = tuple(data_cfg.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(data_cfg.get("std", [0.229, 0.224, 0.225]))
    views = _build_tta_views(image_size, mean, std, args.crop_pct)
    base_dataset = WeatherDataset(data_dir=data_dir, transform=None, label_mapper=label_mapper)
    dataset = TtaDataset(base_dataset, views)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    all_preds: list[int] = []
    all_labels: list[int] = []
    all_confidences: list[float] = []
    start = time.perf_counter()

    logger.info("Evaluating %d images with %d TTA views on %s", len(dataset), len(views), args.device)
    with torch.no_grad():
        from tqdm import tqdm

        for view_batch, labels in tqdm(loader, desc="Evaluating TTA"):
            batch_size, num_views = view_batch.shape[:2]
            flat = view_batch.view(batch_size * num_views, *view_batch.shape[2:]).to(args.device)
            logits = model(flat).view(batch_size, num_views, -1).mean(dim=1)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            confs = probs[torch.arange(preds.size(0), device=preds.device), preds]
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_confidences.extend(confs.cpu().numpy().tolist())

    elapsed = time.perf_counter() - start
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), label_mapper.labels)
    if len(all_preds) != len(dataset.images):
        raise RuntimeError(f"Length mismatch: dataset={len(dataset.images)}, preds={len(all_preds)}")

    data_dir_path = Path(data_dir).resolve()
    predictions = []
    error_samples = []
    for (img_path, true_idx), pred_idx, conf in zip(dataset.images, all_preds, all_confidences):
        try:
            rel_path = str(img_path.resolve().relative_to(data_dir_path))
        except ValueError:
            rel_path = img_path.name
        row = {
            "filename": rel_path,
            "true_label": label_mapper.decode(true_idx),
            "predicted_label": label_mapper.decode(pred_idx),
            "confidence": round(conf, 6),
            "correct": int(true_idx == pred_idx),
        }
        predictions.append(row)
        if true_idx != pred_idx:
            error_samples.append({k: row[k] for k in ["filename", "true_label", "predicted_label", "confidence"]})

    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        label_mapper.labels,
        save_path=str(output_dir / "confusion_matrix.png"),
        title=f"TTA Confusion Matrix (Macro F1: {metrics['macro_f1']:.4f})",
    )

    with open(output_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "true_label", "predicted_label", "confidence", "correct"],
        )
        writer.writeheader()
        writer.writerows(predictions)

    with open(output_dir / "error_samples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "true_label", "predicted_label", "confidence"],
        )
        writer.writeheader()
        writer.writerows(error_samples)

    summary = {
        "num_images": len(dataset),
        "num_views": len(views),
        "device": args.device,
        "elapsed_sec": round(elapsed, 3),
        "ms_per_image": round(elapsed * 1000 / max(1, len(dataset)), 3),
        "crop_pct": args.crop_pct,
        "metrics": metrics,
    }
    with open(output_dir / "tta_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    tta_config = dict(config)
    tta_config["model"] = dict(tta_config.get("model", {}))
    tta_config["model"]["name"] = f"{tta_config['model'].get('name', 'model')}_tta"
    tta_config["tta"] = {
        "views": ["center_crop", "center_crop_hflip"],
        "crop_pct": args.crop_pct,
        "weights": args.weights,
    }
    tracker = ExperimentTracker(output_dir)
    result = tracker.build_result(
        config=tta_config,
        evaluation_metrics=metrics,
        notes=args.notes,
        experiment_id=args.experiment_id,
    )
    tracker.save(result)

    print("\n" + "=" * 60)
    print("TTA EVALUATION RESULTS")
    print("=" * 60)
    print(f"Macro F1:      {metrics['macro_f1']:.4f}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"CPU/device ms: {summary['ms_per_image']:.3f} ms/image on {args.device}")
    print("=" * 60)


if __name__ == "__main__":
    main()
