#!/usr/bin/env python3
"""Evaluate an ensemble by averaging member logits."""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import WeatherDataset
from data.label_mapping import detect_label_mapping, load_label_mapping
from data.transforms import build_transforms
from experiment_tracking.tracker import ExperimentTracker
from models.model_factory import create_model
from training.metrics import compute_metrics, plot_confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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


def _parse_member(spec: str) -> Dict:
    parts = spec.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(
            "--member must be WEIGHTS:CONFIG or WEIGHTS:CONFIG:WEIGHT, "
            f"got {spec!r}"
        )
    weight = float(parts[2]) if len(parts) == 3 else 1.0
    if weight <= 0:
        raise ValueError(f"Member weight must be positive, got {weight}")
    return {
        "weights": Path(parts[0]),
        "config": Path(parts[1]),
        "weight": weight,
    }


def _create_model(config: dict, num_classes: int, device: str) -> torch.nn.Module:
    model_cfg = config.get("model", {})
    model = create_model(
        name=model_cfg.get("name", "resnet18"),
        num_classes=num_classes,
        pretrained=False,
        dropout=float(model_cfg.get("dropout", 0.3)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
    )
    return model.to(device).eval()


def _ensemble_config(base_config: dict, members: List[Dict]) -> dict:
    config = dict(base_config)
    config["model"] = dict(config.get("model", {}))
    config["training"] = dict(config.get("training", {}))
    config["model"]["name"] = "logits_ensemble"
    config["training"]["loss"] = {"name": "logits_average"}
    config["ensemble"] = {
        "type": "logits_average",
        "members": [
            {
                "weights": str(member["weights"]),
                "config": str(member["config"]),
                "weight": member["weight"],
            }
            for member in members
        ],
    }
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a logits-average ensemble")
    parser.add_argument(
        "--member",
        action="append",
        required=True,
        help="Ensemble member as WEIGHTS:CONFIG or WEIGHTS:CONFIG:WEIGHT",
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--label_mapping", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--experiment_id", type=str, default=None)
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    members = [_parse_member(spec) for spec in args.member]
    for member in members:
        if not member["weights"].is_file():
            raise FileNotFoundError(f"Weights not found: {member['weights']}")
        if not member["config"].is_file():
            raise FileNotFoundError(f"Config not found: {member['config']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [_load_config(member["config"]) for member in members]
    base_config = configs[0]
    data_cfg = base_config.get("data", {})
    data_dir = args.data_dir or data_cfg.get("val_dir") or data_cfg.get("test_dir")
    if not data_dir:
        raise ValueError("No evaluation data_dir specified")

    if args.label_mapping and Path(args.label_mapping).exists():
        label_mapper = load_label_mapping(args.label_mapping)
    else:
        label_mapper = detect_label_mapping(data_dir)
    logger.info("Classes: %s", label_mapper.labels)

    models = []
    for member, config in zip(members, configs):
        model = _create_model(config, label_mapper.num_classes, args.device)
        model.load_state_dict(_load_state(member["weights"]))
        models.append(model)
        logger.info("Loaded member: %s", member["weights"])

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
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        multiprocessing_context=mp_ctx,
    )

    all_preds = []
    all_labels = []
    all_confidences = []
    weights = torch.tensor([member["weight"] for member in members], device=args.device)
    weights = weights / weights.sum()

    logger.info("Evaluating ensemble on %d images...", len(dataset))
    with torch.no_grad():
        from tqdm import tqdm

        for images, labels in tqdm(loader, desc="Evaluating ensemble"):
            images = images.to(args.device)
            logits_stack = torch.stack([model(images) for model in models])
            logits = (logits_stack * weights.view(-1, 1, 1)).sum(dim=0)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            confs = probs[torch.arange(preds.size(0), device=preds.device), preds]
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_confidences.extend(confs.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    metrics = compute_metrics(y_true, y_pred, label_mapper.labels)

    if len(all_preds) != len(dataset.images):
        raise RuntimeError(
            f"Length mismatch: dataset={len(dataset.images)}, preds={len(all_preds)}"
        )

    data_dir_path = Path(data_dir).resolve()
    error_samples = []
    for (img_path, true_idx), pred_idx, conf in zip(
        dataset.images, all_preds, all_confidences
    ):
        if true_idx != pred_idx:
            try:
                rel_path = str(img_path.resolve().relative_to(data_dir_path))
            except ValueError:
                rel_path = img_path.name
            error_samples.append(
                {
                    "filename": rel_path,
                    "true_label": label_mapper.decode(true_idx),
                    "predicted_label": label_mapper.decode(pred_idx),
                    "confidence": round(conf, 4),
                }
            )

    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(
        cm,
        label_mapper.labels,
        save_path=str(output_dir / "confusion_matrix.png"),
        title=f"Ensemble Confusion Matrix (Macro F1: {metrics['macro_f1']:.4f})",
    )

    with open(output_dir / "error_samples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "true_label", "predicted_label", "confidence"],
        )
        writer.writeheader()
        writer.writerows(error_samples)

    ensemble_metadata = _ensemble_config(base_config, members)
    with open(output_dir / "ensemble_members.json", "w", encoding="utf-8") as f:
        json.dump(ensemble_metadata["ensemble"], f, indent=2, ensure_ascii=False)

    tracker = ExperimentTracker(output_dir)
    result = tracker.build_result(
        config=ensemble_metadata,
        evaluation_metrics=metrics,
        notes=args.notes,
        experiment_id=args.experiment_id,
    )
    tracker.save(result)

    print("\n" + "=" * 60)
    print("ENSEMBLE EVALUATION RESULTS")
    print("=" * 60)
    print(f"Macro F1:  {metrics['macro_f1']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"{'Class':>10s}  {'Precision':>9s}  {'Recall':>9s}  {'F1':>9s}  {'Support':>8s}")
    print("-" * 56)
    for cls_name, cls_metrics in metrics["per_class"].items():
        print(
            f"{cls_name:>10s}  {cls_metrics['precision']:>9.4f}  "
            f"{cls_metrics['recall']:>9.4f}  {cls_metrics['f1']:>9.4f}  "
            f"{cls_metrics['support']:>8d}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
