#!/usr/bin/env python3
"""Evaluate logits post-processing experiments.

Supported modes:
- temperature: fit per-member temperature values, then average logits.
- class_bias: greedily search per-class logit bias on a weighted ensemble.
- hflip_tta: average original and horizontal-flip logits for one model.
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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


class HFlipDataset(Dataset):
    """Return original and horizontally flipped views for each image."""

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
        return torch.stack([view(image) for view in self.views], dim=0), label_idx


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
    return {"weights": Path(parts[0]), "config": Path(parts[1]), "weight": weight}


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


def _build_val_dataset(data_dir: str | Path, label_mapper, config: dict) -> WeatherDataset:
    data_cfg = config.get("data", {})
    transform = build_transforms(
        mode="val",
        image_size=int(data_cfg.get("image_size", 224)),
        mean=tuple(data_cfg.get("mean", [0.485, 0.456, 0.406])),
        std=tuple(data_cfg.get("std", [0.229, 0.224, 0.225])),
    )
    return WeatherDataset(data_dir=data_dir, transform=transform, label_mapper=label_mapper)


def _collect_model_logits(
    member: Dict,
    config: dict,
    data_dir: str | Path,
    label_mapper,
    device: str,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, WeatherDataset]:
    model = _create_model(config, label_mapper.num_classes, device)
    model.load_state_dict(_load_state(member["weights"]))
    dataset = _build_val_dataset(data_dir, label_mapper, config)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    logits_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    logger.info("Collecting logits: %s", member["weights"])
    with torch.no_grad():
        from tqdm import tqdm

        for images, labels in tqdm(loader, desc=f"Logits {member['weights'].parent.name}"):
            logits = model(images.to(device))
            logits_out.append(logits.cpu().numpy())
            labels_out.append(labels.numpy())
    return np.concatenate(logits_out), np.concatenate(labels_out), dataset


def _build_hflip_views(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> list[transforms.Compose]:
    base = [transforms.Resize((image_size, image_size))]
    tail = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return [
        transforms.Compose(base + tail),
        transforms.Compose(base + [transforms.RandomHorizontalFlip(p=1.0)] + tail),
    ]


def _collect_hflip_logits(
    member: Dict,
    config: dict,
    data_dir: str | Path,
    label_mapper,
    device: str,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, HFlipDataset]:
    model = _create_model(config, label_mapper.num_classes, device)
    model.load_state_dict(_load_state(member["weights"]))

    data_cfg = config.get("data", {})
    views = _build_hflip_views(
        image_size=int(data_cfg.get("image_size", 224)),
        mean=tuple(data_cfg.get("mean", [0.485, 0.456, 0.406])),
        std=tuple(data_cfg.get("std", [0.229, 0.224, 0.225])),
    )
    base = WeatherDataset(data_dir=data_dir, transform=None, label_mapper=label_mapper)
    dataset = HFlipDataset(base, views)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    logits_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    logger.info("Collecting hflip TTA logits: %s", member["weights"])
    with torch.no_grad():
        from tqdm import tqdm

        for view_batch, labels in tqdm(loader, desc="HFlip TTA"):
            batch_size_actual, num_views = view_batch.shape[:2]
            flat = view_batch.view(batch_size_actual * num_views, *view_batch.shape[2:]).to(device)
            logits = model(flat).view(batch_size_actual, num_views, -1).mean(dim=1)
            logits_out.append(logits.cpu().numpy())
            labels_out.append(labels.numpy())
    return np.concatenate(logits_out), np.concatenate(labels_out), dataset


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, labels: np.ndarray) -> float:
    probs = _softmax(logits)
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def _fit_temperature_grid(
    logits: np.ndarray,
    labels: np.ndarray,
    candidates: np.ndarray,
) -> float:
    scores = [(float(temp), _nll(logits / float(temp), labels)) for temp in candidates]
    return min(scores, key=lambda item: item[1])[0]


def _weighted_logits(
    member_logits: np.ndarray,
    weights: np.ndarray,
    temperatures: np.ndarray | None = None,
) -> np.ndarray:
    weights = weights.astype(np.float64)
    weights = weights / weights.sum()
    logits = member_logits.astype(np.float64)
    if temperatures is not None:
        logits = logits / temperatures.reshape(-1, 1, 1)
    return (logits * weights.reshape(-1, 1, 1)).sum(axis=0)


def _greedy_search_class_bias(
    logits: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    candidates: np.ndarray,
    rounds: int,
) -> tuple[np.ndarray, float]:
    bias = np.zeros(num_classes, dtype=np.float64)
    best_score = float(f1_score(labels, np.argmax(logits + bias, axis=1), average="macro", zero_division=0))

    for _ in range(rounds):
        improved = False
        for cls_idx in range(num_classes):
            cls_best_value = bias[cls_idx]
            cls_best_score = best_score
            for value in candidates:
                trial = bias.copy()
                trial[cls_idx] = float(value)
                preds = np.argmax(logits + trial, axis=1)
                score = float(f1_score(labels, preds, average="macro", zero_division=0))
                if score > cls_best_score:
                    cls_best_score = score
                    cls_best_value = float(value)
            if cls_best_score > best_score:
                bias[cls_idx] = cls_best_value
                best_score = cls_best_score
                improved = True
        if not improved:
            break
    return bias, round(best_score, 4)


def _validate_same_labels(labels_per_member: list[np.ndarray]) -> np.ndarray:
    first = labels_per_member[0]
    for index, labels in enumerate(labels_per_member[1:], start=2):
        if not np.array_equal(first, labels):
            raise RuntimeError(f"Member {index} label order differs from member 1")
    return first


def _write_outputs(
    output_dir: Path,
    dataset,
    label_mapper,
    labels: np.ndarray,
    logits: np.ndarray,
    config: dict,
    experiment_id: str | None,
    notes: str,
    summary: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    probs = _softmax(logits)
    preds = np.argmax(probs, axis=1)
    confidences = probs[np.arange(len(preds)), preds]
    metrics = compute_metrics(labels, preds, label_mapper.labels)

    data_dir_path = Path(dataset.base.data_dir if hasattr(dataset, "base") else dataset.data_dir).resolve()
    predictions = []
    error_samples = []
    for (img_path, true_idx), pred_idx, conf in zip(dataset.images, preds.tolist(), confidences.tolist()):
        try:
            rel_path = str(img_path.resolve().relative_to(data_dir_path))
        except ValueError:
            rel_path = img_path.name
        row = {
            "filename": rel_path,
            "true_label": label_mapper.decode(int(true_idx)),
            "predicted_label": label_mapper.decode(int(pred_idx)),
            "confidence": round(float(conf), 6),
            "correct": int(true_idx == pred_idx),
        }
        predictions.append(row)
        if true_idx != pred_idx:
            error_samples.append({k: row[k] for k in ["filename", "true_label", "predicted_label", "confidence"]})

    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        label_mapper.labels,
        save_path=str(output_dir / "confusion_matrix.png"),
        title=f"Postprocess Confusion Matrix (Macro F1: {metrics['macro_f1']:.4f})",
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

    summary = dict(summary)
    summary["metrics"] = metrics
    with open(output_dir / "postprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    tracker = ExperimentTracker(output_dir)
    result = tracker.build_result(
        config=config,
        evaluation_metrics=metrics,
        notes=notes,
        experiment_id=experiment_id,
    )
    tracker.save(result)
    return metrics


def _postprocess_config(base_config: dict, mode: str, members: list[Dict], summary: dict) -> dict:
    config = dict(base_config)
    config["model"] = dict(config.get("model", {}))
    config["training"] = dict(config.get("training", {}))
    config["model"]["name"] = f"{config['model'].get('name', 'model')}_{mode}"
    config["training"]["loss"] = {"name": "postprocess"}
    config["postprocess"] = {
        "mode": mode,
        "summary": summary,
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
    parser = argparse.ArgumentParser(description="Evaluate logits post-processing")
    parser.add_argument("--mode", choices=["temperature", "class_bias", "hflip_tta"], required=True)
    parser.add_argument("--member", action="append", required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--label_mapping", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--experiment_id", type=str, default=None)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--temperature_min", type=float, default=0.5)
    parser.add_argument("--temperature_max", type=float, default=3.0)
    parser.add_argument("--temperature_steps", type=int, default=51)
    parser.add_argument("--bias_min", type=float, default=-1.0)
    parser.add_argument("--bias_max", type=float, default=1.0)
    parser.add_argument("--bias_step", type=float, default=0.05)
    parser.add_argument("--bias_rounds", type=int, default=3)
    args = parser.parse_args()

    members = [_parse_member(spec) for spec in args.member]
    for member in members:
        if not member["weights"].is_file():
            raise FileNotFoundError(f"Weights not found: {member['weights']}")
        if not member["config"].is_file():
            raise FileNotFoundError(f"Config not found: {member['config']}")
    if args.mode == "hflip_tta" and len(members) != 1:
        raise ValueError("hflip_tta expects exactly one --member")

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

    start = time.perf_counter()
    if args.mode == "hflip_tta":
        logits, labels, dataset = _collect_hflip_logits(
            members[0],
            configs[0],
            data_dir,
            label_mapper,
            args.device,
            args.batch_size,
            args.num_workers,
        )
        summary = {
            "mode": args.mode,
            "views": ["resize", "resize_hflip"],
            "elapsed_sec": round(time.perf_counter() - start, 3),
        }
    else:
        logits_per_member = []
        labels_per_member = []
        datasets = []
        for member, config in zip(members, configs):
            logits, labels, dataset = _collect_model_logits(
                member,
                config,
                data_dir,
                label_mapper,
                args.device,
                args.batch_size,
                args.num_workers,
            )
            logits_per_member.append(logits)
            labels_per_member.append(labels)
            datasets.append(dataset)
            if datasets[-1].images != datasets[0].images:
                raise RuntimeError("Member image order differs from member 1")

        labels = _validate_same_labels(labels_per_member)
        member_logits = np.stack(logits_per_member, axis=0)
        weights = np.array([member["weight"] for member in members], dtype=np.float64)
        weights = weights / weights.sum()

        if args.mode == "temperature":
            candidates = np.linspace(args.temperature_min, args.temperature_max, args.temperature_steps)
            temperatures = np.array(
                [_fit_temperature_grid(logits, labels, candidates) for logits in member_logits],
                dtype=np.float64,
            )
            logits = _weighted_logits(member_logits, weights=weights, temperatures=temperatures)
            summary = {
                "mode": args.mode,
                "weights": weights.tolist(),
                "temperatures": temperatures.tolist(),
                "temperature_candidates": [float(candidates[0]), float(candidates[-1]), int(len(candidates))],
                "elapsed_sec": round(time.perf_counter() - start, 3),
            }
        else:
            logits = _weighted_logits(member_logits, weights=weights)
            candidates = np.arange(args.bias_min, args.bias_max + args.bias_step / 2, args.bias_step)
            bias, best_score = _greedy_search_class_bias(
                logits,
                labels,
                num_classes=label_mapper.num_classes,
                candidates=candidates,
                rounds=args.bias_rounds,
            )
            logits = logits + bias
            summary = {
                "mode": args.mode,
                "weights": weights.tolist(),
                "class_bias": bias.tolist(),
                "search_best_macro_f1": best_score,
                "bias_candidates": [float(candidates[0]), float(candidates[-1]), int(len(candidates))],
                "elapsed_sec": round(time.perf_counter() - start, 3),
            }
        dataset = datasets[0]

    config = _postprocess_config(base_config, args.mode, members, summary)
    metrics = _write_outputs(
        Path(args.output_dir),
        dataset,
        label_mapper,
        labels,
        logits,
        config,
        args.experiment_id,
        args.notes,
        summary,
    )
    print("\n" + "=" * 60)
    print("LOGITS POSTPROCESS EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mode:      {args.mode}")
    print(f"Macro F1:  {metrics['macro_f1']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
