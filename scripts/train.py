#!/usr/bin/env python3
"""
Training Entry Point

Trains a weather classification model with the specified configuration.
Macro F1 is the primary optimization metric.

Usage:
    python scripts/train.py --config configs/models/resnet18.yaml [--data_dir data/train]

    # Override specific parameters:
    python scripts/train.py --config configs/models/resnet18.yaml \
        --training.epochs 100 --training.batch_size 32
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import create_dataloaders
from data.label_mapping import detect_label_mapping, load_label_mapping
from data.transforms import build_transforms
from models.model_factory import create_model
from training.trainer import Trainer
from training.losses import create_loss_function
from training.callbacks import EarlyStopping, ModelCheckpoint, TrainingLogger
from training.metrics import compute_macro_f1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load default + model-specific config
    default_path = Path("configs/default.yaml")
    config = {}
    if default_path.exists():
        with open(default_path) as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Default config not found at {default_path}")

    with open(config_path) as f:
        model_config = yaml.safe_load(f)

    # Deep merge (model config overrides defaults)
    _deep_merge(config, model_config)
    return config


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def override_config(config: dict, overrides: list) -> dict:
    """Apply command-line overrides in dot notation.

    Example: ['--training.epochs', '100', '--training.batch_size', '32']
    """
    it = iter(overrides)
    for key, val in zip(it, it):
        # Remove leading -- if present
        key = key.lstrip("-")
        val = _parse_override_value(val)

        # Navigate to the nested key
        parts = key.split(".")
        d = config
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = val
        logger.info(f"Override: {key} = {val}")
    return config


def _parse_override_value(val: str):
    """Parse a string value into the appropriate type."""
    # Try int
    try:
        return int(val)
    except ValueError:
        pass
    # Try float
    try:
        return float(val)
    except ValueError:
        pass
    # Try bool
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    # Try null
    if val.lower() in ("null", "none"):
        return None
    # Try list
    if val.startswith("[") and val.endswith("]"):
        import json
        return json.loads(val)
    return val


def main():
    parser = argparse.ArgumentParser(
        description="Train a weather image classification model"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to model config YAML (e.g., configs/models/resnet18.yaml)"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Path to training data (overrides config)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for logs, checkpoints, and results"
    )
    parser.add_argument(
        "--label_mapping", type=str, default=None,
        help="Path to label mapping JSON (auto-detected from data_dir if not provided)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to train on: 'cuda' or 'cpu'"
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="Config overrides in dot notation: --key.subkey value"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.overrides:
        config = override_config(config, args.overrides)

    # Set seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    logger.info(f"Random seed: {seed}")

    # Determine data directory
    data_dir = args.data_dir or config["data"]["train_dir"]

    # Label mapping
    if args.label_mapping and Path(args.label_mapping).exists():
        label_mapper = load_label_mapping(args.label_mapping)
        logger.info(f"Loaded label mapping from {args.label_mapping}")
    else:
        label_mapper = detect_label_mapping(data_dir)
        label_mapping_path = Path(args.output_dir or "outputs") / "label_mapping.json"
        label_mapping_path.parent.mkdir(parents=True, exist_ok=True)
        label_mapper.save(label_mapping_path)
        logger.info(f"Auto-detected and saved label mapping to {label_mapping_path}")

    logger.info(f"Classes: {label_mapper.labels}")
    config["model"]["num_classes"] = label_mapper.num_classes

    # Build transforms
    data_cfg = config["data"]
    train_transform = build_transforms(
        mode="train",
        image_size=data_cfg["image_size"],
        mean=tuple(data_cfg["mean"]),
        std=tuple(data_cfg["std"]),
        augmentation=data_cfg.get("augmentation"),
    )
    val_transform = build_transforms(
        mode="val",
        image_size=data_cfg["image_size"],
        mean=tuple(data_cfg["mean"]),
        std=tuple(data_cfg["std"]),
    )

    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        data_dir=data_dir,
        val_dir=data_cfg.get("val_dir"),
        label_mapper=label_mapper,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config["training"]["batch_size"],
        num_workers=data_cfg.get("num_workers", 4),
        val_split=data_cfg.get("val_split", 0.2),
        seed=seed,
        pin_memory=data_cfg.get("pin_memory", True),
    )

    # Create model
    model_cfg = config["model"]
    model = create_model(
        name=model_cfg["name"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.3),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
    )
    logger.info(f"Model created: {model.backbone_name}")

    # Create loss
    loss_cfg = config["training"].get("loss", {})
    loss_fn = create_loss_function(
        name=loss_cfg.get("name", "cross_entropy"),
        num_classes=label_mapper.num_classes,
        class_weights=loss_cfg.get("class_weights", None),
        label_smoothing=loss_cfg.get("label_smoothing", 0.0),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
    )

    # Create optimizer
    opt_cfg = config["training"]["optimizer"]
    optimizer = _create_optimizer(opt_cfg, model)

    # Create scheduler
    sched_cfg = config["training"].get("scheduler", {})
    scheduler = _create_scheduler(sched_cfg, optimizer, len(train_loader))

    # Determine output directory
    output_dir = args.output_dir or config["logging"].get("log_dir", "outputs")
    exp_name = config["logging"].get("experiment_name") or f"{model_cfg['name']}_{int(time.time())}"
    output_dir = Path(output_dir) / exp_name

    # Callbacks
    early_stopping = EarlyStopping(
        patience=config["training"].get("early_stopping", {}).get("patience", 10),
        min_delta=config["training"].get("early_stopping", {}).get("min_delta", 0.001),
        mode="max",
    )
    checkpoint = ModelCheckpoint(
        save_dir=str(output_dir / "checkpoints"),
        save_top_k=config["training"].get("checkpoint", {}).get("save_top_k", 3),
        monitor="val_macro_f1",
        mode="max",
    )
    logger_cb = TrainingLogger(log_file=str(output_dir / "training_log.jsonl"))

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        label_mapper=label_mapper,
        device=args.device,
        use_amp=(args.device == "cuda"),
        config=config,
    )

    # Train
    logger.info(f"Starting training for {config['training']['epochs']} epochs")
    logger.info(f"Output directory: {output_dir}")
    start = time.time()

    metrics_tracker = trainer.fit(
        epochs=config["training"]["epochs"],
        early_stopping=early_stopping,
        checkpoint=checkpoint,
        logger_callback=logger_cb,
        output_dir=str(output_dir),
    )

    elapsed = time.time() - start
    logger.info(f"Training completed in {elapsed / 60:.1f} minutes")

    # Save final results
    best = metrics_tracker.get_best_epoch()
    logger.info(f"Best epoch: {best.get('epoch', 'N/A') + 1}, "
                 f"Macro F1: {best.get('val_macro_f1', 0):.4f}")

    # Save training history
    import pandas as pd
    df = metrics_tracker.to_dataframe()
    df.to_csv(output_dir / "training_history.csv", index=False)
    logger.info(f"Training history saved to {output_dir}/training_history.csv")

    # Copy best model to weights directory
    import shutil
    best_model_path = output_dir / "best_model.pth"
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    if best_model_path.exists():
        dest = weights_dir / f"{model_cfg['name']}_best.pth"
        shutil.copy(best_model_path, dest)
        logger.info(f"Best model copied to {dest}")


def _create_optimizer(opt_cfg: dict, model: torch.nn.Module):
    """Create optimizer from config."""
    name = opt_cfg.get("name", "adamw").lower()
    lr = opt_cfg.get("lr", 1e-4)
    weight_decay = opt_cfg.get("weight_decay", 1e-4)

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def _create_scheduler(
    sched_cfg: dict,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
):
    """Create learning rate scheduler from config."""
    name = sched_cfg.get("name", "cosine").lower()
    epochs = sched_cfg.get("_epochs", 50)  # Set externally

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - sched_cfg.get("warmup_epochs", 3),
            eta_min=sched_cfg.get("min_lr", 1e-6),
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 15),
            gamma=sched_cfg.get("step_gamma", 0.1),
        )
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=sched_cfg.get("min_lr", 1e-6),
        )
    elif name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=sched_cfg.get("lr", 1e-4),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
        )
    else:
        logger.warning(f"Unknown scheduler: {name}, using CosineAnnealingLR")
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)


if __name__ == "__main__":
    main()
