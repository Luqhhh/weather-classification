#!/usr/bin/env python3
"""
Prepare Submission Package

Assembles the final submission package:
1. Copies inference code → submit/
2. Copies model weights → submit/
3. Generates submit/requirements.txt
4. Runs pre-submission checks
5. Creates a zip archive

Usage:
    python scripts/prepare_submission.py \
        --weights weights/resnet18_best.pth \
        --model resnet18 \
        --label_mapping reports/label_mapping.json
"""

import argparse
import logging
import shutil
import sys
import zipfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.label_mapping import load_label_mapping
from inference.submit_checker import SubmitChecker
from models.model_factory import MODEL_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SUBMIT_INFERENCE_TEMPLATE = '''#!/usr/bin/env python3
"""
Weather Image Classification — Inference Script

Supports two usage modes:

1. Platform import (per-image scoring):
       from inference import predict
       label = predict(X)   # X: np.ndarray from cv2.imread, shape (H,W,3), BGR

2. Standalone batch inference:
       python inference.py --input_dir /path/to/images --output predictions.csv

Designed for CPU-only execution, ≤ 70 minutes runtime.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# ============================================================
# Configuration — update these for your model
# ============================================================
MODEL_NAME = "{model_name}"
NUM_CLASSES = {num_classes}
IN_FEATURES = {in_features}
IMAGE_SIZE = {image_size}
MEAN = {mean}
STD = {std}
WEIGHTS_FILE = "{weights_file}"
LABEL_MAPPING = {label_mapping}  # idx → class name
# ============================================================


def build_model(num_classes: int, in_features: int) -> nn.Module:
    """Reconstruct the model architecture.

    Matches the WeatherClassifier structure used during training:
        backbone → pool → dropout → fc → logits
    """
    backbone = models.__dict__[MODEL_NAME](pretrained=False)

    # Strip the original classification head
    if hasattr(backbone, "classifier"):
        backbone.classifier = nn.Identity()
    if hasattr(backbone, "fc"):
        backbone.fc = nn.Identity()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(p=0.0)
            self.fc = nn.Linear(in_features, num_classes)

        def forward(self, x):
            features = self.backbone(x)
            if isinstance(features, tuple):
                features = features[-1]
            if features.dim() == 4:
                features = self.pool(features)
                features = features.view(features.size(0), -1)
            features = self.dropout(features)
            return self.fc(features)

    return Model()


def get_transform() -> transforms.Compose:
    """Build the inference transform."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


# ============================================================
# Module-level model loading — runs once on import
# ============================================================
_device = torch.device("cpu")
_transform = get_transform()

_model = build_model(NUM_CLASSES, IN_FEATURES)
_state = torch.load(WEIGHTS_FILE, map_location="cpu", weights_only=True)

# Handle wrapped checkpoint format
if "model_state_dict" in _state:
    _state = _state["model_state_dict"]

_model.load_state_dict(_state)
_model = _model.to(_device)
_model.eval()


def predict(X: np.ndarray) -> str:
    """Platform inference interface — called once per image by the scoring system.

    Args:
        X: np.ndarray from ``cv2.imread`` — BGR, uint8, shape (H, W, 3).

    Returns:
        Predicted weather class label: 'cloudy', 'rainy', 'snowy', or 'sunny'.
    """
    # cv2.imread returns BGR — convert to RGB for PIL / torchvision
    X_rgb = X[:, :, ::-1]
    img = Image.fromarray(X_rgb)
    img_tensor = _transform(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(img_tensor)
        pred_idx = torch.argmax(logits, dim=1).item()

    return LABEL_MAPPING[pred_idx]


# ============================================================
# Batch inference (standalone CLI usage)
# ============================================================

def predict_images(
    input_dir: str,
    output_csv: str,
    batch_size: int = 32,
) -> None:
    """Predict classes for all images in a directory."""

    # Find images
    extensions = {{".jpg", ".jpeg", ".png", ".bmp", ".webp"}}
    image_paths = sorted([
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in extensions and p.is_file()
    ])

    print(f"Found {{len(image_paths)}} images in {{input_dir}}")

    # Batch predict
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]

        # Load and preprocess batch
        batch_tensors = []
        for path in batch_paths:
            img = Image.open(path).convert("RGB")
            img_tensor = _transform(img)
            batch_tensors.append(img_tensor)

        batch = torch.stack(batch_tensors).to(_device)

        # Predict
        with torch.no_grad():
            logits = _model(batch)
            preds = torch.argmax(logits, dim=1)

        for path, pred_idx in zip(batch_paths, preds):
            results.append({{
                "filename": path.name,
                "prediction": LABEL_MAPPING[pred_idx.item()],
            }})

        if (i // batch_size + 1) % 50 == 0:
            print(f"  Progress: {{i + len(batch_paths)}}/{{len(image_paths)}}")

    # Write CSV output
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "prediction"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Predictions saved to {{output_csv}} ({{len(results)}} images)")


def main():
    parser = argparse.ArgumentParser(
        description="Weather Image Classification — Inference"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing images to classify"
    )
    parser.add_argument(
        "--output", type=str, default="predictions.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Inference batch size"
    )
    args = parser.parse_args()

    predict_images(
        input_dir=args.input_dir,
        output_csv=args.output,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
'''


SUBMIT_REQUIREMENTS = """# Weather Classification Inference Dependencies
# Competition runtime caps PyTorch at 2.1.7.
# Team standard: torch 2.1.2 — locked to avoid version drift in PRs.
torch==2.1.2
torchvision==0.16.2
numpy>=1.24.0,<2.0.0
Pillow>=9.5.0
"""


def retrain_on_full_data(
    weights_path: Path,
    config: dict,
    data_dirs: list,
    label_mapper,
    output_path: Path,
    finetune_epochs: int = 5,
    finetune_lr: float = 1e-5,
    device: str = "cuda",
) -> Path:
    """Retrain the best model on the full dataset (train+val+holdout).

    Warm-starts from the best checkpoint, uses a low learning rate for a
    small number of epochs to avoid overfitting while letting the model
    see all available labelled data.

    Returns:
        Path to the retrained model weights.
    """
    from data.dataset import WeatherDataset
    from data.transforms import build_transforms
    from models.model_factory import create_model
    from training.trainer import Trainer
    from training.losses import create_loss_function
    from training.callbacks import EarlyStopping, ModelCheckpoint, TrainingLogger
    from torch.utils.data import ConcatDataset, DataLoader

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})

    image_size = data_cfg.get("image_size", 224)
    mean = tuple(data_cfg.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(data_cfg.get("std", [0.229, 0.224, 0.225]))
    batch_size = training_cfg.get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 1)

    logger.info("=" * 60)
    logger.info("Retraining on full dataset (train + val + holdout)")
    logger.info(f"  Epochs: {finetune_epochs}, LR: {finetune_lr}")
    logger.info(f"  Device: {device}")

    # Build transforms
    train_transform = build_transforms(
        mode="train", image_size=image_size, mean=mean, std=std,
        augmentation=data_cfg.get("augmentation"),
    )

    # Load all data directories into a single dataset
    full_datasets = []
    total_images = 0
    for d in data_dirs:
        if Path(d).is_dir():
            ds = WeatherDataset(data_dir=d, transform=None, label_mapper=label_mapper)
            full_datasets.append(ds)
            total_images += len(ds)
            logger.info(f"  Loaded {len(ds)} images from {d}")

    if not full_datasets:
        logger.error("No data directories found for retraining!")
        sys.exit(1)

    # For simplicity, use the first dataset as base and concatenate rest
    # We use a Subset wrapper to avoid double-transforming
    class RetrainDataset(torch.utils.data.Dataset):
        """Flat dataset combining multiple WeatherDatasets."""
        def __init__(self, datasets):
            self.samples = []
            for ds in datasets:
                for path, label in ds.images:
                    self.samples.append((path, label))
            self.transform = None

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            from PIL import Image, ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            img = Image.open(path).convert("RGB")
            return img, label

    full_ds = RetrainDataset(full_datasets)
    full_ds.transform = train_transform

    # Override transform application
    class TransformDataset(torch.utils.data.Dataset):
        def __init__(self, base_ds, transform):
            self.base = base_ds
            self.transform = transform

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, label = self.base[idx]
            if self.transform and isinstance(img, Image.Image):
                img = self.transform(img)
            return img, label

    train_ds = TransformDataset(full_ds, train_transform)
    logger.info(f"  Total training images: {len(train_ds)}")

    mp_ctx = data_cfg.get("multiprocessing_context", "spawn") or None
    mp_ctx = mp_ctx if num_workers > 0 else None
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
        drop_last=True,
        persistent_workers=num_workers > 0,
        multiprocessing_context=mp_ctx,
    )
    # Dummy val loader (not used for early stopping during finetuning)
    val_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
        multiprocessing_context=mp_ctx,
    )

    # Create model
    model = create_model(
        name=model_cfg.get("name", "resnet18"),
        num_classes=label_mapper.num_classes,
        pretrained=False,
        dropout=model_cfg.get("dropout", 0.3),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
    )

    # Load best weights as initialization
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    logger.info(f"  Warm-started from {weights_path}")

    # Loss (use same config as original training)
    loss_cfg = training_cfg.get("loss", {})
    loss_fn = create_loss_function(
        name=loss_cfg.get("name", "cross_entropy"),
        num_classes=label_mapper.num_classes,
        class_weights=loss_cfg.get("class_weights", None),
        label_smoothing=loss_cfg.get("label_smoothing", 0.0),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
    )

    # Optimizer with very low LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=finetune_epochs, eta_min=1e-7
    )

    # Trainer
    aug_cfg = data_cfg.get("augmentation", {})
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        label_mapper=label_mapper,
        device=device,
        use_amp=(device == "cuda"),
        config=config,
        mixup_alpha=aug_cfg.get("mixup_alpha") or 0.0,
        cutmix_alpha=aug_cfg.get("cutmix_alpha") or 0.0,
    )

    # Minimal callbacks — no early stopping during finetuning
    checkpoint = ModelCheckpoint(
        save_dir=str(output_path.parent / "retrain_checkpoints"),
        save_top_k=1,
        monitor="train_loss",
        mode="min",
    )

    logger.info("  Starting retraining...")
    trainer.fit(
        epochs=finetune_epochs,
        early_stopping=None,
        checkpoint=checkpoint,
        logger_callback=None,
        output_dir=str(output_path.parent),
    )

    # Save the retrained model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, output_path)
    logger.info(f"  Retrained model saved to {output_path}")
    logger.info("=" * 60)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare submission package for weather classification competition"
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to the trained model weights (.pth)"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model architecture name (e.g., resnet18)"
    )
    parser.add_argument(
        "--label_mapping", type=str, required=True,
        help="Path to label mapping JSON"
    )
    parser.add_argument(
        "--image_size", type=int, default=224,
        help="Input image size"
    )
    parser.add_argument(
        "--test_dir", type=str, default=None,
        help="Test images directory for smoke test"
    )
    parser.add_argument(
        "--output_dir", type=str, default="submit",
        help="Output directory for submission package"
    )
    parser.add_argument(
        "--skip_checks", action="store_true",
        help="Skip pre-submission validation"
    )
    parser.add_argument(
        "--retrain_on_full", action="store_true",
        help="Retrain on combined train+val+holdout before packaging"
    )
    parser.add_argument(
        "--retrain_epochs", type=int, default=5,
        help="Number of epochs for retrain-on-full (default: 5)"
    )
    parser.add_argument(
        "--retrain_lr", type=float, default=1e-5,
        help="Learning rate for retrain-on-full (default: 1e-5)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to model config YAML (required for --retrain_on_full)"
    )
    parser.add_argument(
        "--data_dir", type=str, nargs="+", default=None,
        help="Data directories for retrain-on-full (default: data/train data/val data/holdout)"
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    label_path = Path(args.label_mapping)
    output_dir = Path(args.output_dir)

    if not weights_path.exists():
        logger.error(f"Weights file not found: {weights_path}")
        sys.exit(1)
    if not label_path.exists():
        logger.error(f"Label mapping not found: {label_path}")
        sys.exit(1)

    # Load label mapping
    label_mapper = load_label_mapping(label_path)
    label_mapping_dict = label_mapper.idx_to_label
    # Convert int keys back (JSON stores them as strings)
    label_mapping_dict = {int(k): v for k, v in label_mapping_dict.items()}

    # --- Retrain on full data (optional) ---
    if args.retrain_on_full:
        import yaml as _yaml_lib

        if not args.config:
            logger.error("--config is required when using --retrain_on_full")
            sys.exit(1)

        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)
        with open(config_path, encoding="utf-8") as f:
            config = _yaml_lib.safe_load(f)

        data_dirs = args.data_dir or ["data/train", "data/val", "data/holdout"]
        # Only include directories that exist
        data_dirs = [d for d in data_dirs if Path(d).is_dir()]
        if not data_dirs:
            logger.error("No data directories found for retrain-on-full")
            sys.exit(1)

        retrain_weights = output_dir / f"retrained_{weights_path.name}"
        retrain_on_full_data(
            weights_path=weights_path,
            config=config,
            data_dirs=data_dirs,
            label_mapper=label_mapper,
            output_path=retrain_weights,
            finetune_epochs=args.retrain_epochs,
            finetune_lr=args.retrain_lr,
        )
        # Use the retrained weights for the submission package
        weights_path = retrain_weights
        logger.info("Switching to retrained weights for submission package")

    # Prepare output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Preparing submission in {output_dir}/")

    # Resolve in_features from model registry
    if args.model not in MODEL_REGISTRY:
        logger.error(
            f"Unknown model '{args.model}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
        sys.exit(1)
    in_features = MODEL_REGISTRY[args.model]["in_features"]

    # 1. Generate inference script
    logger.info("Generating inference script...")
    inference_code = SUBMIT_INFERENCE_TEMPLATE.format(
        model_name=args.model,
        num_classes=label_mapper.num_classes,
        in_features=in_features,
        image_size=args.image_size,
        mean="[0.485, 0.456, 0.406]",
        std="[0.229, 0.224, 0.225]",
        weights_file=weights_path.name,
        label_mapping=repr(label_mapping_dict),
    )

    inference_script = output_dir / "inference.py"
    with open(inference_script, "w", encoding="utf-8") as f:
        f.write(inference_code)
    logger.info(f"  Inference script: {inference_script}")

    # 2. Copy weights
    logger.info("Copying model weights...")
    weights_dest = output_dir / weights_path.name
    shutil.copy2(weights_path, weights_dest)
    weight_size_mb = weights_dest.stat().st_size / (1024 * 1024)
    logger.info(f"  Weights: {weights_dest} ({weight_size_mb:.1f} MB)")

    # 3. Write requirements.txt
    logger.info("Writing requirements.txt...")
    req_path = output_dir / "requirements.txt"
    with open(req_path, "w", encoding="utf-8") as f:
        f.write(SUBMIT_REQUIREMENTS)
    logger.info(f"  Requirements: {req_path}")

    # 4. Write README
    readme = output_dir / "README.md"
    with open(readme, "w", encoding="utf-8") as f:
        f.write(f"""# Weather Image Classification — Submission

## Model
- Architecture: {args.model}
- Input size: {args.image_size}×{args.image_size}
- Classes: {', '.join(label_mapper.labels)}

## Usage
```bash
pip install -r requirements.txt
python inference.py --input_dir /path/to/images --output predictions.csv
```

## File Summary
- `inference.py` — Main inference script
- `{weights_path.name}` — Model weights
- `requirements.txt` — Python dependencies
""")

    # 5. Run pre-submission checks
    if not args.skip_checks:
        logger.info("\nRunning pre-submission checks...")
        test_dir = args.test_dir or "data/test"
        checker = SubmitChecker(
            inference_script=str(inference_script),
            weights_path=str(weights_dest),
            test_images_dir=test_dir,
            label_mapping_path=str(label_path),
            submit_dir=str(output_dir),
        )
        results = checker.run_all_checks()

        if not results["all_passed"]:
            logger.error("Some checks failed — fix issues before submitting!")
            logger.error("Or use --skip_checks to bypass (not recommended)")
            sys.exit(1)
    else:
        logger.warning("⚠️  Pre-submission checks SKIPPED — review manually before submitting!")

    # 6. Create zip archive
    logger.info("\nCreating submission archive...")
    zip_path = Path(f"submit_{args.model}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in output_dir.iterdir():
            if file_path.is_file():
                zf.write(file_path, file_path.name)
                logger.info(f"  Added: {file_path.name}")

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    logger.info(f"Submission archive: {zip_path} ({zip_size_mb:.1f} MB)")
    logger.info("\n✅ Submission package prepared successfully!")
    logger.info(f"   Archive: {zip_path}")
    logger.info(f"   Contents: inference.py, {weights_path.name}, requirements.txt, README.md")


if __name__ == "__main__":
    main()
