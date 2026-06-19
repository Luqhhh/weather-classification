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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SUBMIT_INFERENCE_TEMPLATE = '''#!/usr/bin/env python3
"""
Weather Image Classification — Inference Script

Reads images from the specified directory, produces a predictions CSV.
Designed for CPU-only execution, ≤ 70 minutes runtime.

Usage:
    python inference.py --input_dir /path/to/images --output predictions.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# ============================================================
# Configuration — update these for your model
# ============================================================
MODEL_NAME = "{model_name}"
NUM_CLASSES = {num_classes}
IMAGE_SIZE = {image_size}
MEAN = {mean}
STD = {std}
WEIGHTS_FILE = "{weights_file}"
LABEL_MAPPING = {label_mapping}  # idx → class name
# ============================================================


def build_model(num_classes: int) -> nn.Module:
    """Reconstruct the model architecture."""
    # Strip classifier and add custom head
    backbone = models.__dict__[MODEL_NAME](pretrained=False)
    if hasattr(backbone, "fc"):
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif hasattr(backbone, "classifier"):
        if isinstance(backbone.classifier, nn.Sequential):
            # Find the last Linear layer
            for layer in reversed(backbone.classifier):
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    break
        backbone.classifier = nn.Identity()
    else:
        raise ValueError(f"Cannot determine in_features for {{MODEL_NAME}}")

    model = nn.Sequential(
        backbone,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(0.0),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_transform() -> transforms.Compose:
    """Build the inference transform."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def predict_images(
    input_dir: str,
    output_csv: str,
    batch_size: int = 32,
) -> None:
    """Predict classes for all images in a directory."""

    device = torch.device("cpu")

    # Load model
    model = build_model(NUM_CLASSES)
    state = torch.load(WEIGHTS_FILE, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    transform = get_transform()

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
            img_tensor = transform(img)
            batch_tensors.append(img_tensor)

        batch = torch.stack(batch_tensors).to(device)

        # Predict
        with torch.no_grad():
            logits = model(batch)
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

    # Prepare output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Preparing submission in {output_dir}/")

    # 1. Generate inference script
    logger.info("Generating inference script...")
    inference_code = SUBMIT_INFERENCE_TEMPLATE.format(
        model_name=args.model,
        num_classes=label_mapper.num_classes,
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
