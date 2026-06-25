#!/usr/bin/env python3
"""
Batch Inference Script

Runs prediction on a directory of images and outputs a CSV file.
Used for testing the model on the competition test set.

Usage:
    python scripts/inference.py \
        --weights weights/best.pth \
        --model resnet18 \
        --input_dir data/test \
        --output predictions.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.label_mapping import load_label_mapping, detect_label_mapping
from inference.predictor import WeatherPredictor
from models.model_factory import create_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run batch inference on weather images"
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to model weights (.pth)"
    )
    parser.add_argument(
        "--model", type=str, default="resnet18",
        help="Model architecture name (e.g., resnet18, efficientnet_b0)"
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
        "--label_mapping", type=str, default=None,
        help="Path to label mapping JSON"
    )
    parser.add_argument(
        "--image_size", type=int, default=224,
        help="Input image size"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Inference batch size"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device: always 'cpu' for final submission"
    )
    parser.add_argument(
        "--estimate_time", type=int, default=None,
        help="Estimate total inference time for N images"
    )
    args = parser.parse_args()

    # Label mapping
    if args.label_mapping and Path(args.label_mapping).exists():
        label_mapper = load_label_mapping(args.label_mapping)
    else:
        label_mapper = detect_label_mapping(args.input_dir)
    logger.info(f"Classes: {label_mapper.labels}")

    # Create model
    model = create_model(
        name=args.model,
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
    logger.info(f"Loaded weights from {args.weights}")

    # Predictor
    predictor = WeatherPredictor(
        model=model,
        label_mapper=label_mapper,
        image_size=args.image_size,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Time estimation
    if args.estimate_time:
        estimate = predictor.estimate_total_time(args.estimate_time)
        print("\nTime Estimate:")
        print(f"  Images:          {estimate['num_images']}")
        print(f"  Avg image time:  {estimate['avg_image_time_ms']} ms")
        print(f"  Total estimate:  {estimate['estimated_total_time_min']} min")
        print(f"  Within 70min:    {'✅ YES' if estimate['within_70min_limit'] else '❌ NO'}")

    # Run inference
    results = predictor.predict_batch(
        image_dir=args.input_dir,
        output_csv=args.output,
    )

    print(f"\nInference complete: {len(results)} predictions saved to {args.output}")


if __name__ == "__main__":
    main()
