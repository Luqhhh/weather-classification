#!/usr/bin/env python3
"""
CPU Benchmark Script

Measures inference speed, memory usage, and estimates total scoring time.
Critical step before final submission — ensures the model fits within 70 minutes.

Usage:
    python scripts/benchmark_cpu.py --weights weights/best.pth --model resnet18
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.label_mapping import load_label_mapping
from inference.benchmark import CpuBenchmark
from models.model_factory import create_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CPU inference performance"
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to model weights (.pth)"
    )
    parser.add_argument(
        "--model", type=str, default="resnet18",
        help="Model architecture name"
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
        "--output", type=str, default="reports/cpu_benchmark.csv",
        help="Output CSV for benchmark results"
    )
    parser.add_argument(
        "--num_images", type=int, default=3000,
        help="Estimated number of scoring images"
    )
    args = parser.parse_args()

    # Label mapping
    if args.label_mapping:
        label_mapper = load_label_mapping(args.label_mapping)
    else:
        # Minimal mapping for benchmark
        from data.label_mapping import LabelMapper
        label_mapper = LabelMapper(["cloudy", "rainy", "snowy", "sunny"])

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
    model.load_state_dict(state)
    logger.info(f"Loaded weights from {args.weights}")

    # Run benchmark
    benchmark = CpuBenchmark(
        model=model,
        input_size=args.image_size,
        batch_sizes=[1, 4, 8, 16, 32, 64],
    )
    results = benchmark.run()

    # Save results
    benchmark.save_csv(results, args.output)

    # Final verdict
    se = results["scoring_estimate"]
    print("\n" + "=" * 50)
    print("CPU BENCHMARK VERDICT")
    print("=" * 50)
    print(f"Model:           {results['model_name']}")
    print(f"Params:          {results['params_millions']}M")
    print(f"Weight size:     {results['weight_size_mb']} MB")
    print(f"Optimal batch:   {results['optimal_batch_size']}")
    print(f"Throughput:      {results['optimal_throughput']} imgs/s")
    print(f"\nScoring estimate ({se['num_scoring_images']} images):")
    print(f"  Total time:    {se['estimated_total_time_min']} min")
    print(f"  Within limit:  {'✅ YES' if se['within_70min'] else '❌ NO — optimize or use smaller model'}")

    if not se["within_70min"]:
        print("\n⚠️  WARNING: Model is too slow for the 70-minute limit!")
        print("   Consider:")
        print("   1. Using a smaller model (MobileNetV3, EfficientNet-B0)")
        print("   2. Reducing input image size")
        print("   3. Using ONNX runtime for CPU speedup")
        print("   4. Increasing batch size")

    print("=" * 50)


if __name__ == "__main__":
    main()
