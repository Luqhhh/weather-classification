#!/usr/bin/env python3
"""
Data Analysis Script

Analyzes the weather image dataset and generates a comprehensive report.
Run this FIRST before any training to understand your data.

Usage:
    python scripts/analyze_data.py --data_dir /path/to/train [--output_dir reports]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset_report import generate_report
from data.label_mapping import detect_label_mapping, save_label_mapping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze weather image dataset and generate a report"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to the training data directory (with class subdirectories)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="reports",
        help="Directory for output files (default: reports/)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    # Auto-detect and save label mapping
    logger.info("Detecting label mapping...")
    mapper = detect_label_mapping(data_dir)
    save_label_mapping(mapper, Path(args.output_dir) / "label_mapping.json")
    logger.info(f"Classes detected: {mapper.labels}")
    logger.info(f"Label mapping saved to {args.output_dir}/label_mapping.json")

    # Generate full report
    logger.info("Generating dataset report...")
    report = generate_report(data_dir, args.output_dir)

    # Print summary
    print("\n" + "=" * 50)
    print("Dataset Analysis Summary")
    print("=" * 50)
    print(f"Total images:      {report['total_images']}")
    print(f"Classes:           {', '.join(report['classes'])}")
    print(f"Imbalance ratio:   {report['imbalance_ratio']}")
    print(f"Bad images:        {report['bad_images_count']}")
    print(f"Recommended size:  {report['recommended_image_size']}px")
    print(f"\nClass distribution:")
    for cls, info in report["class_distribution"].items():
        print(f"  {cls:10s}: {info['count']:5d} ({info['percentage']:5.1f}%)")
    print(f"\nFull report saved to {args.output_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
