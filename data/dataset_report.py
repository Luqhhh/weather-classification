"""
Dataset Analysis & Reporting

Generates a comprehensive analysis of the weather image dataset:
- Class distribution & balance
- Image size statistics
- Format distribution
- Corrupted/bad image detection
- Suggested preprocessing strategies
"""

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFile

from .label_mapping import LabelMapper, detect_label_mapping

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


class DatasetAnalyzer:
    """Analyzes a weather image dataset and generates a report."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        label_mapper: Optional[LabelMapper] = None,
    ):
        self.data_dir = Path(data_dir)
        self.label_mapper = label_mapper or detect_label_mapping(self.data_dir)

        # Analysis results
        self.class_counts: Dict[str, int] = {}
        self.bad_images: List[Tuple[Path, str]] = []
        self.image_sizes: List[Tuple[int, int]] = []
        self.image_formats: Counter[str] = Counter()
        self.image_sizes_per_class: Dict[str, List[Tuple[int, int]]] = {}
        self.file_sizes_bytes: List[int] = []

    def analyze(self) -> Dict:
        """Run full dataset analysis.

        Returns:
            Dict with analysis results suitable for report generation.
        """
        logger.info(f"Analyzing dataset at {self.data_dir}...")

        for class_name in self.label_mapper.labels:
            self.image_sizes_per_class[class_name] = []
            class_dir = self.data_dir / class_name

            if not class_dir.is_dir():
                logger.warning(f"Class directory not found: {class_dir}")
                self.class_counts[class_name] = 0
                continue

            count = 0
            for file_path in class_dir.iterdir():
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in SUPPORTED_FORMATS:
                    continue

                self.image_formats[file_path.suffix.lower()] += 1

                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        self.image_sizes.append((width, height))
                        self.image_sizes_per_class[class_name].append((width, height))
                        self.file_sizes_bytes.append(file_path.stat().st_size)
                        img.verify()
                    count += 1
                except Exception as e:
                    self.bad_images.append((file_path, str(e)))

            self.class_counts[class_name] = count

        total = sum(self.class_counts.values())
        logger.info(f"Analysis complete: {total} images, {len(self.bad_images)} bad")

        return self._build_report_dict()

    def _build_report_dict(self) -> Dict:
        """Build a structured report dictionary."""
        total = sum(self.class_counts.values())

        # Class distribution
        distribution = {}
        for cls, count in self.class_counts.items():
            distribution[cls] = {
                "count": count,
                "percentage": round(count / total * 100, 2) if total > 0 else 0,
            }

        # Class balance
        if self.class_counts:
            counts = list(self.class_counts.values())
            min_count, max_count = min(counts), max(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")
        else:
            imbalance_ratio = 0

        # Image size statistics
        if self.image_sizes:
            widths = [s[0] for s in self.image_sizes]
            heights = [s[1] for s in self.image_sizes]
            areas = [w * h for w, h in self.image_sizes]
            aspect_ratios = [w / h for w, h in self.image_sizes]
        else:
            widths, heights, areas, aspect_ratios = [], [], [], []

        # Format distribution
        format_dist = dict(self.image_formats)

        # File size stats
        if self.file_sizes_bytes:
            file_sizes_mb = [s / (1024 * 1024) for s in self.file_sizes_bytes]
        else:
            file_sizes_mb = []

        return {
            "total_images": total,
            "num_classes": self.label_mapper.num_classes,
            "classes": self.label_mapper.labels,
            "class_distribution": distribution,
            "imbalance_ratio": round(imbalance_ratio, 2),
            "is_balanced": imbalance_ratio < 1.5 if imbalance_ratio > 0 else True,
            "bad_images_count": len(self.bad_images),
            "bad_images": [(str(p), reason) for p, reason in self.bad_images[:50]],
            "image_formats": format_dist,
            "image_size_stats": {
                "min_width": int(np.min(widths)) if widths else 0,
                "max_width": int(np.max(widths)) if widths else 0,
                "mean_width": round(float(np.mean(widths)), 1) if widths else 0,
                "min_height": int(np.min(heights)) if heights else 0,
                "max_height": int(np.max(heights)) if heights else 0,
                "mean_height": round(float(np.mean(heights)), 1) if heights else 0,
                "min_area": int(np.min(areas)) if areas else 0,
                "max_area": int(np.max(areas)) if areas else 0,
                "mean_area": round(float(np.mean(areas)), 1) if areas else 0,
                "mean_aspect_ratio": round(float(np.mean(aspect_ratios)), 3) if aspect_ratios else 0,
            },
            "file_size_stats": {
                "min_mb": round(np.min(file_sizes_mb), 3) if file_sizes_mb else 0,
                "max_mb": round(np.max(file_sizes_mb), 3) if file_sizes_mb else 0,
                "mean_mb": round(np.mean(file_sizes_mb), 3) if file_sizes_mb else 0,
                "total_mb": round(sum(file_sizes_mb), 2) if file_sizes_mb else 0,
            },
            "recommended_image_size": self._recommend_image_size(),
        }

    def _recommend_image_size(self) -> int:
        """Recommend a resize dimension based on observed image sizes."""
        if not self.image_sizes:
            return 224
        areas = [w * h for w, h in self.image_sizes]
        median_area = np.median(areas)
        # Choose a size close to the median but in common model sizes
        common_sizes = [160, 192, 224, 256, 288, 320, 384]
        target = int(np.sqrt(median_area))
        return min(common_sizes, key=lambda x: abs(x - target))

    def save_bad_images(self, path: Union[str, Path]) -> None:
        """Save list of bad/corrupted images to a file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            for file_path, reason in self.bad_images:
                f.write(f"{file_path}\t{reason}\n")
        logger.info(f"Bad images list saved to {path}")

    def save_class_distribution_csv(self, path: Union[str, Path]) -> None:
        """Save class distribution to CSV."""
        path = Path(path)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "count", "percentage"])
            total = sum(self.class_counts.values())
            for cls, count in self.class_counts.items():
                pct = round(count / total * 100, 2) if total > 0 else 0
                writer.writerow([cls, count, pct])
        logger.info(f"Class distribution saved to {path}")

    def save_label_mapping(self, path: Union[str, Path]) -> None:
        """Save the detected label mapping."""
        self.label_mapper.save(path)


def generate_report(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path] = "reports",
) -> Dict:
    """Generate a full dataset analysis report.

    Creates:
    - reports/dataset_report.md
    - reports/class_distribution.csv
    - reports/bad_images.txt
    - reports/label_mapping.json

    Args:
        data_dir: Path to the dataset directory.
        output_dir: Directory for output files.

    Returns:
        Analysis report as a dictionary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = DatasetAnalyzer(data_dir)
    report = analyzer.analyze()

    # Save artifacts
    analyzer.save_class_distribution_csv(output_dir / "class_distribution.csv")
    analyzer.save_bad_images(output_dir / "bad_images.txt")
    analyzer.save_label_mapping(output_dir / "label_mapping.json")

    # Generate markdown report
    md_report = _format_markdown_report(report)
    with open(output_dir / "dataset_report.md", "w", encoding="utf-8") as f:
        f.write(md_report)

    logger.info(f"Full dataset report saved to {output_dir}/")
    return report


def _format_markdown_report(report: Dict) -> str:
    """Format analysis results as a Markdown report."""
    lines = [
        "# Weather Dataset Analysis Report",
        "",
        f"**Total Images**: {report['total_images']}",
        f"**Number of Classes**: {report['num_classes']}",
        f"**Classes**: {', '.join(report['classes'])}",
        "",
        "## Class Distribution",
        "",
        "| Class | Count | Percentage |",
        "|-------|-------|------------|",
    ]
    for cls, info in report["class_distribution"].items():
        lines.append(f"| {cls} | {info['count']} | {info['percentage']}% |")

    lines.extend([
        "",
        f"**Imbalance Ratio** (max/min): {report['imbalance_ratio']}",
        f"**Is Balanced**: {'Yes' if report['is_balanced'] else 'No — consider class weights or oversampling'}",
        "",
        "## Image Size Statistics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Min Width | {report['image_size_stats']['min_width']}px |",
        f"| Max Width | {report['image_size_stats']['max_width']}px |",
        f"| Mean Width | {report['image_size_stats']['mean_width']}px |",
        f"| Min Height | {report['image_size_stats']['min_height']}px |",
        f"| Max Height | {report['image_size_stats']['max_height']}px |",
        f"| Mean Height | {report['image_size_stats']['mean_height']}px |",
        f"| Mean Aspect Ratio | {report['image_size_stats']['mean_aspect_ratio']} |",
        f"| Recommended Resize | **{report['recommended_image_size']}px** |",
        "",
        "## Image Formats",
        "",
        "| Format | Count |",
        "|--------|-------|",
    ])
    for fmt, count in report["image_formats"].items():
        lines.append(f"| {fmt} | {count} |")

    lines.extend([
        "",
        "## File Size Statistics",
        "",
        f"- Min: {report['file_size_stats']['min_mb']} MB",
        f"- Max: {report['file_size_stats']['max_mb']} MB",
        f"- Mean: {report['file_size_stats']['mean_mb']} MB",
        f"- Total: {report['file_size_stats']['total_mb']} MB",
        "",
        "## Bad / Corrupted Images",
        "",
        f"**Count**: {report['bad_images_count']}",
    ])

    if report["bad_images"]:
        lines.append("")
        lines.append("First 50 bad images:")
        for path, reason in report["bad_images"]:
            lines.append(f"- `{path}`: {reason}")

    lines.extend([
        "",
        "## Preprocessing Recommendations",
        "",
        f"1. Resize all images to **{report['recommended_image_size']}×{report['recommended_image_size']}px**",
        "2. Use RandomResizedCrop during training for robustness to varied sizes",
        "3. Convert all images to RGB (handle grayscale, RGBA)",
        "4. Use conservative ColorJitter — weather depends on color information",
        "5. Skip the identified bad images during training",
    ])

    if not report["is_balanced"]:
        lines.append("6. Apply class weights or focal loss to handle class imbalance")

    return "\n".join(lines)
