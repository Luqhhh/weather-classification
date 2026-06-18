#!/usr/bin/env python3
"""
Error Sample Visualization

Reads error_samples.csv produced by evaluate.py and produces a grid image
showing misclassified samples with true label, predicted label, and confidence.

Usage:
    python scripts/visualize_errors.py \
        --error_csv reports/error_samples.csv \
        --data_dir data/test \
        --output_dir reports

Output:
    reports/error_samples_grid.png
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_error_samples(csv_path: Path) -> list[dict]:
    """Load error samples from CSV.

    Returns:
        List of dicts with keys: filename, true_label, predicted_label, confidence.
    """
    if not csv_path.exists():
        logger.error("Error CSV not found: %s", csv_path)
        sys.exit(1)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    if not samples:
        logger.info("No error samples found in %s — nothing to visualize.", csv_path)
        return []

    logger.info("Loaded %d error samples from %s", len(samples), csv_path)
    return samples


def _build_grid(
    samples: list[dict],
    data_dir: Path,
    max_samples: int,
    cols: int,
    thumbnail_size: int,
) -> list[dict]:
    """Prepare grid cells, loading and resizing images.

    Each cell dict: img (PIL Image), true_label, predicted_label, confidence.
    Skips samples whose image file is missing (warns and continues).
    Truncates to *max_samples* successfully loaded cells.
    """
    cells: list[dict] = []

    for sample in samples:
        if len(cells) >= max_samples:
            break

        rel_path = sample["filename"]
        img_path = data_dir / rel_path

        if not img_path.is_file():
            logger.warning("Image not found, skipping: %s", img_path)
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            logger.warning("Failed to open image, skipping: %s", img_path)
            continue

        img = img.resize((thumbnail_size, thumbnail_size), Image.LANCZOS)
        cells.append({
            "img": img,
            "true_label": sample["true_label"],
            "predicted_label": sample["predicted_label"],
            "confidence": float(sample.get("confidence", 0)),
        })

    return cells


def _render_grid(cells: list[dict], cols: int, thumbnail_size: int) -> plt.Figure:
    """Render the grid into a matplotlib Figure."""
    n = len(cells)
    if n == 0:
        return None

    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 2.5, rows * 2.8),
        squeeze=False,
    )

    for idx, cell in enumerate(cells):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ax.imshow(cell["img"])
        ax.set_xticks([])
        ax.set_yticks([])

        # Build annotation text
        conf_pct = cell["confidence"] * 100
        label_text = (
            f"True: {cell['true_label']}\n"
            f"Pred: {cell['predicted_label']}\n"
            f"conf: {conf_pct:.1f}%"
        )
        ax.set_title(label_text, fontsize=7, pad=2)

        # Thin border: green if both labels match (shouldn't happen), red otherwise
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_color("red")

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.tight_layout(pad=1.0)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize error samples from evaluate.py output"
    )
    parser.add_argument(
        "--error_csv", type=str, required=True,
        help="Path to error_samples.csv (produced by evaluate.py)",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root directory of the image dataset used during evaluation",
    )
    parser.add_argument(
        "--output_dir", type=str, default="reports",
        help="Directory for output PNG (default: reports)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=64,
        help="Maximum number of error samples to display (default: 64)",
    )
    parser.add_argument(
        "--cols", type=int, default=8,
        help="Number of columns in the grid (default: 8)",
    )
    parser.add_argument(
        "--thumbnail_size", type=int, default=128,
        help="Thumbnail size in pixels for each cell (default: 128)",
    )
    args = parser.parse_args()

    error_csv = Path(args.error_csv)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load error samples
    samples = _load_error_samples(error_csv)
    if not samples:
        logger.info("Exiting normally — no error samples to render.")
        return

    # 2. Build grid cells
    cells = _build_grid(
        samples,
        data_dir=data_dir,
        max_samples=args.max_samples,
        cols=args.cols,
        thumbnail_size=args.thumbnail_size,
    )

    if not cells:
        logger.warning("All %d samples had missing images — no grid to render.",
                       min(len(samples), args.max_samples))
        return

    logger.info("Prepared %d cells for the grid.", len(cells))

    # 3. Render and save
    fig = _render_grid(cells, cols=args.cols, thumbnail_size=args.thumbnail_size)
    if fig is None:
        logger.warning("No cells to render.")
        return

    output_path = output_dir / "error_samples_grid.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Verify output
    size_bytes = output_path.stat().st_size
    if size_bytes == 0:
        logger.error("Output file is empty: %s", output_path)
        sys.exit(1)

    logger.info(
        "Error grid saved to %s (%d cells, %.1f kB)",
        output_path, len(cells), size_bytes / 1024,
    )


if __name__ == "__main__":
    main()
