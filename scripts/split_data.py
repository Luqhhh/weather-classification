#!/usr/bin/env python3
"""
Re-split existing data directories into train/val/holdout.

Useful when you have an existing dataset and need to create a holdout split,
or when competition data arrives and you need to create the splits from scratch.

Usage:
    # Split a single directory into train/val/holdout (70/15/15):
    python scripts/split_data.py --source data/merged --output_dir data

    # Split from train+val back into train/val/holdout:
    python scripts/split_data.py --source data/train data/val --output_dir data/new_split

    # Custom ratios:
    python scripts/split_data.py --source data/merged --val_split 0.15 --holdout_split 0.15
"""

import argparse
import hashlib
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_CLASSES = ["cloudy", "rainy", "snowy", "sunny"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def file_sha256(path: Path) -> str:
    """Content hash for deduplication."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_images(source_dirs: List[Path]) -> Dict[str, List[Path]]:
    """Collect all images from source directories, grouped by class."""
    class_images: Dict[str, List[Path]] = {c: [] for c in TARGET_CLASSES}

    for src in source_dirs:
        if not src.is_dir():
            logger.warning(f"Source directory not found, skipping: {src}")
            continue
        for cls in TARGET_CLASSES:
            cls_dir = src / cls
            if cls_dir.is_dir():
                for f in cls_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS:
                        class_images[cls].append(f)

    return class_images


def split_and_copy(
    class_images: Dict[str, List[Path]],
    output_dir: Path,
    val_split: float = 0.15,
    holdout_split: float = 0.15,
    seed: int = 42,
) -> Dict[str, int]:
    """Perform SHA-256 dedup + stratified split, then copy files."""
    holdout_enabled = holdout_split > 0
    rng = np.random.RandomState(seed)

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    holdout_dir = output_dir / "holdout"

    stats = {"train": 0, "val": 0, "holdout": 0, "duplicates_skipped": 0}
    seen_hashes: Set[str] = set()

    for cls in TARGET_CLASSES:
        images = sorted(class_images.get(cls, []))

        # Dedup
        unique = []
        dupes = 0
        for img in images:
            h = file_sha256(img)
            if h in seen_hashes:
                dupes += 1
                continue
            seen_hashes.add(h)
            unique.append(img)

        rng.shuffle(unique)
        total = len(unique)

        n_val = max(1, int(total * val_split))
        n_holdout = max(1, int(total * holdout_split)) if holdout_enabled else 0
        if n_val + n_holdout >= total:
            n_val = max(1, total // 3)
            n_holdout = max(1, total // 3) if holdout_enabled else 0

        val_imgs = unique[:n_val]
        holdout_imgs = unique[n_val:n_val + n_holdout] if holdout_enabled else []
        train_imgs = unique[n_val + n_holdout:]

        # Copy
        for label, imgs in [("train", train_imgs), ("val", val_imgs)]:
            tgt_dir = output_dir / label / cls
            tgt_dir.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                tgt = tgt_dir / img.name
                if not tgt.exists():
                    shutil.copy2(img, tgt)

        if holdout_enabled:
            tgt_dir = output_dir / "holdout" / cls
            tgt_dir.mkdir(parents=True, exist_ok=True)
            for img in holdout_imgs:
                tgt = tgt_dir / img.name
                if not tgt.exists():
                    shutil.copy2(img, tgt)

        stats["train"] += len(train_imgs)
        stats["val"] += len(val_imgs)
        stats["holdout"] += len(holdout_imgs)
        stats["duplicates_skipped"] += dupes

        parts = f"train={len(train_imgs)}, val={len(val_imgs)}"
        if holdout_enabled:
            parts += f", holdout={len(holdout_imgs)}"
        if dupes:
            parts += f", dupes={dupes}"
        logger.info(f"  {cls}: {parts}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Re-split images into train/val/holdout directories"
    )
    parser.add_argument(
        "--source", type=str, nargs="+", required=True,
        help="Source directories containing class subdirectories"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data",
        help="Output base directory (creates train/val/holdout subdirs)"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.15,
        help="Fraction for validation (default: 0.15)"
    )
    parser.add_argument(
        "--holdout_split", type=float, default=0.15,
        help="Fraction for holdout (default: 0.15). Set to 0 to disable."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    source_dirs = [Path(s) for s in args.source]
    output_dir = Path(args.output_dir)

    if args.val_split + args.holdout_split >= 1.0:
        logger.error(
            "val_split (%.2f) + holdout_split (%.2f) must be < 1.0",
            args.val_split, args.holdout_split,
        )
        sys.exit(1)

    logger.info("Collecting images from: %s", [str(s) for s in source_dirs])
    class_images = collect_images(source_dirs)
    for cls in TARGET_CLASSES:
        logger.info("  %s: %d images", cls, len(class_images[cls]))

    total = sum(len(v) for v in class_images.values())
    if total == 0:
        logger.error("No images found in source directories!")
        sys.exit(1)

    logger.info(
        "Splitting %d images → train=%.0f%% / val=%.0f%% / holdout=%.0f%% (seed=%d)",
        total,
        (1 - args.val_split - args.holdout_split) * 100,
        args.val_split * 100,
        args.holdout_split * 100,
        args.seed,
    )

    stats = split_and_copy(
        class_images,
        output_dir,
        val_split=args.val_split,
        holdout_split=args.holdout_split,
        seed=args.seed,
    )

    logger.info("=" * 50)
    logger.info("Split complete:")
    logger.info("  train:   %d images → %s/train/", stats["train"], output_dir)
    logger.info("  val:     %d images → %s/val/", stats["val"], output_dir)
    if stats["holdout"] > 0:
        logger.info("  holdout: %d images → %s/holdout/", stats["holdout"], output_dir)
    if stats["duplicates_skipped"]:
        logger.info("  duplicates skipped: %d", stats["duplicates_skipped"])
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
