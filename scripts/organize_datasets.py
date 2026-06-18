#!/usr/bin/env python3
"""
Organize downloaded weather datasets into the project's data directory.

Handles multiple sources with different folder naming conventions:
- Kaggle weather-detection: rain/, snow/
- Kaggle multiclass: Cloudy/, Rain/, Shine/, Sunrise/
- Image2Weather (zip): cloudy/, rainy/, snowy/, sunny/

Maps all sources to our 4 target classes: cloudy, rainy, snowy, sunny.
"""

import json
import logging
import shutil
import sys
import zipfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

TARGET_CLASSES = ["cloudy", "rainy", "snowy", "sunny"]

# Folder name → target class mapping (case-insensitive)
FOLDER_TO_CLASS = {
    # cloudy
    "cloudy": "cloudy",
    # rainy
    "rain": "rainy",
    "rainy": "rainy",
    # snowy
    "snow": "snowy",
    "snowy": "snowy",
    # sunny
    "sunny": "sunny",
    "sun": "sunny",
    "shine": "sunny",
    "sunrise": "sunny",   # close enough, outdoor sunny-ish photos
}

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

VAL_SPLIT = 0.2
RANDOM_SEED = 42


def find_all_kaggle_sources() -> List[Path]:
    """Find all Kaggle dataset directories in the cache."""
    sources = []
    cache_root = Path.home() / ".cache" / "kagglehub" / "datasets"

    if not cache_root.exists():
        return sources

    # Manually add known paths (use specific version dirs to avoid duplicates)
    known_paths = [
        cache_root / "tamimresearch" / "weather-detection-image-dataset" / "versions" / "1",
        cache_root / "pratik2901" / "multiclass-weather-dataset" / "versions" / "3" / "Multi-class Weather Dataset",
    ]
    for p in known_paths:
        if p.exists() and p not in sources:
            sources.append(p)

    return sources


def scan_class_folders(source_paths: List[Path]) -> Dict[str, List[Path]]:
    """Scan all dataset sources and collect image paths per target class.

    Returns:
        Dict[target_class, List[Path to image]]
    """
    class_images: Dict[str, List[Path]] = {c: [] for c in TARGET_CLASSES}
    skipped_folders = set()

    for source in source_paths:
        logger.info(f"Scanning: {source}")

        # Find all subdirectories that could be class folders
        for item in sorted(source.rglob("*")):
            if not item.is_dir():
                continue

            folder_name = item.name.strip()
            folder_lower = folder_name.lower()

            target_class = FOLDER_TO_CLASS.get(folder_lower)
            if target_class is None:
                if folder_lower not in skipped_folders and not any(
                    c in folder_lower for c in TARGET_CLASSES
                ):
                    # Only log non-image-containing directories once
                    images_here = [
                        f for f in item.iterdir()
                        if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS
                    ]
                    if not images_here:
                        skipped_folders.add(folder_lower)
                    else:
                        skipped_folders.add(folder_lower)
                        logger.info(f"  Skipping: {folder_name}/ ({len(images_here)} images)")
                continue

            # Collect images
            images = [
                f for f in item.iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS
            ]
            class_images[target_class].extend(images)

    return class_images


def merge_and_copy(
    class_images: Dict[str, List[Path]],
    output_dir: Path,
) -> Dict[str, int]:
    """Copy all collected images to the output directory, deduplicating by filename.

    Args:
        class_images: Images grouped by target class.
        output_dir: Where to create class subdirectories.

    Returns:
        Count of unique images per class.
    """
    counts: Dict[str, int] = {}

    for cls in TARGET_CLASSES:
        cls_dir = output_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        images = class_images.get(cls, [])
        seen_names = set()
        copied = 0
        skipped_dupes = 0

        for img_path in tqdm(images, desc=f"  Copying {cls}", unit="img"):
            target_path = cls_dir / img_path.name

            # Handle duplicate filenames from different sources
            if img_path.name in seen_names:
                # Add source prefix to avoid collision
                stem = img_path.stem
                suffix = img_path.suffix
                # Use parent folder as disambiguator
                parent_name = img_path.parent.name
                new_name = f"{parent_name}_{stem}{suffix}"
                target_path = cls_dir / new_name

            if target_path.exists():
                skipped_dupes += 1
                continue

            seen_names.add(img_path.name)
            try:
                shutil.copy2(img_path, target_path)
                copied += 1
            except Exception as e:
                logger.warning(f"  Failed to copy {img_path}: {e}")

        counts[cls] = copied
        logger.info(f"  {cls}: {copied} images copied"
                     + (f" ({skipped_dupes} duplicates skipped)" if skipped_dupes else ""))

    return counts


def create_train_val_split(
    source_dir: Path,
    val_split: float = VAL_SPLIT,
    seed: int = RANDOM_SEED,
) -> Dict[str, int]:
    """Create content-deduplicated stratified train/val split from merged data."""
    logger.info(f"Creating train/val split (val={val_split:.0%}, seed={seed})...")

    rng = np.random.RandomState(seed)
    train_dir = Path("data/train")
    val_dir = Path("data/val")
    test_dir = Path("data/test")

    stats = {"train": 0, "val": 0, "duplicates_skipped": 0}
    seen_hashes: Set[str] = set()

    for cls in TARGET_CLASSES:
        cls_src = source_dir / cls
        if not cls_src.exists():
            logger.warning(f"  Source class dir not found: {cls_src}")
            continue

        images = sorted([f for f in cls_src.iterdir()
                         if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS])
        unique_images = []
        duplicates_skipped = 0
        for img in images:
            content_hash = file_sha256(img)
            if content_hash in seen_hashes:
                duplicates_skipped += 1
                continue
            seen_hashes.add(content_hash)
            unique_images.append(img)

        images = unique_images
        rng.shuffle(images)

        n_val = max(1, int(len(images) * val_split))
        val_imgs = images[:n_val]
        train_imgs = images[n_val:]

        (train_dir / cls).mkdir(parents=True, exist_ok=True)
        (val_dir / cls).mkdir(parents=True, exist_ok=True)

        for img in tqdm(train_imgs, desc=f"  train/{cls}", unit="img"):
            tgt = train_dir / cls / img.name
            if not tgt.exists():
                shutil.copy2(img, tgt)

        for img in tqdm(val_imgs, desc=f"  val/{cls}", unit="img"):
            tgt = val_dir / cls / img.name
            if not tgt.exists():
                shutil.copy2(img, tgt)

        stats["train"] += len(train_imgs)
        stats["val"] += len(val_imgs)
        stats["duplicates_skipped"] += duplicates_skipped
        logger.info(
            f"  {cls}: train={len(train_imgs)}, val={len(val_imgs)}"
            + (f", duplicates_skipped={duplicates_skipped}" if duplicates_skipped else "")
        )

    # Create minimal test set for smoke testing
    test_dir.mkdir(parents=True, exist_ok=True)
    for cls in TARGET_CLASSES:
        (test_dir / cls).mkdir(parents=True, exist_ok=True)
        val_cls_dir = val_dir / cls
        if val_cls_dir.exists():
            val_imgs = sorted(val_cls_dir.iterdir())
            for img in val_imgs[:min(5, len(val_imgs))]:
                tgt = test_dir / cls / img.name
                if not tgt.exists():
                    shutil.copy2(img, tgt)

    return stats


def file_sha256(path: Path) -> str:
    """Return a content hash used to keep duplicate images out of train/val."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_label_mapping() -> None:
    """Save the project label mapping to reports/."""
    mapping = {
        "labels": TARGET_CLASSES,
        "idx_to_label": {str(i): label for i, label in enumerate(TARGET_CLASSES)},
        "label_to_idx": {label: i for i, label in enumerate(TARGET_CLASSES)},
        "num_classes": len(TARGET_CLASSES),
    }
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    logger.info("Label mapping saved to reports/label_mapping.json")


def main():
    logger.info("=" * 60)
    logger.info("Weather Dataset Organizer")
    logger.info("=" * 60)

    # Step 1: Find all data sources
    sources = find_all_kaggle_sources()

    # Also check for Image2Weather zip
    i2w_zip = Path("data/raw/Image2Weather.zip")
    use_i2w = i2w_zip.exists()

    if not sources and not use_i2w:
        logger.error("No datasets found! Download at least one dataset first.")
        sys.exit(1)

    logger.info(f"Found {len(sources)} Kaggle source(s):")
    for s in sources:
        logger.info(f"  - {s}")
    if use_i2w:
        zip_size_gb = i2w_zip.stat().st_size / (1024**3)
        logger.info(f"  - Image2Weather zip ({zip_size_gb:.1f} GB)")

    # Step 2: Scan and collect images from Kaggle sources
    class_images = scan_class_folders(sources)

    # Step 3: Extract Image2Weather if available and complete
    if use_i2w:
        # Verify the zip is complete
        try:
            with zipfile.ZipFile(i2w_zip, "r") as test_zf:
                test_zf.namelist()  # Will raise BadZipFile if incomplete
        except (zipfile.BadZipFile, Exception) as e:
            logger.warning(f"Image2Weather zip is incomplete/corrupt (still downloading?): {e}")
            logger.warning("Skipping Image2Weather — re-run this script after download completes.")
            use_i2w = False

    if use_i2w:
        logger.info(f"\nExtracting Image2Weather (this may take a while)...")
        merge_dir = Path("data/merged")

        # Image2Weather zip structure: Image/{class}/{filename}.jpg
        # Classes inside: cloudy, rain, snow, sunny, foggy, z-other
        I2W_CLASS_MAP = {
            "cloudy": "cloudy",
            "rain": "rainy",
            "snow": "snowy",
            "sunny": "sunny",
        }

        with zipfile.ZipFile(i2w_zip, "r") as zf:
            all_files = [f for f in zf.namelist()
                         if Path(f).suffix.lower() in IMG_EXTENSIONS]
            logger.info(f"  Total image files in zip: {len(all_files)}")

            # Group files by class
            class_files_in_zip: Dict[str, List[str]] = {}
            for f in all_files:
                parts = f.split("/")
                # Structure: Image/cloudy/xxx.jpg (2 levels)
                if len(parts) >= 2:
                    folder = parts[-2].lower()
                    target = I2W_CLASS_MAP.get(folder)
                    if target:
                        if target not in class_files_in_zip:
                            class_files_in_zip[target] = []
                        class_files_in_zip[target].append(f)

            for cls in TARGET_CLASSES:
                cls_files = class_files_in_zip.get(cls, [])
                if not cls_files:
                    logger.warning(f"  No '{cls}' folder found in Image2Weather zip")
                    continue

                # Cap per class to avoid extreme imbalance (keep at most 8000 per class)
                max_per_class = 8000
                if len(cls_files) > max_per_class:
                    import random
                    random.seed(RANDOM_SEED)
                    cls_files = random.sample(cls_files, max_per_class)

                logger.info(f"  Extracting {cls}: {len(cls_files)} images...")
                (merge_dir / cls).mkdir(parents=True, exist_ok=True)
                extracted_count = 0

                for f in tqdm(cls_files, desc=f"  i2w/{cls}", unit="img"):
                    tgt = merge_dir / cls / Path(f).name
                    if not tgt.exists():
                        try:
                            with zf.open(f) as src:
                                with open(tgt, "wb") as dst:
                                    dst.write(src.read())
                            extracted_count += 1
                        except Exception as e:
                            logger.warning(f"  Failed to extract {f}: {e}")

                logger.info(f"    Extracted {extracted_count} new images")

                # Add to class_images
                extracted = list((merge_dir / cls).iterdir())
                class_images[cls].extend(extracted)

    # Step 4: Print collection summary
    logger.info("\n" + "-" * 40)
    logger.info("Images collected per class (before dedup):")
    for cls in TARGET_CLASSES:
        logger.info(f"  {cls:10s}: {len(class_images[cls]):6d} images")
    logger.info("-" * 40)

    # Step 5: Merge and copy to data/merged/
    merge_dir = Path("data/merged")
    logger.info(f"\nMerging and deduplicating to {merge_dir}/ ...")
    counts = merge_and_copy(class_images, merge_dir)

    # Step 6: Summary
    logger.info("\n" + "=" * 60)
    logger.info("Merge Summary")
    logger.info("=" * 60)
    total = 0
    for cls in TARGET_CLASSES:
        logger.info(f"  {cls:10s}: {counts.get(cls, 0):6d} unique images")
        total += counts.get(cls, 0)
    logger.info(f"  {'TOTAL':10s}: {total:6d} unique images")
    logger.info("=" * 60)

    if total == 0:
        logger.error("No images collected! Check dataset paths.")
        sys.exit(1)

    # Step 7: Train/val split
    stats = create_train_val_split(merge_dir)

    # Step 8: Save label mapping
    save_label_mapping()

    # Step 9: Final summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ Dataset organization complete!")
    logger.info(f"   Train:  {stats['train']} images → data/train/")
    logger.info(f"   Val:    {stats['val']} images → data/val/")
    logger.info(f"   Test:   ~5 per class → data/test/ (smoke testing)")
    logger.info(f"\n   reports/label_mapping.json → use this for submission!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
