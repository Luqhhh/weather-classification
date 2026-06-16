"""
Weather Image Dataset

Robust dataset loader that handles:
- Multiple image formats (jpg, jpeg, png, bmp, webp)
- Grayscale → RGB conversion
- RGBA → RGB conversion
- Corrupted images (skip/log, don't crash)
- Variable image sizes (handled by transforms)
- Directory-based class structure
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, random_split

from .label_mapping import LabelMapper, detect_label_mapping

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class WeatherDataset(Dataset):
    """Weather image classification dataset.

    Expects directory structure:
        data_dir/
        ├── cloudy/
        │   ├── img001.jpg
        │   └── ...
        ├── rainy/
        ├── snowy/
        └── sunny/

    If an annotations_file (CSV) is provided, uses that instead of directory scanning.
    CSV format: filename,label
    """

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

    def __init__(
        self,
        data_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        label_mapper: Optional[LabelMapper] = None,
        annotations_file: Optional[Union[str, Path]] = None,
        auto_detect_labels: bool = True,
    ):
        """
        Args:
            data_dir: Root directory containing class subdirectories.
            transform: torchvision transforms to apply.
            label_mapper: Pre-built LabelMapper. If None, auto-detected.
            annotations_file: Optional CSV with (filename, label) pairs.
            auto_detect_labels: Auto-detect class labels from directory structure.
        """
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Build or use label mapper
        if label_mapper is not None:
            self.label_mapper = label_mapper
        elif auto_detect_labels:
            self.label_mapper = detect_label_mapping(self.data_dir)
        else:
            raise ValueError("Either label_mapper or auto_detect_labels=True is required.")

        # Build image list
        self.images: List[Tuple[Path, int]] = []  # (path, label_idx)
        self.bad_images: List[Tuple[Path, str]] = []  # (path, error_reason)

        if annotations_file is not None:
            self._load_from_annotations(annotations_file)
        else:
            self._load_from_directory()

        logger.info(
            f"WeatherDataset: {len(self.images)} images across "
            f"{self.label_mapper.num_classes} classes "
            f"({', '.join(self.label_mapper.labels)})"
        )
        if self.bad_images:
            logger.warning(
                f"{len(self.bad_images)} bad images found and skipped. "
                f"See dataset_report for details."
            )

    def _load_from_directory(self) -> None:
        """Scan data directory for images in class subdirectories."""
        for class_name in self.label_mapper.labels:
            class_dir = self.data_dir / class_name
            if not class_dir.is_dir():
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            class_idx = self.label_mapper.encode(class_name)
            for file_path in class_dir.iterdir():
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                    continue

                # Quick integrity check: try opening the image
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Verify without fully loading
                except Exception as e:
                    self.bad_images.append((file_path, str(e)))
                    continue

                self.images.append((file_path, class_idx))

    def _load_from_annotations(self, annotations_file: Union[str, Path]) -> None:
        """Load image list from a CSV annotations file.

        Expected CSV format: filename,label
        """
        import csv

        annotations_file = Path(annotations_file)
        with open(annotations_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"]
                label = row["label"]
                file_path = self.data_dir / filename

                if not file_path.is_file():
                    self.bad_images.append((file_path, f"File not found: {filename}"))
                    continue

                try:
                    class_idx = self.label_mapper.encode(label)
                except KeyError:
                    logger.warning(f"Unknown label '{label}' for {filename}, skipping")
                    self.bad_images.append(
                        (file_path, f"Unknown label: {label}")
                    )
                    continue

                self.images.append((file_path, class_idx))

    def _safe_load_image(self, path: Path) -> Optional[Image.Image]:
        """Safely load an image, handling common issues.

        Returns:
            PIL Image in RGB mode, or None if loading fails.
        """
        try:
            img = Image.open(path)

            # Handle grayscale → RGB
            if img.mode == "L":
                img = img.convert("RGB")
            # Handle RGBA → RGB
            elif img.mode == "RGBA":
                # Paste onto white background
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode == "P":
                img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")

            return img

        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")
            return None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get an image and its label index.

        Returns:
            (image_tensor, label_index)
        """
        img_path, label_idx = self.images[idx]

        img = self._safe_load_image(img_path)
        if img is None:
            # Fallback: return a blank image (shouldn't happen due to pre-check)
            logger.error(f"Failed to load image at runtime: {img_path}")
            img = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        return img, label_idx

    def get_class_distribution(self) -> Dict[str, int]:
        """Return the count of images per class."""
        distribution = {label: 0 for label in self.label_mapper.labels}
        for _, label_idx in self.images:
            distribution[self.label_mapper.decode(label_idx)] += 1
        return distribution


def create_dataloaders(
    data_dir: Union[str, Path],
    label_mapper: Optional[LabelMapper] = None,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, LabelMapper]:
    """Create training and validation DataLoaders.

    Args:
        data_dir: Root directory with class subdirectories.
        label_mapper: Pre-built LabelMapper (auto-detected if None).
        train_transform: Transforms for training.
        val_transform: Transforms for validation.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        val_split: Fraction of data for validation.
        seed: Random seed for reproducible split.
        pin_memory: Pin memory for faster GPU transfer.

    Returns:
        (train_loader, val_loader, label_mapper)
    """
    # Auto-detect labels if needed
    if label_mapper is None:
        label_mapper = detect_label_mapping(data_dir)

    # Full dataset
    full_dataset = WeatherDataset(
        data_dir=data_dir,
        transform=None,  # We'll apply transforms per split
        label_mapper=label_mapper,
    )

    # Stratified split
    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # Apply transforms (via wrapper to avoid pickling issues)
    train_dataset = _TransformWrapper(train_dataset, train_transform)
    val_dataset = _TransformWrapper(val_dataset, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(
        f"Created dataloaders: train={train_size}, val={val_size} "
        f"(split={val_split:.0%}, seed={seed})"
    )
    return train_loader, val_loader, label_mapper


class _TransformWrapper(Dataset):
    """Wraps a subset dataset to apply transforms."""

    def __init__(self, subset: Dataset, transform: Optional[Callable]):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.subset[idx]
        if self.transform and isinstance(img, Image.Image):
            img = self.transform(img)
        return img, label
