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
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import hashlib

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

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
            for file_path in sorted(class_dir.iterdir()):
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
    val_dir: Optional[Union[str, Path]] = None,
    label_mapper: Optional[LabelMapper] = None,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
    pin_memory: bool = True,
    deduplicate: bool = True,
    multiprocessing_context: str = "spawn",
    sampler_config: Optional[Dict] = None,
) -> Tuple[DataLoader, DataLoader, LabelMapper]:
    """Create training and validation DataLoaders.

    Args:
        data_dir: Root directory with class subdirectories.
        val_dir: Optional explicit validation directory. If provided, no
            auto-split is performed.
        label_mapper: Pre-built LabelMapper (auto-detected if None).
        train_transform: Transforms for training.
        val_transform: Transforms for validation.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        val_split: Fraction of training data to use for validation (when
            val_dir is not provided).
        seed: Random seed for train/val split and sampler reproducibility.
        pin_memory: Pin memory for faster GPU transfer.
        deduplicate: Remove duplicate image content before splitting and remove
            train samples that duplicate explicit validation images.
        multiprocessing_context: '' for default (fork on Linux), 'spawn' to
            avoid WSL2 deadlocks with num_workers > 0, 'fork', or 'forkserver'.
        sampler_config: Optional dict with keys ``name`` and sampler-specific
            parameters.  ``name`` values:
            - ``"none"`` or ``None``: no sampler, DataLoader uses ``shuffle=True``.
            - ``"class_balanced"``: use ``WeightedRandomSampler`` with weights
              computed as 1 / class-frequency on the training subset.
            Default behaviour (``sampler_config=None``) is unchanged.

    Returns:
        (train_loader, val_loader, label_mapper)
    """
    # Auto-detect labels if needed
    if label_mapper is None:
        label_mapper = detect_label_mapping(data_dir)

    train_base = WeatherDataset(
        data_dir=data_dir,
        transform=None,  # We'll apply transforms per split
        label_mapper=label_mapper,
    )

    if val_dir is not None:
        val_base = WeatherDataset(
            data_dir=val_dir,
            transform=None,
            label_mapper=label_mapper,
        )
        train_indices = list(range(len(train_base)))
        val_indices = list(range(len(val_base)))

        if deduplicate:
            val_indices = _deduplicate_indices(val_base, split_name="val")
            val_hashes = _hashes_for_indices(val_base, val_indices)
            train_indices = _deduplicate_indices(
                train_base,
                split_name="train",
                excluded_hashes=val_hashes,
            )

        train_dataset = _TransformWrapper(Subset(train_base, train_indices), train_transform)
        val_dataset = _TransformWrapper(Subset(val_base, val_indices), val_transform)
        train_size = len(train_dataset)
        val_size = len(val_dataset)

        logger.info(
            f"Created dataloaders from explicit splits: train={train_size}, "
            f"val={val_size}, val_dir={val_dir}"
        )
    else:
        indices = list(range(len(train_base)))
        if deduplicate:
            indices = _deduplicate_indices(train_base, split_name="full")

        train_indices, val_indices = _stratified_split_indices(
            train_base,
            indices=indices,
            val_split=val_split,
            seed=seed,
        )

        train_dataset = _TransformWrapper(Subset(train_base, train_indices), train_transform)
        val_dataset = _TransformWrapper(Subset(train_base, val_indices), val_transform)
        train_size = len(train_dataset)
        val_size = len(val_dataset)

        logger.info(
            f"Created deduplicated stratified dataloaders: train={train_size}, "
            f"val={val_size} (split={val_split:.0%}, seed={seed})"
        )

    # Only pass multiprocessing_context when num_workers > 0 (PyTorch constraint)
    mp_context = multiprocessing_context or None if num_workers > 0 else None

    # --- Sampler ---
    train_sampler = _build_train_sampler(
        train_dataset=train_dataset,
        sampler_config=sampler_config,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        multiprocessing_context=mp_context,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        multiprocessing_context=mp_context,
    )

    return train_loader, val_loader, label_mapper


def _file_sha256(path: Path) -> str:
    """Return a content hash for an image file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _deduplicate_indices(
    dataset: WeatherDataset,
    split_name: str,
    excluded_hashes: Optional[set] = None,
) -> List[int]:
    """Keep one sample per content hash and optionally exclude known hashes.

    Duplicate content can leak validation signal if one copy lands in train and
    another lands in validation. This helper keeps the first copy in stable
    dataset order and drops later copies.
    """
    excluded_hashes = excluded_hashes or set()
    seen_hashes = set()
    kept_indices: List[int] = []
    dropped_duplicates = 0
    dropped_excluded = 0
    label_conflicts = 0
    hash_to_label: Dict[str, int] = {}

    for idx, (path, label_idx) in enumerate(dataset.images):
        content_hash = _file_sha256(path)
        if content_hash in excluded_hashes:
            dropped_excluded += 1
            continue
        if content_hash in seen_hashes:
            dropped_duplicates += 1
            if hash_to_label.get(content_hash) != label_idx:
                label_conflicts += 1
            continue

        seen_hashes.add(content_hash)
        hash_to_label[content_hash] = label_idx
        kept_indices.append(idx)

    if dropped_duplicates or dropped_excluded or label_conflicts:
        logger.warning(
            "Deduplicated %s split: kept=%d, dropped_duplicates=%d, "
            "dropped_cross_split=%d, label_conflicts=%d",
            split_name,
            len(kept_indices),
            dropped_duplicates,
            dropped_excluded,
            label_conflicts,
        )

    return kept_indices


def _hashes_for_indices(dataset: WeatherDataset, indices: Iterable[int]) -> set:
    """Return content hashes for selected dataset indices."""
    return {_file_sha256(dataset.images[idx][0]) for idx in indices}


def _stratified_split_indices(
    dataset: WeatherDataset,
    indices: Sequence[int],
    val_split: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Split indices by class so validation preserves class proportions."""
    if not 0 < val_split < 1:
        raise ValueError(f"val_split must be between 0 and 1, got {val_split}")

    generator = torch.Generator().manual_seed(seed)
    by_label: Dict[int, List[int]] = defaultdict(list)
    for idx in indices:
        _, label_idx = dataset.images[idx]
        by_label[label_idx].append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    for label_idx in sorted(by_label):
        class_indices = by_label[label_idx]
        perm = torch.randperm(len(class_indices), generator=generator).tolist()
        shuffled = [class_indices[i] for i in perm]
        n_val = max(1, int(len(shuffled) * val_split)) if len(shuffled) > 1 else 1
        n_val = min(n_val, len(shuffled))
        val_indices.extend(shuffled[:n_val])
        train_indices.extend(shuffled[n_val:])

    train_counts = _class_counts_for_indices(dataset, train_indices)
    val_counts = _class_counts_for_indices(dataset, val_indices)
    logger.info("Stratified split train class counts: %s", train_counts)
    logger.info("Stratified split val class counts: %s", val_counts)

    return train_indices, val_indices


def _class_counts_for_indices(dataset: WeatherDataset, indices: Sequence[int]) -> Dict[str, int]:
    """Return class counts for a list of dataset indices."""
    counts = Counter()
    for idx in indices:
        _, label_idx = dataset.images[idx]
        counts[dataset.label_mapper.decode(label_idx)] += 1
    return dict(counts)


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


def _build_train_sampler(
    train_dataset: Dataset,
    sampler_config: Optional[Dict],
    seed: int,
) -> Optional[WeightedRandomSampler]:
    """Build a sampler for the training DataLoader.

    Args:
        train_dataset: Training dataset (may be a ``Subset`` wrapping a
            ``WeatherDataset``).
        sampler_config: Dict with ``name`` and optional parameters, or
            ``None`` (default — no sampler).
        seed: Random seed for sampler reproducibility.

    Returns:
        A ``WeightedRandomSampler`` instance, or ``None`` if no sampler
        should be used.
    """
    if sampler_config is None:
        return None

    name = (sampler_config.get("name") or "none").lower()
    if name == "none":
        return None

    # --- Resolve the underlying WeatherDataset ---
    base_dataset = train_dataset
    indices = None

    # Unwrap _TransformWrapper (holds a .subset attribute)
    if isinstance(base_dataset, _TransformWrapper):
        base_dataset = base_dataset.subset

    while isinstance(base_dataset, Subset):
        indices = (
            base_dataset.indices
            if indices is None
            else [indices[i] for i in base_dataset.indices]
        )
        base_dataset = base_dataset.dataset

    if not isinstance(base_dataset, WeatherDataset):
        raise TypeError(
            "Sampler requires the underlying dataset to be a WeatherDataset, "
            f"got {type(base_dataset)}"
        )

    # --- Build per-sample weights ---
    if name == "class_balanced":
        if indices is None:
            indices = list(range(len(base_dataset)))

        # Class counts on the *training* subset only
        class_counts: Dict[int, int] = Counter()
        sample_labels: List[int] = []
        for idx in indices:
            _, label_idx = base_dataset.images[idx]
            class_counts[label_idx] += 1
            sample_labels.append(label_idx)

        # Weight = (1 / count) ^ effective_scale
        #   effective_scale=1.0 → full inverse-frequency balancing
        #   effective_scale=0.5 → sqrt balancing (mild)
        #   effective_scale=0.0 → uniform (no balancing)
        effective_scale = float(sampler_config.get("effective_scale", 1.0))
        effective_scale = max(0.0, min(1.0, effective_scale))

        num_classes = len(class_counts)
        if effective_scale == 1.0:
            class_weight = {
                cls: 1.0 / max(count, 1) for cls, count in class_counts.items()
            }
        elif effective_scale == 0.0:
            class_weight = {cls: 1.0 for cls in class_counts}
        else:
            class_weight = {
                cls: (1.0 / max(count, 1)) ** effective_scale
                for cls, count in class_counts.items()
            }

        sample_weights = [class_weight[lbl] for lbl in sample_labels]

        logger.info(
            "Class-balanced sampler enabled: class_counts=%s, "
            "effective_scale=%.2f, class_weights={%s}, seed=%d",
            dict(class_counts),
            effective_scale,
            ", ".join(
                f"{base_dataset.label_mapper.decode(cls)}: {w:.6f}"
                for cls, w in sorted(class_weight.items())
            ),
            seed,
        )

        generator = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float64),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )

    raise ValueError(f"Unknown sampler name: {name}")
