"""
Label Mapping Management

Auto-detects class labels from directory structure or annotation files.
NEVER hardcode label order — always derive from the actual data.

The mapping must be consistent across training, validation, and inference.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class LabelMapper:
    """Manages the bidirectional mapping between class names and integer indices.

    Attributes:
        idx_to_label: Dict[int, str] — index → class name
        label_to_idx: Dict[str, int] — class name → index
        num_classes: int — number of classes
    """

    def __init__(self, labels: List[str]):
        """Initialize from a list of class names in order.

        Args:
            labels: Class names sorted alphabetically or in the order
                    they appear in the dataset directory.
        """
        if len(labels) != len(set(labels)):
            raise ValueError(f"Duplicate labels detected: {labels}")

        self.idx_to_label: Dict[int, str] = {i: label for i, label in enumerate(labels)}
        self.label_to_idx: Dict[str, int] = {label: i for i, label in enumerate(labels)}
        self.num_classes = len(labels)
        self.labels = labels

    def __repr__(self) -> str:
        return f"LabelMapper({self.label_to_idx})"

    def __len__(self) -> int:
        return self.num_classes

    def encode(self, label: str) -> int:
        """Convert a class name to its integer index."""
        if label not in self.label_to_idx:
            raise KeyError(
                f"Unknown label '{label}'. Known labels: {list(self.label_to_idx.keys())}"
            )
        return self.label_to_idx[label]

    def decode(self, index: int) -> str:
        """Convert an integer index to its class name."""
        if index not in self.idx_to_label:
            raise KeyError(
                f"Unknown index {index}. Known indices: {list(self.idx_to_label.keys())}"
            )
        return self.idx_to_label[index]

    def encode_batch(self, labels: List[str]) -> List[int]:
        """Convert a list of class names to indices."""
        return [self.encode(label) for label in labels]

    def decode_batch(self, indices: List[int]) -> List[str]:
        """Convert a list of indices to class names."""
        return [self.decode(idx) for idx in indices]

    def save(self, path: Union[str, Path]) -> None:
        """Save the label mapping to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "labels": self.labels,
            "idx_to_label": {str(k): v for k, v in self.idx_to_label.items()},
            "label_to_idx": self.label_to_idx,
            "num_classes": self.num_classes,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Label mapping saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LabelMapper":
        """Load a label mapping from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mapper = cls(data["labels"])
        # Verify consistency
        expected_num = data.get("num_classes", len(data["labels"]))
        if mapper.num_classes != expected_num:
            logger.warning(
                f"Loaded mapping has {mapper.num_classes} classes, "
                f"expected {expected_num}"
            )
        return mapper


def detect_label_mapping(
    data_dir: Union[str, Path],
    sort: bool = True,
) -> LabelMapper:
    """Auto-detect class labels from a directory structure.

    Expects the data directory to have one subdirectory per class:
        data_dir/
        ├── cloudy/
        ├── rainy/
        ├── snowy/
        └── sunny/

    Args:
        data_dir: Path to the dataset root directory.
        sort: If True, sort class names alphabetically for deterministic ordering.
              If False, use OS directory listing order (not recommended).

    Returns:
        LabelMapper initialized with detected class names.
    """
    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all subdirectories (each should be a class)
    class_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir()],
        key=lambda x: x.name if sort else x.name,
    )

    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {data_dir}")

    labels = [d.name for d in class_dirs]
    logger.info(f"Detected {len(labels)} classes from directory structure: {labels}")

    mapper = LabelMapper(labels)
    return mapper


def load_label_mapping(path: Union[str, Path]) -> LabelMapper:
    """Load label mapping from a JSON file."""
    return LabelMapper.load(path)


def save_label_mapping(mapper: LabelMapper, path: Union[str, Path]) -> None:
    """Save label mapping to a JSON file."""
    mapper.save(path)
