"""Data pipeline for weather image classification.

Provides robust dataset loading, automatic label mapping detection,
configurable transforms, and data analysis utilities.
"""

from .dataset import WeatherDataset, create_dataloaders
from .transforms import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
    build_transforms,
)
from .label_mapping import (
    LabelMapper,
    detect_label_mapping,
    load_label_mapping,
    save_label_mapping,
)
from .dataset_report import DatasetAnalyzer, generate_report

__all__ = [
    "WeatherDataset",
    "create_dataloaders",
    "get_train_transforms",
    "get_val_transforms",
    "get_test_transforms",
    "build_transforms",
    "LabelMapper",
    "detect_label_mapping",
    "load_label_mapping",
    "save_label_mapping",
    "DatasetAnalyzer",
    "generate_report",
]
