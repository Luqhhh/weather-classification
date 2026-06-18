"""
Tests for the data pipeline: dataset loading, label mapping, transforms.
"""

import tempfile
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from data.dataset import WeatherDataset, create_dataloaders
from data.label_mapping import LabelMapper, detect_label_mapping, load_label_mapping, save_label_mapping
from data.transforms import build_transforms, get_train_transforms, get_val_transforms


class TestLabelMapper:
    """Tests for LabelMapper."""

    def test_basic_mapping(self):
        mapper = LabelMapper(["cloudy", "rainy", "snowy", "sunny"])
        assert mapper.num_classes == 4
        assert mapper.encode("cloudy") == 0
        assert mapper.decode(0) == "cloudy"
        assert mapper.encode("sunny") == 3
        assert mapper.decode(3) == "sunny"

    def test_encode_unknown_label(self):
        mapper = LabelMapper(["cloudy", "rainy"])
        with pytest.raises(KeyError):
            mapper.encode("hurricane")

    def test_decode_unknown_index(self):
        mapper = LabelMapper(["cloudy", "rainy"])
        with pytest.raises(KeyError):
            mapper.decode(5)

    def test_save_load_roundtrip(self, tmp_path):
        mapper = LabelMapper(["cloudy", "rainy", "snowy", "sunny"])
        path = tmp_path / "labels.json"
        mapper.save(path)

        loaded = load_label_mapping(path)
        assert loaded.labels == mapper.labels
        assert loaded.num_classes == mapper.num_classes
        assert loaded.encode("rainy") == mapper.encode("rainy")

    def test_detect_from_directory(self, tmp_path):
        # Create mock directory structure
        for cls in ["cloudy", "rainy", "snowy", "sunny"]:
            (tmp_path / cls).mkdir()
            # Create a dummy image
            img = Image.new("RGB", (64, 64), color=(100, 100, 100))
            img.save(tmp_path / cls / "dummy.jpg")

        mapper = detect_label_mapping(tmp_path)
        assert set(mapper.labels) == {"cloudy", "rainy", "snowy", "sunny"}


class TestTransforms:
    """Tests for data transforms."""

    def test_train_transform_output_shape(self):
        transform = get_train_transforms(image_size=224)
        img = Image.new("RGB", (300, 200), color=(100, 150, 200))
        output = transform(img)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)

    def test_val_transform_output_shape(self):
        transform = get_val_transforms(image_size=224)
        img = Image.new("RGB", (300, 200), color=(100, 150, 200))
        output = transform(img)
        assert output.shape == (3, 224, 224)

    def test_build_transforms_factory(self):
        train = build_transforms("train", image_size=160)
        val = build_transforms("val", image_size=160)
        test = build_transforms("test", image_size=160)

        img = Image.new("RGB", (100, 100))
        assert train(img).shape == (3, 160, 160)
        assert val(img).shape == (3, 160, 160)
        assert test(img).shape == (3, 160, 160)

    def test_normalization_range(self):
        """Ensure normalized output is roughly zero-centered."""
        transform = get_val_transforms(image_size=224)
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        output = transform(img)
        # With ImageNet normalization, mid-gray should be near-zero
        assert -0.5 < output.mean() < 0.5


class TestWeatherDataset:
    """Tests for WeatherDataset."""

    def test_basic_loading(self, tmp_path):
        # Create mock dataset
        for cls in ["cloudy", "rainy"]:
            (tmp_path / cls).mkdir()
            for i in range(5):
                img = Image.new("RGB", (64, 64), color=(i * 50, 100, 150))
                img.save(tmp_path / cls / f"img_{i}.jpg")

        dataset = WeatherDataset(
            data_dir=tmp_path,
            transform=None,
        )
        assert len(dataset) == 10
        assert dataset.label_mapper.num_classes == 2

    def test_class_distribution(self, tmp_path):
        for cls in ["cloudy", "rainy"]:
            (tmp_path / cls).mkdir()
            for i in range(3 if cls == "cloudy" else 7):
                img = Image.new("RGB", (64, 64))
                img.save(tmp_path / cls / f"img_{i}.jpg")

        dataset = WeatherDataset(data_dir=tmp_path, transform=None)
        dist = dataset.get_class_distribution()
        assert dist["cloudy"] == 3
        assert dist["rainy"] == 7

    def test_handle_grayscale(self, tmp_path):
        (tmp_path / "cloudy").mkdir(parents=True)
        img = Image.new("L", (64, 64), color=128)  # Grayscale
        img.save(tmp_path / "cloudy" / "gray.jpg")

        transform = get_val_transforms(image_size=64)
        dataset = WeatherDataset(data_dir=tmp_path, transform=transform)
        tensor, label = dataset[0]
        assert tensor.shape == (3, 64, 64)  # Should be converted to RGB

    def test_handle_rgba(self, tmp_path):
        (tmp_path / "cloudy").mkdir(parents=True)
        img = Image.new("RGBA", (64, 64), color=(100, 150, 200, 128))
        img.save(tmp_path / "cloudy" / "rgba.png")

        transform = get_val_transforms(image_size=64)
        dataset = WeatherDataset(data_dir=tmp_path, transform=transform)
        tensor, label = dataset[0]
        assert tensor.shape == (3, 64, 64)  # RGBA → RGB

    def test_skip_corrupted_images(self, tmp_path):
        (tmp_path / "cloudy").mkdir(parents=True)
        # Create a valid image
        img = Image.new("RGB", (64, 64))
        img.save(tmp_path / "cloudy" / "good.jpg")
        # Create a corrupted "image"
        (tmp_path / "cloudy" / "bad.jpg").write_text("not an image")

        dataset = WeatherDataset(data_dir=tmp_path, transform=None)
        assert len(dataset) == 1  # Only the good image
        assert len(dataset.bad_images) == 1  # Bad image recorded

    def test_create_dataloaders_deduplicates_and_stratifies_auto_split(self, tmp_path):
        for cls in ["cloudy", "rainy"]:
            (tmp_path / cls).mkdir()
            for i in range(10):
                color = (i * 10, 80, 120) if cls == "cloudy" else (120, i * 10, 80)
                _save_image(tmp_path / cls / f"{cls}_{i}.jpg", color=color)

        duplicate_bytes = (tmp_path / "cloudy" / "cloudy_0.jpg").read_bytes()
        (tmp_path / "cloudy" / "cloudy_duplicate.jpg").write_bytes(duplicate_bytes)

        train_loader, val_loader, _ = create_dataloaders(
            data_dir=tmp_path,
            batch_size=4,
            num_workers=0,
            val_split=0.2,
            seed=42,
        )

        train_paths = _paths_from_wrapped_subset(train_loader.dataset)
        val_paths = _paths_from_wrapped_subset(val_loader.dataset)

        assert len(train_paths) == 16
        assert len(val_paths) == 4
        assert _class_counts(train_paths) == {"cloudy": 8, "rainy": 8}
        assert _class_counts(val_paths) == {"cloudy": 2, "rainy": 2}
        assert _hashes(train_paths).isdisjoint(_hashes(val_paths))

    def test_create_dataloaders_drops_explicit_train_val_leakage(self, tmp_path):
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        for split_dir in [train_dir, val_dir]:
            for cls in ["cloudy", "rainy"]:
                (split_dir / cls).mkdir(parents=True)

        _save_image(train_dir / "cloudy" / "train_cloudy.jpg", color=(10, 20, 30))
        _save_image(train_dir / "rainy" / "train_rainy.jpg", color=(40, 50, 60))
        _save_image(val_dir / "cloudy" / "val_cloudy.jpg", color=(70, 80, 90))
        _save_image(val_dir / "rainy" / "val_rainy.jpg", color=(100, 110, 120))

        leaked_bytes = (val_dir / "cloudy" / "val_cloudy.jpg").read_bytes()
        (train_dir / "cloudy" / "leaked_from_val.jpg").write_bytes(leaked_bytes)
        (val_dir / "cloudy" / "val_cloudy_duplicate.jpg").write_bytes(leaked_bytes)

        train_loader, val_loader, _ = create_dataloaders(
            data_dir=train_dir,
            val_dir=val_dir,
            batch_size=2,
            num_workers=0,
        )

        train_paths = _paths_from_wrapped_subset(train_loader.dataset)
        val_paths = _paths_from_wrapped_subset(val_loader.dataset)

        assert len(train_paths) == 2
        assert len(val_paths) == 2
        assert all(path.name != "leaked_from_val.jpg" for path in train_paths)
        assert sum(path.name.startswith("val_cloudy") for path in val_paths) == 1
        assert _hashes(train_paths).isdisjoint(_hashes(val_paths))


def _save_image(path: Path, color: tuple) -> None:
    img = Image.new("RGB", (64, 64), color=color)
    img.save(path)


def _paths_from_wrapped_subset(dataset) -> list:
    subset = dataset.subset
    base_dataset = subset.dataset
    return [base_dataset.images[idx][0] for idx in subset.indices]


def _class_counts(paths: list) -> dict:
    counts = {}
    for path in paths:
        counts[path.parent.name] = counts.get(path.parent.name, 0) + 1
    return counts


def _hashes(paths: list) -> set:
    return {hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
