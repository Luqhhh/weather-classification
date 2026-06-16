"""
Tests for the data pipeline: dataset loading, label mapping, transforms.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from data.dataset import WeatherDataset
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
