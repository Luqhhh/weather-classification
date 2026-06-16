"""
Tests for inference pipeline: predictor, benchmark, submit checker.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from data.label_mapping import LabelMapper, save_label_mapping
from inference.benchmark import CpuBenchmark
from inference.predictor import WeatherPredictor
from inference.submit_checker import SubmitChecker
from models.model_factory import create_model


class TestWeatherPredictor:
    """Tests for WeatherPredictor."""

    @pytest.fixture
    def predictor(self):
        model = create_model("resnet18", num_classes=4, pretrained=False)
        model.eval()
        mapper = LabelMapper(["cloudy", "rainy", "snowy", "sunny"])
        return WeatherPredictor(
            model=model,
            label_mapper=mapper,
            image_size=64,
            device="cpu",
            batch_size=4,
        )

    def test_predict_single_file(self, predictor, tmp_path):
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)

        result = predictor.predict_file(img_path)
        assert "filename" in result
        assert "predicted_class" in result
        assert "class_index" in result
        assert "probabilities" in result
        assert result["filename"] == "test.jpg"
        assert result["predicted_class"] in predictor.label_mapper.labels

    def test_predict_batch(self, predictor, tmp_path):
        for i in range(10):
            img = Image.new("RGB", (64, 64), color=(i * 25, 100, 150))
            img.save(tmp_path / f"img_{i:03d}.jpg")

        output_csv = tmp_path / "predictions.csv"
        results = predictor.predict_batch(tmp_path, output_csv=output_csv, show_progress=False)

        assert len(results) == 10
        assert output_csv.exists()

        # Check CSV content
        with open(output_csv) as f:
            lines = f.readlines()
            assert lines[0].strip() == "filename,prediction"
            assert len(lines) == 11  # header + 10 rows

    def test_estimate_total_time(self, predictor):
        estimate = predictor.estimate_total_time(num_images=100)
        assert "estimated_total_time_min" in estimate
        assert "within_70min_limit" in estimate
        assert estimate["num_images"] == 100


class TestCpuBenchmark:
    """Tests for CpuBenchmark."""

    def test_benchmark_runs(self):
        model = create_model("mobilenetv3_small", num_classes=4, pretrained=False)
        model.eval()
        benchmark = CpuBenchmark(
            model=model,
            input_size=64,
            batch_sizes=[1, 4],
        )
        results = benchmark.run()
        assert results["model_name"] == "mobilenetv3_small"
        assert "1" in results["batch_results"]
        assert "4" in results["batch_results"]
        assert "scoring_estimate" in results

    def test_save_csv(self, tmp_path):
        model = create_model("mobilenetv3_small", num_classes=4, pretrained=False)
        model.eval()
        benchmark = CpuBenchmark(model=model, input_size=64, batch_sizes=[1])
        results = benchmark.run()
        output = tmp_path / "bench.csv"
        benchmark.save_csv(results, str(output))
        assert output.exists()


class TestSubmitChecker:
    """Tests for SubmitChecker."""

    def test_weights_check(self, tmp_path):
        # Create a minimal model weights file
        model = create_model("resnet18", num_classes=4, pretrained=False)
        weights_path = tmp_path / "model.pth"
        torch.save(model.state_dict(), weights_path)

        # Create mock files
        (tmp_path / "inference.py").write_text("# inference script")
        (tmp_path / "test_images").mkdir()
        label_path = tmp_path / "labels.json"
        mapper = LabelMapper(["cloudy", "rainy", "snowy", "sunny"])
        mapper.save(label_path)

        checker = SubmitChecker(
            inference_script=str(tmp_path / "inference.py"),
            weights_path=str(weights_path),
            test_images_dir=str(tmp_path / "test_images"),
            label_mapping_path=str(label_path),
            submit_dir=str(tmp_path),
        )
        # Run individual checks
        passed, msg = checker.check_weights_exist()
        assert passed

        passed, msg = checker.check_label_mapping_valid()
        assert passed
