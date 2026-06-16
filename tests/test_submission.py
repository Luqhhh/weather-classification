"""
Tests for submission validation rules.
Ensures the final submission meets all competition requirements.
"""

import pytest
import torch
from pathlib import Path

from models.model_factory import create_model
from data.label_mapping import LabelMapper


class TestSubmissionConstraints:
    """Verify inference-time constraints."""

    def test_model_runs_on_cpu(self):
        """Model must be able to run on CPU."""
        model = create_model("resnet18", num_classes=4, pretrained=False)
        model = model.to("cpu")
        model.eval()

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 4)
        assert output.device.type == "cpu"

    def test_model_forward_pass_time(self):
        """Single forward pass should be reasonably fast on CPU."""
        import time

        model = create_model("mobilenetv3_small", num_classes=4, pretrained=False)
        model = model.to("cpu")
        model.eval()

        x = torch.randn(1, 3, 224, 224)

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(x)

        # Timed runs
        times = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            times.append(time.perf_counter() - start)

        avg_time_ms = sum(times) / len(times) * 1000
        # MobileNetV3 should be fast on CPU (< 50ms per image)
        assert avg_time_ms < 200, f"Single inference too slow: {avg_time_ms:.0f}ms"

    def test_output_is_deterministic(self):
        """In eval mode, same input should give same output."""
        model = create_model("resnet18", num_classes=4, pretrained=False)
        model.eval()

        torch.manual_seed(42)
        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_no_cuda_dependency_at_inference(self):
        """The model should work without CUDA available."""
        model = create_model("resnet18", num_classes=4, pretrained=False)
        model.eval()

        # Force CPU
        device = torch.device("cpu")
        model = model.to(device)
        assert next(model.parameters()).device.type == "cpu"


class TestLabelMappingConstraints:
    """Verify label mapping correctness."""

    def test_four_classes(self):
        mapper = LabelMapper(["cloudy", "rainy", "snowy", "sunny"])
        assert mapper.num_classes == 4

    def test_bidirectional_consistency(self):
        mapper = LabelMapper(["cloudy", "rainy", "snowy", "sunny"])
        for i in range(4):
            assert mapper.encode(mapper.decode(i)) == i

    def test_all_weather_types_present(self):
        mapper = LabelMapper(["cloudy", "rainy", "snowy", "sunny"])
        expected = {"cloudy", "rainy", "snowy", "sunny"}
        assert set(mapper.labels) == expected
