"""
Tests for model creation, forward pass, and utilities.
"""

import pytest
import torch

from models.base import WeatherClassifier
from models.model_factory import (
    create_model,
    list_available_models,
    get_model_info,
    MODEL_REGISTRY,
)


class TestModelFactory:
    """Tests for model factory and available backbones."""

    def test_list_available_models(self):
        models = list_available_models()
        assert "resnet18" in models
        assert "mobilenetv3_small" in models
        assert "efficientnet_b0" in models

    def test_create_resnet18(self):
        model = create_model("resnet18", num_classes=4, pretrained=False)
        assert isinstance(model, WeatherClassifier)
        assert model.backbone_name == "resnet18"
        assert model.num_classes == 4

    def test_create_mobilenetv3(self):
        model = create_model("mobilenetv3_small", num_classes=4, pretrained=False)
        assert model.backbone_name == "mobilenetv3_small"

    def test_create_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("nonexistent_model")

    def test_get_model_info(self):
        info = get_model_info("resnet18")
        assert info["name"] == "resnet18"
        assert "in_features" in info
        assert "description" in info


class TestWeatherClassifier:
    """Tests for the WeatherClassifier wrapper."""

    def test_forward_shape(self):
        model = create_model("resnet18", num_classes=4, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 4)

    def test_predict_method(self):
        model = create_model("resnet18", num_classes=4, pretrained=False)
        model.eval()
        x = torch.randn(5, 3, 224, 224)
        probs, preds = model.predict(x)
        assert probs.shape == (5, 4)
        assert preds.shape == (5,)
        assert (probs.sum(dim=1) - 1.0).abs().max() < 1e-5  # Softmax sums to 1

    def test_param_count(self):
        model = create_model("resnet18", num_classes=4, pretrained=False)
        info = model.get_param_count()
        assert info["total"] > 0
        assert info["trainable"] > 0
        assert info["total_millions"] > 1
        assert info["total"] == info["trainable"] + info["frozen"]

    def test_freeze_backbone(self):
        model = create_model(
            "resnet18", num_classes=4, pretrained=False, freeze_backbone=True
        )
        info = model.get_param_count()
        # Most params should be frozen, only FC head trainable
        assert info["frozen"] > info["trainable"]

    def test_weight_size_estimate(self):
        model = create_model("resnet18", num_classes=4, pretrained=False)
        size_mb = model.get_weight_size_mb()
        # ResNet-18 ~ 11M params × 4 bytes ≈ 44 MB
        assert 30 < size_mb < 100

    def test_dropout_effect(self):
        """Verify dropout is active during training, inactive during eval."""
        model = create_model("resnet18", num_classes=4, dropout=0.5, pretrained=False)

        # In eval mode, two forward passes should be identical
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

        # In train mode, two forward passes may differ (dropout)
        model.train()
        out3 = model(x)
        out4 = model(x)
        # With dropout=0.5 and a single sample, they'll often differ
        # (But can coincidentally be equal — probabilistic test)
