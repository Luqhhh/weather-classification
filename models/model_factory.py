"""
Model Factory

Creates weather classifiers with different backbone architectures.
All models share the same classification head structure.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch.nn as nn
from torchvision import models

from .base import WeatherClassifier

logger = logging.getLogger(__name__)

# Registry of available model configurations
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "resnet18": {
        "builder": models.resnet18,
        "in_features": 512,
        "description": "ResNet-18 — Stable baseline, fast CPU inference",
    },
    "resnet34": {
        "builder": models.resnet34,
        "in_features": 512,
        "description": "ResNet-34 — Slightly better accuracy than ResNet-18",
    },
    "resnet50": {
        "builder": models.resnet50,
        "in_features": 2048,
        "description": "ResNet-50 — Higher capacity, may overfit small datasets",
    },
    "efficientnet_b0": {
        "builder": models.efficientnet_b0,
        "in_features": 1280,
        "description": "EfficientNet-B0 — Good accuracy/speed balance",
    },
    "efficientnet_b1": {
        "builder": models.efficientnet_b1,
        "in_features": 1280,
        "description": "EfficientNet-B1 — Slightly better than B0, slower",
    },
    "efficientnet_b2": {
        "builder": models.efficientnet_b2,
        "in_features": 1408,
        "description": "EfficientNet-B2 — Higher accuracy, slower CPU",
    },
    "mobilenetv3_small": {
        "builder": models.mobilenet_v3_small,
        "in_features": 576,
        "description": "MobileNetV3-Small — Lightweight, fastest CPU inference",
    },
    "mobilenetv3_large": {
        "builder": models.mobilenet_v3_large,
        "in_features": 960,
        "description": "MobileNetV3-Large — Good speed, decent accuracy",
    },
    "convnext_tiny": {
        "builder": models.convnext_tiny,
        "in_features": 768,
        "description": "ConvNeXt-Tiny — Modern CNN, check CPU speed",
    },
    "shufflenet_v2_x1_0": {
        "builder": models.shufflenet_v2_x1_0,
        "in_features": 1024,
        "description": "ShuffleNetV2 — Very lightweight, fast CPU",
    },
    "squeezenet1_0": {
        "builder": models.squeezenet1_0,
        "in_features": 512,
        "description": "SqueezeNet — Very small model, limited accuracy",
    },
    "densenet121": {
        "builder": models.densenet121,
        "in_features": 1024,
        "description": "DenseNet-121 — Dense connections, may be slow on CPU",
    },
}


def create_model(
    name: str,
    num_classes: int = 4,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
    **kwargs,
) -> WeatherClassifier:
    """Create a WeatherClassifier with the specified backbone.

    Args:
        name: Model name (key in MODEL_REGISTRY).
        num_classes: Number of output classes.
        pretrained: Use pretrained ImageNet weights.
        dropout: Dropout probability for classification head.
        freeze_backbone: Freeze backbone during training.
        **kwargs: Passed to the backbone builder.

    Returns:
        Configured WeatherClassifier.

    Raises:
        ValueError: If model name is not registered.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(list_available_models())
        raise ValueError(
            f"Unknown model '{name}'. Available models: {available}"
        )

    config = MODEL_REGISTRY[name]
    builder = config["builder"]
    in_features = config["in_features"]

    # Build backbone
    weights = "DEFAULT" if pretrained else None
    try:
        backbone = builder(weights=weights, **kwargs)
    except TypeError:
        # Fallback for older torchvision versions
        backbone = builder(pretrained=pretrained, **kwargs)

    # Remove the original classifier/FC layer
    backbone = _strip_classifier(backbone, name)

    # Wrap in weather classifier
    model = WeatherClassifier(
        backbone=backbone,
        num_classes=num_classes,
        in_features=in_features,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        backbone_name=name,
    )

    param_info = model.get_param_count()
    logger.info(
        f"Created {name}: {param_info['total_millions']}M params "
        f"({param_info['trainable_millions']}M trainable), "
        f"num_classes={num_classes}, dropout={dropout}"
    )

    return model


def _strip_classifier(backbone: nn.Module, name: str) -> nn.Module:
    """Remove the classification head from a backbone.

    Different architectures store the classifier in different attributes.
    """
    if name.startswith("resnet") or name.startswith("resnext"):
        backbone.fc = nn.Identity()
    elif name.startswith("efficientnet"):
        backbone.classifier = nn.Identity()
    elif name.startswith("mobilenetv3"):
        backbone.classifier = nn.Identity()
    elif name.startswith("convnext"):
        backbone.classifier = nn.Identity()
    elif name.startswith("shufflenet"):
        backbone.fc = nn.Identity()
    elif name.startswith("squeezenet"):
        backbone.classifier = nn.Identity()
    elif name.startswith("densenet"):
        backbone.classifier = nn.Identity()
    elif name.startswith("vgg"):
        backbone.classifier = nn.Identity()
    else:
        # Generic: try common attribute names
        for attr in ["fc", "classifier", "head"]:
            if hasattr(backbone, attr):
                setattr(backbone, attr, nn.Identity())
                break
    return backbone


def list_available_models() -> list:
    """Return list of registered model names."""
    return sorted(MODEL_REGISTRY.keys())


def get_model_info(name: str) -> Dict[str, Any]:
    """Get information about a registered model.

    Args:
        name: Model name in MODEL_REGISTRY.

    Returns:
        Dict with model metadata.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(list_available_models())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    config = MODEL_REGISTRY[name]
    return {
        "name": name,
        "in_features": config["in_features"],
        "description": config["description"],
    }
