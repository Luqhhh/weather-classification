"""Model definitions for weather image classification.

Provides a unified interface for multiple backbone architectures,
with a factory pattern for easy model switching.
"""

from .base import WeatherClassifier
from .model_factory import (
    create_model,
    list_available_models,
    get_model_info,
    MODEL_REGISTRY,
)

__all__ = [
    "WeatherClassifier",
    "create_model",
    "list_available_models",
    "get_model_info",
    "MODEL_REGISTRY",
]
