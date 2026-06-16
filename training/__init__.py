"""Training module for weather image classification.

Provides training loop, metric computation, loss functions,
and callbacks for model training and evaluation.
"""

from .trainer import Trainer
from .metrics import (
    compute_metrics,
    compute_macro_f1,
    compute_per_class_metrics,
    plot_confusion_matrix,
    MetricsTracker,
)
from .losses import (
    create_loss_function,
    LabelSmoothingCrossEntropy,
    FocalLoss,
)
from .callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TrainingLogger,
)

__all__ = [
    "Trainer",
    "compute_metrics",
    "compute_macro_f1",
    "compute_per_class_metrics",
    "plot_confusion_matrix",
    "MetricsTracker",
    "create_loss_function",
    "LabelSmoothingCrossEntropy",
    "FocalLoss",
    "EarlyStopping",
    "ModelCheckpoint",
    "TrainingLogger",
]
