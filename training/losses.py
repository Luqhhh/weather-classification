"""
Loss Functions for Weather Classification

Includes:
- Standard CrossEntropy
- Label Smoothing CrossEntropy (helps with easy confusion between similar weather types)
- Focal Loss (addresses class imbalance, focuses on hard examples)
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing.

    Label smoothing prevents the model from becoming overconfident,
    which helps with the subtle visual differences between weather types.

    Args:
        smoothing: Smoothing factor (0 = no smoothing, 0.1 = moderate).
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed cross-entropy loss.

        Args:
            logits: (N, C) raw model outputs.
            targets: (N,) integer class indices.

        Returns:
            Scalar loss.
        """
        n_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)

        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = (-smooth_targets * log_probs).sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Down-weights easy examples and focuses on hard ones.
    Useful when some weather classes are harder to classify than others.

    Args:
        alpha: Per-class weighting factor (list of floats, one per class).
               If None, no class weighting.
        gamma: Focusing parameter. Higher = more focus on hard examples.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: (N, C) raw model outputs.
            targets: (N,) integer class indices.

        Returns:
            Scalar loss.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)  # p_t = probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def create_loss_function(
    name: str = "cross_entropy",
    num_classes: int = 4,
    class_weights: Optional[list] = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Factory function for creating loss functions.

    Args:
        name: Loss type: 'cross_entropy', 'label_smoothing', 'focal'.
        num_classes: Number of classes (for label smoothing).
        class_weights: Optional per-class weights for imbalanced data.
        label_smoothing: Smoothing factor (for label_smoothing type).
        focal_gamma: Focusing parameter (for focal type).

    Returns:
        PyTorch loss module.
    """
    name = name.lower()

    if name == "cross_entropy":
        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        return nn.CrossEntropyLoss(weight=weight)

    elif name == "label_smoothing":
        return LabelSmoothingCrossEntropy(smoothing=label_smoothing)

    elif name == "focal":
        alpha = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        return FocalLoss(alpha=alpha, gamma=focal_gamma)

    else:
        raise ValueError(
            f"Unknown loss function: {name}. "
            f"Use 'cross_entropy', 'label_smoothing', or 'focal'."
        )
