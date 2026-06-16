"""
Base Weather Classifier

Wraps a backbone CNN with a classification head for 4-class weather prediction.
Provides a unified forward pass, weight loading, and ONNX export.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class WeatherClassifier(nn.Module):
    """Wrapper around a backbone CNN for weather classification.

    Architecture:
        backbone → pooling → dropout → fc_head → logits

    Supports:
    - Any torchvision/timm backbone
    - Dropout for regularization
    - Optional backbone freezing
    - ONNX export for CPU speedup
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 4,
        in_features: int = 512,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
        backbone_name: str = "unknown",
    ):
        """
        Args:
            backbone: Pre-built CNN backbone (without classification head).
            num_classes: Number of output classes (default 4 for weather).
            in_features: Number of features from backbone (before FC layer).
            dropout: Dropout probability for regularization.
            freeze_backbone: If True, freeze all backbone parameters.
            backbone_name: Human-readable name for logging.
        """
        super().__init__()
        self.backbone = backbone
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # Optional backbone freezing
        if freeze_backbone:
            self._freeze_backbone()

        # Classification head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, num_classes)

        # Initialize head
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen — only head will be trained")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        features = self.backbone(x)

        # Handle different backbone output shapes
        if isinstance(features, tuple):
            features = features[-1]  # Some backbones return (features, aux)

        if features.dim() == 4:
            features = self.pool(features)
            features = features.view(features.size(0), -1)

        features = self.dropout(features)
        logits = self.fc(features)
        return logits

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict class probabilities and indices.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            (probabilities, predicted_indices)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return probs, preds

    def get_param_count(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "total_millions": round(total / 1e6, 2),
            "trainable_millions": round(trainable / 1e6, 2),
        }

    def get_weight_size_mb(self, path: Optional[Union[str, Path]] = None) -> float:
        """Estimate model weight file size in MB.

        Args:
            path: If provided, check file size of saved weights.
                  Otherwise, estimate from state_dict.

        Returns:
            Estimated size in MB.
        """
        if path is not None:
            return Path(path).stat().st_size / (1024 * 1024)

        # Rough estimate: count parameters × 4 bytes (float32)
        param_count = self.get_param_count()["total"]
        return param_count * 4 / (1024 * 1024)

    def export_onnx(
        self,
        output_path: Union[str, Path],
        input_size: int = 224,
        batch_size: int = 1,
    ) -> None:
        """Export model to ONNX format for CPU inference speedup.

        Args:
            output_path: Path to save the .onnx file.
            input_size: Input image size.
            batch_size: Dynamic batch size dimension.
        """
        self.eval()
        dummy_input = torch.randn(batch_size, 3, input_size, input_size)

        torch.onnx.export(
            self,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        logger.info(f"Model exported to ONNX: {output_path}")
