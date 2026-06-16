"""
Metrics for Weather Classification

Primary metric: Macro F1 (each class weighted equally)
Secondary: Per-class precision, recall, F1
Tertiary: Confusion matrix, accuracy
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def compute_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> float:
    """Compute macro F1 score.

    Args:
        y_true: Ground truth labels (integers).
        y_pred: Predicted labels (integers).
        average: Averaging method for F1 ('macro' by default per competition rules).

    Returns:
        F1 score as a float.
    """
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision, recall, and F1.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class names in index order.

    Returns:
        Dict mapping class_name → {precision, recall, f1, support}.
    """
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Support (number of true instances per class)
    supports = np.bincount(y_true, minlength=len(class_names))[:len(class_names)]

    metrics = {}
    for i, name in enumerate(class_names):
        metrics[name] = {
            "precision": round(float(precisions[i]), 4),
            "recall": round(float(recalls[i]), 4),
            "f1": round(float(f1s[i]), 4),
            "support": int(supports[i]),
        }

    return metrics


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict:
    """Compute all metrics for a classification run.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class names in index order.

    Returns:
        Dict with macro_f1, accuracy, per_class_metrics, confusion_matrix.
    """
    macro_f1 = compute_macro_f1(y_true, y_pred)
    accuracy = float(accuracy_score(y_true, y_pred))
    per_class = compute_per_class_metrics(y_true, y_pred, class_names)
    cm = confusion_matrix(y_true, y_pred)

    # Identify weak classes (F1 below average)
    f1_values = [per_class[name]["f1"] for name in class_names]
    avg_per_class_f1 = np.mean(f1_values) if f1_values else 0
    weak_classes = [
        name for name in class_names
        if per_class[name]["f1"] < avg_per_class_f1
    ]

    return {
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "weak_classes": weak_classes,
        "avg_per_class_f1": round(avg_per_class_f1, 4),
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Plot a confusion matrix.

    Args:
        cm: Confusion matrix array (n_classes × n_classes).
        class_names: Class name labels.
        save_path: Optional path to save the figure.
        normalize: If True, normalize rows to sum=1.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    return fig


@dataclass
class MetricsTracker:
    """Tracks metrics across training epochs.

    Automatically identifies the best epoch by macro F1.
    """

    class_names: List[str]
    history: List[Dict] = field(default_factory=list)

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metrics: Dict,
        lr: float,
    ) -> None:
        """Record metrics for one epoch."""
        self.history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "per_class_f1": {
                name: val_metrics["per_class"][name]["f1"]
                for name in self.class_names
            },
            "lr": lr,
        })

    def get_best_epoch(self) -> Dict:
        """Return the epoch with the highest macro F1."""
        if not self.history:
            return {}
        return max(self.history, key=lambda x: x["val_macro_f1"])

    def get_latest(self) -> Dict:
        """Return the most recent epoch."""
        if not self.history:
            return {}
        return self.history[-1]

    def to_dataframe(self):
        """Convert history to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.history)
