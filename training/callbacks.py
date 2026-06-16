"""
Training Callbacks

EarlyStopping: Stop training when validation metric plateaus.
ModelCheckpoint: Save the best model weights.
TrainingLogger: Log training progress to console and file.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as improvement.
        mode: 'max' (higher is better, e.g., F1) or
              'min' (lower is better, e.g., loss).
        verbose: Print messages on improvement/stop.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "max",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_epoch = 0

        if mode == "max":
            self.is_better = lambda score, best: score > best + min_delta
            self.worst_score = float("-inf")
        elif mode == "min":
            self.is_better = lambda score, best: score < best - min_delta
            self.worst_score = float("inf")
        else:
            raise ValueError(f"Mode must be 'max' or 'min', got '{mode}'")

    def __call__(self, score: float, epoch: int) -> bool:
        """Check if training should stop.

        Args:
            score: Current metric value.
            epoch: Current epoch number (0-indexed).

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                logger.info(f"EarlyStopping: new best score {score:.4f} at epoch {epoch}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping: no improvement for {self.counter}/{self.patience} epochs"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(
                    f"EarlyStopping triggered at epoch {epoch}. "
                    f"Best score: {self.best_score:.4f} at epoch {self.best_epoch}"
                )

        return self.early_stop


class ModelCheckpoint:
    """Save model checkpoints during training.

    Tracks the top-K best models by a monitored metric.
    """

    def __init__(
        self,
        save_dir: str = "outputs/checkpoints",
        save_top_k: int = 3,
        monitor: str = "val_macro_f1",
        mode: str = "max",
        filename_pattern: str = "epoch_{epoch:03d}_{monitor}_{score:.4f}.pth",
    ):
        """
        Args:
            save_dir: Directory to save checkpoints.
            save_top_k: Keep only the top-K best models.
            monitor: Metric name to monitor.
            mode: 'max' or 'min'.
            filename_pattern: Pattern for checkpoint filenames.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.filename_pattern = filename_pattern
        self.saved_paths: List[Path] = []
        self.best_score = float("-inf") if mode == "max" else float("inf")

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        metrics: Dict,
    ) -> Optional[Path]:
        """Save a checkpoint.

        Args:
            model: The model to save.
            optimizer: Optional optimizer for resume capability.
            epoch: Current epoch number.
            metrics: Dictionary of current metrics.

        Returns:
            Path to saved checkpoint, or None if not saved.
        """
        current_score = metrics.get(self.monitor, 0)

        # Check if this checkpoint should be saved
        if self.mode == "max" and current_score <= self.best_score:
            # Always save latest for resume, even if not best
            self._save_latest(model, optimizer, epoch, metrics)
            return None
        elif self.mode == "min" and current_score >= self.best_score:
            self._save_latest(model, optimizer, epoch, metrics)
            return None

        self.best_score = current_score

        # Build checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "metrics": metrics,
            "best_score": self.best_score,
        }

        filename = self.filename_pattern.format(
            epoch=epoch + 1,
            monitor=self.monitor,
            score=current_score,
        )
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)
        self.saved_paths.append(filepath)

        # Remove excess checkpoints
        self._prune_checkpoints()

        logger.info(f"Checkpoint saved: {filepath} (score={current_score:.4f})")
        return filepath

    def _save_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        metrics: Dict,
    ) -> None:
        """Save a 'latest' checkpoint for resume capability."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "metrics": metrics,
            "best_score": self.best_score,
        }
        filepath = self.save_dir / "latest.pth"
        torch.save(checkpoint, filepath)

    def _prune_checkpoints(self) -> None:
        """Remove excess checkpoints beyond the top-K."""
        # Sort by score in filename (best first)
        # This is approximate — for exact sorting, parse scores from filenames
        if len(self.saved_paths) > self.save_top_k:
            # Remove oldest saved (by insertion order)
            while len(self.saved_paths) > self.save_top_k:
                old_path = self.saved_paths.pop(0)
                if old_path.exists():
                    old_path.unlink()
                    logger.info(f"Pruned old checkpoint: {old_path}")


class TrainingLogger:
    """Log training progress to console and JSON file."""

    def __init__(self, log_file: Optional[str] = None):
        self.start_time = time.time()
        self.epoch_times: List[float] = []
        self.log_file = Path(log_file) if log_file else None

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_start(self) -> float:
        """Record epoch start time."""
        return time.time()

    def on_epoch_end(self, epoch: int, metrics: Dict) -> None:
        """Log epoch results."""
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)

        # Format per-class F1
        per_class_str = ""
        if "per_class" in metrics:
            per_class_items = []
            for cls_name, cls_metrics in metrics["per_class"].items():
                per_class_items.append(f"{cls_name}: {cls_metrics['f1']:.3f}")
            per_class_str = " | ".join(per_class_items)

        log_msg = (
            f"Epoch {epoch + 1:3d} | "
            f"F1: {metrics.get('macro_f1', 0):.4f} | "
            f"Acc: {metrics.get('accuracy', 0):.4f} | "
            f"Loss: {metrics.get('val_loss', 0):.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        if per_class_str:
            log_msg += f"\n  Per-class: {per_class_str}"

        logger.info(log_msg)

        # Append to JSON log
        if self.log_file:
            self._write_json_log(epoch, metrics)

    def _write_json_log(self, epoch: int, metrics: Dict) -> None:
        """Append metrics to a JSON log file."""
        record = {
            "epoch": epoch + 1,
            "timestamp": time.time(),
            **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
        }
        # Append to JSON lines file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
