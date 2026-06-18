"""
Training Loop

Implements a full training pipeline:
- Train/val loop with macro F1 as primary metric
- Automatic mixed precision (AMP) support
- Gradient accumulation
- Learning rate scheduling
- Checkpointing and early stopping
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from .metrics import compute_metrics, MetricsTracker, plot_confusion_matrix
from .callbacks import EarlyStopping, ModelCheckpoint, TrainingLogger
from data.label_mapping import LabelMapper

logger = logging.getLogger(__name__)


class Trainer:
    """Training handler for weather classification models.

    Manages the full training lifecycle with macro F1 optimization.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        label_mapper: Optional[LabelMapper] = None,
        device: str = "cuda",
        use_amp: bool = True,
        grad_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        config: Optional[Dict] = None,
    ):
        """
        Args:
            model: The WeatherClassifier model.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            criterion: Loss function.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler (optional).
            label_mapper: LabelMapper for class names.
            device: Device to train on ('cuda' or 'cpu').
            use_amp: Use automatic mixed precision (CUDA only).
            grad_accumulation_steps: Number of steps to accumulate gradients.
            max_grad_norm: Max gradient norm for clipping.
            config: Full training configuration dict.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.label_mapper = label_mapper
        self.device = device
        self.use_amp = use_amp and device == "cuda"
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.config = config or {}

        self.scaler = GradScaler() if self.use_amp else None
        self.metrics_tracker: Optional[MetricsTracker] = None

        # Move model to device
        self.model = self.model.to(device)

    def fit(
        self,
        epochs: int,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint: Optional[ModelCheckpoint] = None,
        logger_callback: Optional[TrainingLogger] = None,
        output_dir: str = "outputs",
    ) -> MetricsTracker:
        """Run the full training loop.

        Args:
            epochs: Maximum number of epochs.
            early_stopping: EarlyStopping callback (optional).
            checkpoint: ModelCheckpoint callback (optional).
            logger_callback: TrainingLogger callback (optional).
            output_dir: Directory for outputs.

        Returns:
            MetricsTracker with training history.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        class_names = self.label_mapper.labels if self.label_mapper else [str(i) for i in range(4)]
        self.metrics_tracker = MetricsTracker(class_names=class_names)
        best_f1 = 0.0

        logger.info(f"Starting training on {self.device} for {epochs} epochs")
        logger.info(f"Model: {self.model.backbone_name}, "
                     f"Params: {self.model.get_param_count()['total_millions']}M")

        for epoch in range(epochs):
            epoch_start = time.time()

            # Training phase
            train_loss = self._train_epoch(epoch)

            # Validation phase
            val_loss, y_true, y_pred = self._validate_epoch()

            # Compute metrics
            val_metrics = compute_metrics(
                np.array(y_true), np.array(y_pred), class_names
            )

            # Update scheduler (epoch-level)
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                # Handle ReduceLROnPlateau (uses metric) vs others
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["macro_f1"])
                else:
                    self.scheduler.step()

            # Track metrics
            self.metrics_tracker.update(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
                lr=current_lr,
            )

            # Logging
            log_metrics = {
                "macro_f1": val_metrics["macro_f1"],
                "accuracy": val_metrics["accuracy"],
                "val_loss": val_loss,
                "train_loss": train_loss,
                "lr": current_lr,
                "per_class": val_metrics["per_class"],
                "confusion_matrix": val_metrics["confusion_matrix"],
            }
            if logger_callback:
                logger_callback.on_epoch_end(epoch, log_metrics)

            # Checkpoint
            if checkpoint:
                checkpoint.save(self.model, self.optimizer, epoch, {
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_loss": val_loss,
                })

            # Track best
            if val_metrics["macro_f1"] > best_f1:
                best_f1 = val_metrics["macro_f1"]
                # Save best model separately
                torch.save(
                    self.model.state_dict(),
                    output_dir / "best_model.pth",
                )

            # Early stopping
            if early_stopping and early_stopping(val_metrics["macro_f1"], epoch):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            epoch_time = time.time() - epoch_start
            logger.debug(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")

        # Final: save confusion matrix
        if self.metrics_tracker.history:
            cm = np.array(val_metrics["confusion_matrix"])
            plot_confusion_matrix(
                cm,
                class_names,
                save_path=str(output_dir / "confusion_matrix_final.png"),
                title=f"Confusion Matrix (Best F1: {best_f1:.4f})",
            )

        logger.info(f"Training complete. Best macro F1: {best_f1:.4f}")
        return self.metrics_tracker

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)

            if self.use_amp:
                with autocast(device_type=self.device):
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)
                    loss = loss / self.grad_accumulation_steps

                self.scaler.scale(loss).backward()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, targets)
                loss = loss / self.grad_accumulation_steps
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accumulation_steps

        avg_loss = total_loss / num_batches
        return avg_loss

    def _validate_epoch(self) -> tuple:
        """Run one validation epoch.

        Returns:
            (avg_val_loss, all_y_true, all_y_pred)
        """
        self.model.eval()
        total_loss = 0.0
        all_y_true = []
        all_y_pred = []

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, targets)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)

                all_y_true.extend(targets.cpu().numpy().tolist())
                all_y_pred.extend(preds.cpu().numpy().tolist())

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, all_y_true, all_y_pred
