"""
Weather Prediction Pipeline

Handles batch inference on CPU with:
- Efficient batching
- Progress tracking
- CSV output in competition format
- Time estimation
"""

import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.label_mapping import LabelMapper
from data.transforms import build_transforms
from models.base import WeatherClassifier

logger = logging.getLogger(__name__)


class InferenceDataset(Dataset):
    """Dataset for inference (no labels needed)."""

    def __init__(self, image_paths: List[Path], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, str(path)


class WeatherPredictor:
    """Handles inference for weather image classification.

    Optimized for CPU inference with batching and optional ONNX runtime.
    """

    def __init__(
        self,
        model: WeatherClassifier,
        label_mapper: LabelMapper,
        image_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        device: str = "cpu",
        batch_size: int = 32,
        num_workers: int = 2,
    ):
        """
        Args:
            model: Trained WeatherClassifier.
            label_mapper: LabelMapper for class names.
            image_size: Input image size.
            mean: Normalization mean.
            std: Normalization std.
            device: 'cpu' or 'cuda'.
            batch_size: Inference batch size.
            num_workers: DataLoader workers.
        """
        self.model = model.to(device)
        self.model.eval()
        self.label_mapper = label_mapper
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = build_transforms(
            mode="inference",
            image_size=image_size,
            mean=mean,
            std=std,
        )

    def predict_file(self, image_path: Union[str, Path]) -> Dict:
        """Predict a single image file.

        Returns:
            Dict with 'predicted_class', 'class_index', 'probabilities'.
        """
        image_path = Path(image_path)
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()

        return {
            "filename": image_path.name,
            "predicted_class": self.label_mapper.decode(pred_idx),
            "class_index": pred_idx,
            "probabilities": {
                self.label_mapper.decode(i): round(probs[0, i].item(), 4)
                for i in range(self.label_mapper.num_classes)
            },
        }

    def predict_batch(
        self,
        image_dir: Union[str, Path],
        output_csv: Optional[Union[str, Path]] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
        show_progress: bool = True,
    ) -> List[Dict]:
        """Predict all images in a directory.

        Args:
            image_dir: Directory containing images.
            output_csv: If provided, save predictions to CSV.
            extensions: Allowed image file extensions.
            show_progress: Show tqdm progress bar.

        Returns:
            List of prediction dicts with filename, predicted_class, class_index.
        """
        image_dir = Path(image_dir)
        image_paths = sorted(
            [p for p in image_dir.iterdir()
             if p.suffix.lower() in extensions and p.is_file()]
        )

        if not image_paths:
            logger.warning(f"No images found in {image_dir}")
            return []

        logger.info(f"Predicting {len(image_paths)} images from {image_dir}")

        dataset = InferenceDataset(image_paths, self.transform)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        results = []
        start_time = time.time()

        iterator = loader
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(loader, desc="Inference", unit="batch")

        with torch.no_grad():
            for images, paths in iterator:
                images = images.to(self.device)
                logits = self.model(images)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                for path, pred_idx, prob in zip(paths, preds, probs):
                    results.append({
                        "filename": Path(path).name,
                        "predicted_class": self.label_mapper.decode(pred_idx.item()),
                        "class_index": pred_idx.item(),
                        "confidence": round(prob[pred_idx].item(), 4),
                    })

        elapsed = time.time() - start_time
        avg_time = elapsed / len(image_paths) * 1000 if image_paths else 0
        logger.info(
            f"Prediction complete: {len(results)} images in {elapsed:.1f}s "
            f"({avg_time:.1f}ms/image)"
        )

        # Save to CSV
        if output_csv:
            self._save_csv(results, output_csv)

        return results

    def _save_csv(self, results: List[Dict], output_path: Union[str, Path]) -> None:
        """Save predictions to CSV in competition format.

        Expected format: filename,prediction
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "prediction"])
            for r in results:
                writer.writerow([r["filename"], r["predicted_class"]])

        logger.info(f"Predictions saved to {output_path}")

    def estimate_total_time(self, num_images: int) -> Dict[str, float]:
        """Estimate total inference time for a given number of images.

        Uses a small warmup run to measure per-image latency.

        Args:
            num_images: Expected number of scoring images.

        Returns:
            Dict with estimated times and feasibility check.
        """
        # Run a quick benchmark on a few dummy images
        dummy_input = torch.randn(self.batch_size, 3, 224, 224).to(self.device)

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = self.model(dummy_input)

        # Timed run
        times = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model(dummy_input)
            times.append(time.perf_counter() - start)

        avg_batch_time = np.mean(times)
        avg_image_time = avg_batch_time / self.batch_size

        num_batches = (num_images + self.batch_size - 1) // self.batch_size
        estimated_model_time = num_batches * avg_batch_time
        estimated_io_time = num_images * 0.005  # Rough I/O estimate: 5ms per image
        total_estimate = estimated_model_time + estimated_io_time

        return {
            "num_images": num_images,
            "batch_size": self.batch_size,
            "num_batches": num_batches,
            "avg_batch_time_ms": round(avg_batch_time * 1000, 1),
            "avg_image_time_ms": round(avg_image_time * 1000, 1),
            "estimated_model_time_min": round(estimated_model_time / 60, 1),
            "estimated_io_time_min": round(estimated_io_time / 60, 1),
            "estimated_total_time_min": round(total_estimate / 60, 1),
            "within_70min_limit": total_estimate < (70 * 60),
        }
