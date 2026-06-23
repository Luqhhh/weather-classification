"""
Data Transforms for Weather Image Classification

Key design considerations:
- Weather classification depends on color, lighting, and sky texture
- Augmentation must NOT be too aggressive — it can destroy weather semantics
- Train/val/test transforms must be consistently controllable
- All images are resized to a uniform size before feeding to the model
"""

import logging
import random
from typing import Dict, Optional, Tuple

import torch
from torchvision import transforms

logger = logging.getLogger(__name__)


class RandomMultiCutout:
    """Mask several small square-ish regions after tensor normalization.

    This is intentionally milder than CutMix: labels stay unchanged and the
    total occluded area is capped to preserve weather-scene semantics.
    """

    def __init__(
        self,
        p: float = 0.5,
        holes: int = 4,
        max_area: float = 0.02,
        fill: float | Tuple[float, float, float] = 0.0,
    ):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"cutout p must be in [0, 1], got {p}")
        if holes < 1:
            raise ValueError(f"cutout holes must be >= 1, got {holes}")
        if not 0.0 < max_area < 1.0:
            raise ValueError(f"cutout max_area must be in (0, 1), got {max_area}")
        self.p = p
        self.holes = holes
        self.max_area = max_area
        self.fill = fill

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        if tensor.dim() != 3:
            raise ValueError(f"RandomMultiCutout expects CHW tensor, got {tuple(tensor.shape)}")

        _, height, width = tensor.shape
        out = tensor.clone()
        fill = self._fill_tensor(tensor)

        for _ in range(self.holes):
            area = random.uniform(self.max_area * 0.25, self.max_area) * height * width
            side = max(1, int(area ** 0.5))
            cut_h = min(height, side)
            cut_w = min(width, side)
            top = random.randint(0, height - cut_h)
            left = random.randint(0, width - cut_w)
            out[:, top:top + cut_h, left:left + cut_w] = fill

        return out

    def _fill_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(self.fill, tuple):
            return torch.tensor(self.fill, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
        return torch.tensor(float(self.fill), dtype=tensor.dtype, device=tensor.device)


def get_train_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    augmentation: Optional[Dict] = None,
) -> transforms.Compose:
    """Build training transforms with augmentation.

    Augmentation is conservative to preserve weather semantics:
    - ColorJitter with small ranges (weather depends on color/lighting)
    - Mild RandomRotation (sky orientation matters less, but extreme rotation confuses)
    - RandomHorizontalFlip (safe for weather)
    - RandomResizedCrop (helps with varied image sizes)

    Args:
        image_size: Target square image size.
        mean: Normalization mean (ImageNet default).
        std: Normalization std (ImageNet default).
        augmentation: Optional dict to override default augmentation params.

    Returns:
        Composed transforms for training.
    """
    aug = augmentation or {}

    transform_list = []

    # --- Spatial transforms ---
    # RandomResizedCrop: helps with varied image sizes, provides scale augmentation
    rrc_config = aug.get("random_resized_crop", {"scale": (0.7, 1.0), "ratio": (0.9, 1.1)})
    transform_list.append(
        transforms.RandomResizedCrop(
            image_size,
            scale=rrc_config.get("scale", (0.7, 1.0)),
            ratio=rrc_config.get("ratio", (0.9, 1.1)),
        )
    )

    # Horizontal flip: safe for all weather types
    hflip_prob = aug.get("random_horizontal_flip", {}).get("prob", 0.5)
    transform_list.append(transforms.RandomHorizontalFlip(p=hflip_prob))

    # Mild rotation: weather photos can have slight tilt
    rotation_degrees = aug.get("random_rotation", {}).get("degrees", 10)
    if rotation_degrees:
        transform_list.append(transforms.RandomRotation(degrees=rotation_degrees))

    # --- Color transforms (conservative!) ---
    cj_config = aug.get("color_jitter", {
        "brightness": 0.15,
        "contrast": 0.15,
        "saturation": 0.15,
        "hue": 0.05,
    })
    if cj_config:
        transform_list.append(
            transforms.ColorJitter(
                brightness=cj_config.get("brightness", 0.15),
                contrast=cj_config.get("contrast", 0.15),
                saturation=cj_config.get("saturation", 0.15),
                hue=cj_config.get("hue", 0.05),
            )
        )

    # --- RandAugment (optional, replaces / supplements manual augmentations) ---
    # RandAugment randomly selects N augmentation ops and applies each at the
    # given magnitude. Stronger regularization than hand-picked jitter+rotation
    # — useful when the training set is small or classes are visually similar.
    rand_aug_config = aug.get("rand_augment")
    if rand_aug_config:
        num_ops = rand_aug_config.get("num_ops", 2)
        magnitude = rand_aug_config.get("magnitude", 9)
        transform_list.append(
            transforms.RandAugment(num_ops=num_ops, magnitude=magnitude)
        )

    # --- Tensor conversion & normalization ---
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=mean, std=std))

    # --- Optional: RandomErasing (small patches only) ---
    if aug.get("random_erasing_prob", 0) > 0:
        transform_list.append(
            transforms.RandomErasing(p=aug["random_erasing_prob"], scale=(0.02, 0.08))
        )

    cutout_cfg = aug.get("cutout")
    if cutout_cfg:
        transform_list.append(
            RandomMultiCutout(
                p=float(cutout_cfg.get("prob", 0.5)),
                holes=int(cutout_cfg.get("holes", 4)),
                max_area=float(cutout_cfg.get("max_area", 0.02)),
                fill=cutout_cfg.get("fill", 0.0),
            )
        )

    return transforms.Compose(transform_list)


def get_val_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Build validation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_test_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Build test/inference transforms (identical to val)."""
    return get_val_transforms(image_size=image_size, mean=mean, std=std)


def build_transforms(
    mode: str = "train",
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    augmentation: Optional[Dict] = None,
) -> transforms.Compose:
    """Factory function for building transforms by mode.

    Args:
        mode: One of 'train', 'val', 'test', 'inference'.
        image_size: Target square image size.
        mean: Normalization mean.
        std: Normalization std.
        augmentation: Augmentation config (train mode only).

    Returns:
        Composed transforms for the specified mode.
    """
    mode = mode.lower()
    if mode == "train":
        return get_train_transforms(image_size, mean, std, augmentation)
    elif mode in ("val", "valid", "validation"):
        return get_val_transforms(image_size, mean, std)
    elif mode in ("test", "inference", "submit"):
        return get_test_transforms(image_size, mean, std)
    else:
        raise ValueError(f"Unknown transform mode: {mode}. Use 'train', 'val', or 'test'.")
