"""
Tests for loss function creation and weighted focal behavior.
"""

import torch

from training.losses import FocalLoss, create_loss_function


def test_create_cross_entropy_with_class_weights():
    loss_fn = create_loss_function(
        name="cross_entropy",
        num_classes=4,
        class_weights=[0.64, 2.31, 2.71, 0.61],
    )

    assert torch.allclose(
        loss_fn.weight,
        torch.tensor([0.64, 2.31, 2.71, 0.61], dtype=torch.float32),
    )


def test_create_focal_uses_gamma_and_class_weights():
    loss_fn = create_loss_function(
        name="focal",
        num_classes=4,
        class_weights=[0.64, 2.31, 2.71, 0.61],
        focal_gamma=1.5,
    )

    assert isinstance(loss_fn, FocalLoss)
    assert loss_fn.gamma == 1.5
    assert torch.allclose(
        loss_fn.alpha,
        torch.tensor([0.64, 2.31, 2.71, 0.61], dtype=torch.float32),
    )
    assert "alpha" in dict(loss_fn.named_buffers())


def test_focal_alpha_moves_with_module_device():
    loss_fn = create_loss_function(
        name="focal",
        num_classes=4,
        class_weights=[1.0, 2.0, 3.0, 4.0],
    )

    moved = loss_fn.to("cpu")

    assert moved.alpha.device.type == "cpu"
