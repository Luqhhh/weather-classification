import numpy as np
import torch
from PIL import Image

from scripts.evaluate_logits_postprocess import (
    _build_hflip_views,
    _fit_temperature_grid,
    _greedy_search_class_bias,
    _weighted_logits,
)


def test_weighted_logits_applies_member_temperatures_and_weights():
    logits = np.array(
        [
            [[2.0, 0.0], [0.0, 2.0]],
            [[1.0, 3.0], [3.0, 1.0]],
        ],
        dtype=np.float32,
    )

    combined = _weighted_logits(logits, weights=np.array([0.75, 0.25]), temperatures=np.array([1.0, 2.0]))

    expected = 0.75 * logits[0] + 0.25 * (logits[1] / 2.0)
    np.testing.assert_allclose(combined, expected)


def test_fit_temperature_grid_selects_lower_temperature_for_correct_confident_logits():
    logits = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)

    best = _fit_temperature_grid(logits, labels, candidates=np.array([0.5, 1.0, 2.0]))

    assert best == 0.5


def test_greedy_search_class_bias_can_recover_underpredicted_class():
    logits = np.array(
        [
            [0.2, 0.0],
            [0.2, 0.0],
            [0.2, 0.0],
            [0.2, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 1, 1, 1], dtype=np.int64)

    bias, score = _greedy_search_class_bias(
        logits,
        labels,
        num_classes=2,
        candidates=np.array([-0.5, 0.0, 0.5]),
        rounds=2,
    )

    assert bias[1] > bias[0]
    assert score > 0.3


def test_build_hflip_views_returns_original_and_horizontal_flip():
    image = Image.new("RGB", (2, 1))
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((1, 0), (0, 0, 255))

    original, flipped = _build_hflip_views(
        image_size=2,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    original_tensor = original(image)
    flipped_tensor = flipped(image)

    assert torch.argmax(original_tensor[:, 0, 0]).item() == 0
    assert torch.argmax(flipped_tensor[:, 0, 0]).item() == 2
