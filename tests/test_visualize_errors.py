"""
Tests for visualize_errors.py — error sample grid generation.
"""

import csv
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visualize_errors import _load_error_samples, _build_grid, _render_grid


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Helper: write a minimal error_samples.csv."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "true_label", "predicted_label", "confidence"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _make_dummy_image(path: Path, size: int = 32) -> None:
    """Helper: create a solid-colour RGB image."""
    img = Image.new("RGB", (size, size), color=(100, 150, 200))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


class TestLoadErrorSamples:
    """Tests for _load_error_samples()."""

    def test_loads_valid_csv(self, tmp_path):
        csv_path = tmp_path / "errors.csv"
        _write_csv(csv_path, [
            {"filename": "cloudy/a.jpg", "true_label": "cloudy",
             "predicted_label": "rainy", "confidence": "0.85"},
        ])
        samples = _load_error_samples(csv_path)
        assert len(samples) == 1
        assert samples[0]["filename"] == "cloudy/a.jpg"

    def test_empty_csv_returns_empty_list(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        _write_csv(csv_path, [])
        samples = _load_error_samples(csv_path)
        assert samples == []

    def test_missing_csv_exits(self, tmp_path):
        csv_path = tmp_path / "not_found.csv"
        with pytest.raises(SystemExit):
            _load_error_samples(csv_path)


class TestBuildGrid:
    """Tests for _build_grid()."""

    def test_loads_images_and_truncates_to_max(self, tmp_path):
        data_dir = tmp_path / "images"
        _make_dummy_image(data_dir / "cloudy/a.jpg")
        _make_dummy_image(data_dir / "rainy/b.jpg")
        samples = [
            {"filename": "cloudy/a.jpg", "true_label": "cloudy",
             "predicted_label": "rainy", "confidence": "0.90"},
            {"filename": "rainy/b.jpg", "true_label": "rainy",
             "predicted_label": "snowy", "confidence": "0.60"},
        ]
        cells = _build_grid(samples, data_dir=data_dir, max_samples=1,
                            cols=4, thumbnail_size=32)
        assert len(cells) == 1
        assert cells[0]["true_label"] == "cloudy"

    def test_skips_missing_image(self, tmp_path):
        data_dir = tmp_path / "images"
        _make_dummy_image(data_dir / "cloudy/a.jpg")
        samples = [
            {"filename": "cloudy/a.jpg", "true_label": "cloudy",
             "predicted_label": "rainy", "confidence": "0.90"},
            {"filename": "missing/m.jpg", "true_label": "snowy",
             "predicted_label": "sunny", "confidence": "0.55"},
        ]
        cells = _build_grid(samples, data_dir=data_dir, max_samples=4,
                            cols=4, thumbnail_size=32)
        # Only the first sample loads; the missing one is skipped.
        assert len(cells) == 1
        assert cells[0]["true_label"] == "cloudy"

    def test_all_missing_returns_empty(self, tmp_path):
        data_dir = tmp_path / "images"
        samples = [
            {"filename": "missing/a.jpg", "true_label": "cloudy",
             "predicted_label": "rainy", "confidence": "0.90"},
        ]
        cells = _build_grid(samples, data_dir=data_dir, max_samples=4,
                            cols=4, thumbnail_size=32)
        assert cells == []


class TestRenderGrid:
    """Tests for _render_grid()."""

    def test_renders_png_non_empty(self, tmp_path):
        # Create a single dummy cell
        img = Image.new("RGB", (32, 32), color=(255, 0, 0))
        cells = [{
            "img": img,
            "true_label": "cloudy",
            "predicted_label": "rainy",
            "confidence": 0.82,
        }]
        fig = _render_grid(cells, cols=4, thumbnail_size=32)
        assert fig is not None

        out = tmp_path / "grid.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        assert out.stat().st_size > 0

    def test_renders_multiple_cells(self, tmp_path):
        cells = []
        for i in range(5):
            img = Image.new("RGB", (32, 32), color=(i * 40, 100, 200))
            cells.append({
                "img": img,
                "true_label": "cloudy",
                "predicted_label": "rainy",
                "confidence": 0.5 + i * 0.1,
            })
        fig = _render_grid(cells, cols=3, thumbnail_size=32)
        assert fig is not None

        out = tmp_path / "grid.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        assert out.stat().st_size > 0

    def test_empty_cells_returns_none(self):
        fig = _render_grid([], cols=4, thumbnail_size=32)
        assert fig is None
