"""
Tests for experiment_tracking — git_utils, tracker, aggregator, and
the aggregate_experiments.py CLI.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_tracking.git_utils import capture_git_metadata
from experiment_tracking.tracker import (
    CSV_COLUMNS,
    ExperimentResult,
    ExperimentTracker,
)
from experiment_tracking.aggregator import (
    ExperimentAggregator,
    _safe_float,
    _fmt_f1,
    _AUTO_MARKER,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> dict:
    """Minimal training config."""
    cfg = {
        "model": {"name": "resnet18", "dropout": 0.3},
        "data": {"image_size": 224, "augmentation": {"scale": [0.8, 1.0]}},
        "training": {
            "batch_size": 64,
            "loss": {"name": "cross_entropy"},
        },
        "seed": 42,
    }
    cfg.update(overrides)
    return cfg


def _make_evaluation(macro_f1=0.87, per_class=None) -> dict:
    """Minimal evaluation metrics dict (as returned by compute_metrics)."""
    if per_class is None:
        per_class = {
            "cloudy": {"precision": 0.85, "recall": 0.89, "f1": 0.8677, "support": 100},
            "rainy":  {"precision": 0.91, "recall": 0.75, "f1": 0.8240, "support": 50},
            "snowy":  {"precision": 0.91, "recall": 0.87, "f1": 0.8927, "support": 50},
            "sunny":  {"precision": 0.90, "recall": 0.90, "f1": 0.8990, "support": 100},
        }
    return {
        "macro_f1": macro_f1,
        "accuracy": 0.8784,
        "per_class": per_class,
        "confusion_matrix": [[0, 0], [0, 0]],
        "weak_classes": ["cloudy", "rainy"],
        "avg_per_class_f1": 0.87,
    }


def _make_benchmark() -> dict:
    """Minimal benchmark result (as returned by CpuBenchmark.run)."""
    return {
        "model_name": "resnet18",
        "params_millions": 11.18,
        "weight_size_mb": 42.6,
        "optimal_batch_size": 8,
        "batch_results": {
            "8": {"per_image_mean_ms": 5.38, "throughput_imgs_per_sec": 185.7},
        },
        "scoring_estimate": {
            "total_time_min": 0.5,
            "within_70min": True,
        },
    }


def _write_results_json(path: Path, data: dict) -> None:
    """Write a results.json file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# git_utils
# ---------------------------------------------------------------------------

class TestCaptureGitMetadata:
    """Tests for capture_git_metadata()."""

    def test_returns_dict_in_repo(self):
        """In this project repo, we should get valid git metadata."""
        meta = capture_git_metadata()
        assert isinstance(meta, dict)
        assert set(meta.keys()) == {
            "branch", "commit_hash", "commit_message", "dirty", "dirty_files",
        }
        # We're in a git repo → should have a commit
        assert meta["commit_hash"] is not None
        assert len(meta["commit_hash"]) >= 7
        assert meta["branch"] is not None

    def test_returns_none_for_non_repo(self, tmp_path):
        """Outside a git repo, all values should be None."""
        meta = capture_git_metadata(repo_root=str(tmp_path))
        assert meta["branch"] is None
        assert meta["commit_hash"] is None
        assert meta["dirty"] is None


# ---------------------------------------------------------------------------
# tracker — ExperimentResult
# ---------------------------------------------------------------------------

class TestExperimentResult:
    """Tests for ExperimentResult dataclass."""

    def test_flatten_for_csv(self):
        r = ExperimentResult(
            experiment_id="exp_001",
            model="resnet18",
            val_macro_f1="0.8708",
            cloudy_f1="0.8677",
        )
        flat = r.flatten_for_csv(CSV_COLUMNS)
        assert flat["experiment_id"] == "exp_001"
        assert flat["model"] == "resnet18"
        assert flat["val_macro_f1"] == "0.8708"
        assert flat["cloudy_f1"] == "0.8677"
        # Unset fields → empty string
        assert flat["branch"] == ""
        assert flat["notes"] == ""

    def test_roundtrip_to_dict(self):
        r = ExperimentResult(
            experiment_id="exp_002",
            model="mobilenetv3_small",
            val_macro_f1="0.82",
        )
        data = r.to_dict()
        r2 = ExperimentResult.from_dict(data)
        assert r2.experiment_id == "exp_002"
        assert r2.model == "mobilenetv3_small"
        assert r2.val_macro_f1 == "0.82"


# ---------------------------------------------------------------------------
# tracker — ExperimentTracker
# ---------------------------------------------------------------------------

class TestExperimentTracker:
    """Tests for ExperimentTracker."""

    def test_build_result_basic(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "exp_001")
        result = tracker.build_result(
            config=_make_config(),
            evaluation_metrics=_make_evaluation(),
            notes="Baseline run",
            experiment_id="exp_001",
        )
        assert result.experiment_id == "exp_001"
        assert result.model == "resnet18"
        assert result.image_size == "224"
        assert result.loss == "cross_entropy"
        assert result.augmentation == "default"
        assert result.dropout == "0.3"
        assert result.batch_size == "64"
        assert result.val_macro_f1 == "0.8700"
        assert result.cloudy_f1 == "0.8677"
        assert result.rainy_f1 == "0.8240"
        assert result.snowy_f1 == "0.8927"
        assert result.sunny_f1 == "0.8990"
        assert result.notes == "Baseline run"
        assert result.commit_hash is not None  # git info captured

    def test_build_result_no_augmentation(self, tmp_path):
        config = _make_config()
        config["data"]["augmentation"] = {
            "scale": [1.0, 1.0],
            "horizontal_flip_prob": 0,
            "rotation_degrees": 0,
            "color_jitter": {},
        }
        tracker = ExperimentTracker(tmp_path / "exp_none")
        result = tracker.build_result(config=config, experiment_id="exp_none")
        assert result.augmentation == "none"

    def test_build_result_with_benchmark(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "exp_full")
        result = tracker.build_result(
            config=_make_config(),
            evaluation_metrics=_make_evaluation(),
            benchmark_results=_make_benchmark(),
            submit_check_results={"all_passed": True},
            experiment_id="exp_full",
            notes="Full pipeline",
        )
        assert result.status == "complete"
        assert result.cpu_time_per_image == "5.38"
        assert result.model_size_mb == "42.6"
        assert result.submit_check_passed == "True"

    def test_save_and_load(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "exp_save")
        result = tracker.build_result(
            config=_make_config(),
            evaluation_metrics=_make_evaluation(),
            experiment_id="exp_save",
        )
        saved_path = tracker.save(result)
        assert saved_path.is_file()

        loaded = tracker.load()
        assert loaded is not None
        assert loaded.experiment_id == "exp_save"
        assert loaded.val_macro_f1 == "0.8700"

    def test_get_next_experiment_id_empty(self, tmp_path):
        tracker = ExperimentTracker(tmp_path)
        eid = tracker.get_next_experiment_id(experiments_dir=tmp_path / "experiments")
        assert eid == "exp_001"

    def test_get_next_experiment_id_increments(self, tmp_path):
        exp_dir = tmp_path / "experiments"
        exp_dir.mkdir(parents=True)
        # Write a CSV with existing exp_001 and exp_002
        csv_path = exp_dir / "results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["experiment_id", "val_macro_f1"])
            writer.writeheader()
            writer.writerows([
                {"experiment_id": "exp_001", "val_macro_f1": "0.88"},
                {"experiment_id": "exp_002", "val_macro_f1": "0.86"},
            ])

        tracker = ExperimentTracker(tmp_path)
        eid = tracker.get_next_experiment_id(experiments_dir=exp_dir)
        assert eid == "exp_003"

    def test_extract_config_fields_custom_loss(self, tmp_path):
        config = _make_config()
        config["training"]["loss"] = {"name": "focal", "focal_gamma": 2.0}
        tracker = ExperimentTracker(tmp_path / "exp_focal")
        result = tracker.build_result(config=config, experiment_id="exp_focal")
        assert result.loss == "focal"


# ---------------------------------------------------------------------------
# aggregator
# ---------------------------------------------------------------------------

class TestAggregator:
    """Tests for ExperimentAggregator."""

    def test_scan_discovers_results_json(self, tmp_path):
        outputs = tmp_path / "outputs"
        _write_results_json(outputs / "exp_a" / "results.json", {
            "experiment_id": "exp_a",
            "model": "resnet18",
            "val_macro_f1": "0.88",
            "cloudy_f1": "0.86",
            "rainy_f1": "0.82",
            "snowy_f1": "0.89",
            "sunny_f1": "0.90",
        })
        _write_results_json(outputs / "exp_b" / "results.json", {
            "experiment_id": "exp_b",
            "model": "resnet34",
            "val_macro_f1": "0.91",
        })

        agg = ExperimentAggregator(outputs_dir=outputs, experiments_dir=tmp_path / "experiments")
        results = agg.scan()
        assert len(results) == 2
        # sorted desc by val_macro_f1
        assert results[0].experiment_id == "exp_b"
        assert results[1].experiment_id == "exp_a"

    def test_scan_skips_bad_json(self, tmp_path, caplog):
        outputs = tmp_path / "outputs"
        (outputs / "bad_exp").mkdir(parents=True)
        (outputs / "bad_exp" / "results.json").write_text("{not valid json")

        agg = ExperimentAggregator(outputs_dir=outputs, experiments_dir=tmp_path / "experiments")
        results = agg.scan()
        assert len(results) == 0

    def test_generate_csv(self, tmp_path):
        r1 = ExperimentResult(
            experiment_id="exp_001", model="resnet18",
            val_macro_f1="0.88", cloudy_f1="0.86", rainy_f1="0.82",
            snowy_f1="0.89", sunny_f1="0.90",
        )
        r2 = ExperimentResult(
            experiment_id="exp_002", model="resnet34",
            val_macro_f1="0.91", cloudy_f1="0.90",
        )
        agg = ExperimentAggregator(outputs_dir=tmp_path, experiments_dir=tmp_path)
        csv_str = agg.generate_csv([r2, r1])

        reader = csv.DictReader(csv_str.splitlines())
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["experiment_id"] == "exp_002"
        assert rows[0]["val_macro_f1"] == "0.91"
        assert rows[1]["experiment_id"] == "exp_001"
        # verify all 18 columns present
        assert reader.fieldnames == CSV_COLUMNS

    def test_generate_markdown(self, tmp_path):
        r = ExperimentResult(
            experiment_id="exp_001", model="resnet18", loss="cross_entropy",
            image_size="224", augmentation="default", dropout="0.3",
            batch_size="64", val_macro_f1="0.8708", cloudy_f1="0.8677",
            rainy_f1="0.8240", snowy_f1="0.8927", sunny_f1="0.8990",
            status="complete", branch="main", commit_hash="abc1234",
            cpu_time_per_image="5.38", model_size_mb="42.6",
            submit_check_passed="True", notes="Baseline",
            training={"best_epoch": 4, "total_epochs": 10, "early_stopped": True,
                       "training_time_min": 45.2, "best_val_macro_f1": 0.88},
        )
        agg = ExperimentAggregator(outputs_dir=tmp_path, experiments_dir=tmp_path)
        md = agg.generate_markdown([r], preserve_existing=False)

        assert _AUTO_MARKER in md
        assert "exp_001" in md
        assert "0.8708" in md
        assert "resnet18" in md
        assert "Baseline" in md
        assert "abc1234" in md
        assert "42.6" in md

    def test_preserve_existing_preamble(self, tmp_path):
        """Existing content above AUTO-GENERATED marker is preserved."""
        existing = "## 手动笔记\n\n这是手写内容。\n\n<!-- AUTO-GENERATED -->\n"
        exp_dir = tmp_path / "experiments"
        exp_dir.mkdir()
        (exp_dir / "leaderboard.md").write_text(existing, encoding="utf-8")

        r = ExperimentResult(experiment_id="exp_001", val_macro_f1="0.88")
        agg = ExperimentAggregator(outputs_dir=tmp_path, experiments_dir=exp_dir)
        md = agg.generate_markdown([r], preserve_existing=True)

        assert md.startswith("## 手动笔记")
        assert "这是手写内容" in md
        assert _AUTO_MARKER in md
        assert "exp_001" in md

    def test_write_csv(self, tmp_path):
        r = ExperimentResult(experiment_id="exp_001")
        agg = ExperimentAggregator(outputs_dir=tmp_path, experiments_dir=tmp_path / "experiments")
        csv_str = agg.generate_csv([r])
        csv_path = agg.write_csv(csv_str)
        assert csv_path.is_file()
        assert csv_path.name == "results.csv"

    def test_write_markdown(self, tmp_path):
        r = ExperimentResult(experiment_id="exp_001")
        agg = ExperimentAggregator(outputs_dir=tmp_path, experiments_dir=tmp_path / "experiments")
        md_str = agg.generate_markdown([r], preserve_existing=False)
        md_path = agg.write_markdown(md_str)
        assert md_path.is_file()
        assert md_path.name == "leaderboard.md"

    def test_empty_scan_produces_header_only_csv(self, tmp_path):
        agg = ExperimentAggregator(outputs_dir=tmp_path, experiments_dir=tmp_path / "experiments")
        csv_str = agg.generate_csv([])
        lines = csv_str.strip().split("\n")
        assert len(lines) == 1  # header only
        assert lines[0].startswith("experiment_id")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_safe_float(self):
        assert _safe_float("0.88") == 0.88
        assert _safe_float("") == 0.0
        assert _safe_float("not_a_number") == 0.0
        assert _safe_float("0") == 0.0

    def test_fmt_f1(self):
        assert _fmt_f1("0.8708") == "0.8708"
        assert _fmt_f1("") == "—"
        assert _fmt_f1("0") == "0.0000"


# ---------------------------------------------------------------------------
# CSV_COLUMNS completeness
# ---------------------------------------------------------------------------

def test_csv_columns_match_recommended_schema():
    """Verify that CSV_COLUMNS includes all 18 recommended fields."""
    assert CSV_COLUMNS == [
        "experiment_id",
        "branch",
        "commit_hash",
        "model",
        "image_size",
        "loss",
        "augmentation",
        "dropout",
        "batch_size",
        "val_macro_f1",
        "cloudy_f1",
        "rainy_f1",
        "snowy_f1",
        "sunny_f1",
        "cpu_time_per_image",
        "model_size_mb",
        "submit_check_passed",
        "notes",
    ]
