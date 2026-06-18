"""
Tests for summarize_experiments.py — experiment results CSV generation.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.summarize_experiments import (
    _find_experiment_dirs,
    _read_epochs,
    _extract_per_class_f1,
    _summarize_one_experiment,
    main,
    CSV_COLUMNS,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, epochs: list[dict]) -> None:
    """Write a list of epoch dicts as a JSON Lines file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in epochs:
            f.write(json.dumps(e) + "\n")


def _make_epoch(epoch: int, macro_f1: float, **overrides) -> dict:
    """Build a minimal epoch record."""
    rec = {
        "epoch": epoch,
        "timestamp": 1700000000.0 + epoch * 100,
        "macro_f1": macro_f1,
        "accuracy": 0.85 + macro_f1 * 0.01,
        "val_loss": 0.5 - macro_f1 * 0.1,
        "train_loss": 0.4 - macro_f1 * 0.1,
        "lr": 0.0001,
        "per_class": {
            "cloudy": {"precision": 0.80, "recall": 0.80, "f1": 0.80, "support": 100},
            "rainy": {"precision": 0.75, "recall": 0.70, "f1": 0.72, "support": 50},
            "snowy": {"precision": 0.85, "recall": 0.85, "f1": 0.85, "support": 50},
            "sunny": {"precision": 0.90, "recall": 0.90, "f1": 0.90, "support": 100},
        },
    }
    rec.update(overrides)
    return rec


def _read_csv_rows(path: Path) -> list[dict]:
    """Read a CSV file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# unit tests
# ---------------------------------------------------------------------------

class TestExtractPerClassF1:
    """Tests for _extract_per_class_f1()."""

    def test_returns_f1_for_existing_class(self):
        per_class = {"sunny": {"f1": 0.899}}
        result = _extract_per_class_f1(per_class, "sunny")
        assert result == "0.899"

    def test_returns_empty_for_missing_class(self):
        per_class = {"cloudy": {"f1": 0.85}}
        result = _extract_per_class_f1(per_class, "sunny")
        assert result == ""

    def test_returns_empty_when_per_class_is_none(self):
        result = _extract_per_class_f1(None, "sunny")
        assert result == ""

    def test_returns_empty_when_class_entry_is_malformed(self):
        result = _extract_per_class_f1({"sunny": "not_a_dict"}, "sunny")
        assert result == ""


class TestReadEpochs:
    """Tests for _read_epochs()."""

    def test_reads_valid_jsonl(self, tmp_path):
        log = tmp_path / "training_log.jsonl"
        _write_jsonl(log, [
            _make_epoch(1, 0.80),
            _make_epoch(2, 0.85),
        ])
        epochs = _read_epochs(log)
        assert len(epochs) == 2
        assert epochs[0]["epoch"] == 1

    def test_skips_malformed_lines(self, tmp_path):
        log = tmp_path / "training_log.jsonl"
        with open(log, "w") as f:
            f.write('not valid json\n')
            f.write(json.dumps(_make_epoch(1, 0.80)) + "\n")
        epochs = _read_epochs(log)
        assert len(epochs) == 1
        assert epochs[0]["epoch"] == 1

    def test_empty_file_returns_empty_list(self, tmp_path):
        log = tmp_path / "training_log.jsonl"
        log.write_text("")
        assert _read_epochs(log) == []


class TestSummarizeOneExperiment:
    """Tests for _summarize_one_experiment()."""

    def test_picks_best_epoch_by_macro_f1(self, tmp_path):
        exp_dir = tmp_path / "exp_001"
        _write_jsonl(exp_dir / "training_log.jsonl", [
            _make_epoch(1, 0.82),
            _make_epoch(2, 0.88),
            _make_epoch(3, 0.85),
        ])
        row = _summarize_one_experiment(exp_dir)
        assert row is not None
        assert row["experiment_name"] == "exp_001"
        assert row["best_epoch"] == "2"
        assert row["val_macro_f1"] == "0.88"
        assert row["total_epochs"] == "3"

    def test_f1_sunny_comes_from_per_class_sunny(self, tmp_path):
        exp_dir = tmp_path / "exp_test"
        _write_jsonl(exp_dir / "training_log.jsonl", [
            _make_epoch(1, 0.80, per_class={
                "cloudy": {"f1": 0.80},
                "rainy":  {"f1": 0.72},
                "snowy":  {"f1": 0.85},
                "sunny":  {"f1": 0.999},
            }),
        ])
        row = _summarize_one_experiment(exp_dir)
        assert row["f1_sunny"] == "0.999"

    def test_missing_per_class_entry_is_empty(self, tmp_path):
        exp_dir = tmp_path / "exp_partial"
        _write_jsonl(exp_dir / "training_log.jsonl", [
            _make_epoch(1, 0.75, per_class={
                "cloudy": {"f1": 0.80},
                "rainy":  {"f1": 0.72},
            }),
        ])
        row = _summarize_one_experiment(exp_dir)
        assert row["f1_snowy"] == ""
        assert row["f1_sunny"] == ""


class TestBuildExperimentDirs:
    """Tests for _find_experiment_dirs()."""

    def test_finds_dirs_with_training_log(self, tmp_path):
        (tmp_path / "exp_a").mkdir()
        _write_jsonl(tmp_path / "exp_a" / "training_log.jsonl", [_make_epoch(1, 0.8)])
        (tmp_path / "not_an_exp").mkdir()  # no training_log.jsonl
        (tmp_path / "random_file").write_text("hello")

        exp_dirs = _find_experiment_dirs(tmp_path)
        assert len(exp_dirs) == 1
        assert exp_dirs[0].name == "exp_a"

    def test_empty_outputs_dir(self, tmp_path):
        assert _find_experiment_dirs(tmp_path) == []

    def test_missing_outputs_dir(self, tmp_path):
        assert _find_experiment_dirs(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# integration tests (via main)
# ---------------------------------------------------------------------------

class TestMainIntegration:
    """End-to-end tests exercising main() with --outputs_dir and --output."""

    def _run_main(self, outputs_dir: Path, output_csv: Path) -> None:
        """Simulate argparse + main()."""
        import argparse
        # Bypass argparse by injecting a simple namespace.
        # But main() parses sys.argv — easier: just call it after monkeypatching.
        # Simpler: call main directly by patching sys.argv.
        import scripts.summarize_experiments as mod
        old_argv = sys.argv
        try:
            sys.argv = [
                "summarize_experiments.py",
                "--outputs_dir", str(outputs_dir),
                "--output", str(output_csv),
            ]
            mod.main()
        finally:
            sys.argv = old_argv

    def test_two_experiments_sorted_descending(self, tmp_path):
        outputs_dir = tmp_path / "outputs"
        # exp_a: best F1 = 0.88
        _write_jsonl(outputs_dir / "exp_a" / "training_log.jsonl", [
            _make_epoch(1, 0.82),
            _make_epoch(2, 0.88),
        ])
        # exp_b: best F1 = 0.91
        _write_jsonl(outputs_dir / "exp_b" / "training_log.jsonl", [
            _make_epoch(1, 0.90),
            _make_epoch(2, 0.91),
            _make_epoch(3, 0.89),
        ])
        csv_path = tmp_path / "results.csv"

        self._run_main(outputs_dir, csv_path)

        rows = _read_csv_rows(csv_path)
        assert len(rows) == 2
        # Sorted descending: exp_b first, then exp_a
        assert rows[0]["experiment_name"] == "exp_b"
        assert rows[0]["val_macro_f1"] == "0.91"
        assert rows[0]["total_epochs"] == "3"
        assert rows[1]["experiment_name"] == "exp_a"
        assert rows[1]["val_macro_f1"] == "0.88"
        assert rows[1]["total_epochs"] == "2"

    def test_skips_dir_without_training_log(self, tmp_path):
        outputs_dir = tmp_path / "outputs"
        _write_jsonl(outputs_dir / "exp_good" / "training_log.jsonl", [
            _make_epoch(1, 0.8),
        ])
        (outputs_dir / "exp_empty").mkdir(parents=True)  # no jsonl
        csv_path = tmp_path / "results.csv"

        self._run_main(outputs_dir, csv_path)

        rows = _read_csv_rows(csv_path)
        assert len(rows) == 1
        assert rows[0]["experiment_name"] == "exp_good"

    def test_empty_outputs_produces_header_only(self, tmp_path):
        outputs_dir = tmp_path / "empty_outputs"
        outputs_dir.mkdir()
        csv_path = tmp_path / "results.csv"

        self._run_main(outputs_dir, csv_path)

        rows = _read_csv_rows(csv_path)
        assert len(rows) == 0
        # Verify the header row is present
        with open(csv_path, "r") as f:
            header = f.readline().strip()
        assert header == ",".join(CSV_COLUMNS)

    def test_per_class_missing_class_does_not_crash(self, tmp_path):
        outputs_dir = tmp_path / "outputs"
        _write_jsonl(outputs_dir / "exp_partial" / "training_log.jsonl", [
            _make_epoch(1, 0.75, per_class={
                "cloudy": {"f1": 0.80},
                "rainy": {"f1": 0.72},
            }),
        ])
        csv_path = tmp_path / "results.csv"

        self._run_main(outputs_dir, csv_path)

        rows = _read_csv_rows(csv_path)
        assert len(rows) == 1
        assert rows[0]["f1_cloudy"] == "0.8"
        assert rows[0]["f1_rainy"] == "0.72"
        assert rows[0]["f1_snowy"] == ""
        assert rows[0]["f1_sunny"] == ""
