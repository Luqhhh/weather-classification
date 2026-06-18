#!/usr/bin/env python3
"""
Experiment Results Summarizer

Scans the outputs directory for experiment subdirectories containing
training_log.jsonl, extracts the best epoch from each, and produces
a single experiments/results.csv sorted by val_macro_f1 descending.

Usage:
    python scripts/summarize_experiments.py

Output:
    experiments/results.csv
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "experiment_name",
    "best_epoch",
    "val_macro_f1",
    "val_accuracy",
    "val_loss",
    "train_loss",
    "f1_cloudy",
    "f1_rainy",
    "f1_snowy",
    "f1_sunny",
    "total_epochs",
]

WEATHER_CLASSES = ["cloudy", "rainy", "snowy", "sunny"]


def _find_experiment_dirs(outputs_dir: Path) -> list[Path]:
    """Return subdirectories under outputs_dir that contain a training_log.jsonl."""
    if not outputs_dir.is_dir():
        logger.warning("Outputs directory not found: %s", outputs_dir)
        return []

    exp_dirs: list[Path] = []
    for child in sorted(outputs_dir.iterdir()):
        if not child.is_dir():
            continue
        log_path = child / "training_log.jsonl"
        if log_path.is_file():
            exp_dirs.append(child)
        else:
            logger.debug("Skipping %s — no training_log.jsonl", child.name)

    return exp_dirs


def _read_epochs(log_path: Path) -> list[dict]:
    """Read all JSON Lines from a training log file.

    Returns a list of epoch dicts.  Returns an empty list when the file
    cannot be read or is empty.
    """
    epochs: list[dict] = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    epochs.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON line in %s", log_path)
    except OSError as exc:
        logger.warning("Cannot read %s: %s", log_path, exc)

    return epochs


def _extract_per_class_f1(per_class: dict, class_name: str) -> str:
    """Safely extract per-class F1 from the nested dict.

    Returns the F1 value as a string, or an empty string when the class
    is missing or the dict is malformed.
    """
    if not isinstance(per_class, dict):
        return ""
    class_entry = per_class.get(class_name)
    if not isinstance(class_entry, dict):
        return ""
    f1_value = class_entry.get("f1")
    if f1_value is None:
        return ""
    return str(f1_value)


def _summarize_one_experiment(exp_dir: Path) -> dict | None:
    """Summarize a single experiment directory.

    Returns a dict of CSV columns, or None when the log is empty.
    """
    log_path = exp_dir / "training_log.jsonl"
    epochs = _read_epochs(log_path)

    if not epochs:
        logger.warning("No valid epochs in %s — skipping", log_path)
        return None

    # Best epoch = highest macro_f1
    best = max(epochs, key=lambda e: e.get("macro_f1", float("-inf")))
    per_class = best.get("per_class", {})

    return {
        "experiment_name": exp_dir.name,
        "best_epoch": str(best.get("epoch", "")),
        "val_macro_f1": str(best.get("macro_f1", "")),
        "val_accuracy": str(best.get("accuracy", "")),
        "val_loss": str(best.get("val_loss", "")),
        "train_loss": str(best.get("train_loss", "")),
        "f1_cloudy": _extract_per_class_f1(per_class, "cloudy"),
        "f1_rainy": _extract_per_class_f1(per_class, "rainy"),
        "f1_snowy": _extract_per_class_f1(per_class, "snowy"),
        "f1_sunny": _extract_per_class_f1(per_class, "sunny"),
        "total_epochs": str(len(epochs)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize experiment results into a CSV leaderboard"
    )
    parser.add_argument(
        "--outputs_dir", type=str, default="outputs",
        help="Directory containing experiment subdirectories (default: outputs)",
    )
    parser.add_argument(
        "--output", type=str, default="experiments/results.csv",
        help="Output CSV path (default: experiments/results.csv)",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    output_csv = Path(args.output)

    # 1. Discover experiments
    exp_dirs = _find_experiment_dirs(outputs_dir)
    if not exp_dirs:
        logger.info(
            "No experiment directories with training_log.jsonl found under %s",
            outputs_dir,
        )
        # Still produce a header-only CSV so the output path is consistent.
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
        logger.info("Wrote header-only CSV to %s", output_csv)
        return

    # 2. Summarize each experiment
    rows: list[dict] = []
    for exp_dir in exp_dirs:
        logger.info("Processing %s ...", exp_dir.name)
        row = _summarize_one_experiment(exp_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        logger.info("No valid experiments to include in results.")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
        logger.info("Wrote header-only CSV to %s", output_csv)
        return

    # 3. Sort by val_macro_f1 descending
    rows.sort(key=lambda r: float(r.get("val_macro_f1") or 0), reverse=True)

    # 4. Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "Results written to %s (%d experiments, sorted by val_macro_f1 ↓)",
        output_csv, len(rows),
    )


if __name__ == "__main__":
    main()
