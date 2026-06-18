#!/usr/bin/env python3
"""
Experiment Aggregator — Leaderboard & CSV Generator

Scans the outputs/ and experiments/ directories for ``results.json`` files
(produced by evaluate.py or train.py), aggregates them, and produces:

    experiments/results.csv    — machine-readable table (18 fields)
    experiments/leaderboard.md — human-readable leaderboard

Usage:
    python scripts/aggregate_experiments.py
    python scripts/aggregate_experiments.py --outputs-dir outputs --experiments-dir experiments
    python scripts/aggregate_experiments.py --dry-run          # print CSV to stdout
    python scripts/aggregate_experiments.py --no-preserve       # regenerate leaderboard from scratch
    python scripts/aggregate_experiments.py --from-logs         # fallback: scan training_log.jsonl

Output fields (CSV columns):
    experiment_id, branch, commit_hash, model, image_size, loss,
    augmentation, dropout, batch_size, val_macro_f1, cloudy_f1,
    rainy_f1, snowy_f1, sunny_f1, cpu_time_per_image,
    model_size_mb, submit_check_passed, notes
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_tracking.aggregator import ExperimentAggregator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _run_from_logs(outputs_dir: Path, output_csv: Path):
    """Fallback: scan training_log.jsonl (backward-compatible mode)."""
    from scripts.summarize_experiments import (
        _find_experiment_dirs,
        _summarize_one_experiment,
        CSV_COLUMNS,
    )
    import csv

    exp_dirs = _find_experiment_dirs(outputs_dir)
    rows = []
    for d in exp_dirs:
        row = _summarize_one_experiment(d)
        if row:
            rows.append(row)

    rows.sort(key=lambda r: float(r.get("val_macro_f1") or 0), reverse=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "Results from logs written to %s (%d experiments, sorted by val_macro_f1 ↓)",
        output_csv, len(rows),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results → results.csv + leaderboard.md"
    )
    parser.add_argument(
        "--outputs-dir", type=str, default="outputs",
        help="Directory containing experiment output subdirectories (default: outputs)",
    )
    parser.add_argument(
        "--experiments-dir", type=str, default="experiments",
        help="Directory for results.csv and leaderboard.md (default: experiments)",
    )
    parser.add_argument(
        "--no-preserve", action="store_true",
        help="Regenerate leaderboard.md from scratch (discard existing hand-written preamble)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print CSV to stdout instead of writing files",
    )
    parser.add_argument(
        "--from-logs", action="store_true",
        help="Scan training_log.jsonl instead of results.json (backward-compatible mode, CSV only)",
    )
    parser.add_argument(
        "--csv-only", action="store_true",
        help="Only generate results.csv, skip leaderboard.md",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    experiments_dir = Path(args.experiments_dir)

    # --- Backward-compatible mode: scan training_log.jsonl ---
    if args.from_logs:
        output_csv = experiments_dir / "results.csv"
        _run_from_logs(outputs_dir, output_csv)
        return

    # --- Full mode: scan results.json ---
    aggregator = ExperimentAggregator(
        outputs_dir=outputs_dir,
        experiments_dir=experiments_dir,
    )

    results = aggregator.scan()

    if not results:
        logger.info(
            "No results.json files found. "
            "Run evaluate.py first, or use --from-logs to scan training_log.jsonl."
        )
        return

    # CSV
    csv_content = aggregator.generate_csv(results)
    if args.dry_run:
        print(csv_content)
    else:
        csv_path = aggregator.write_csv(csv_content)
        logger.info("Results CSV: %s (%d experiments)", csv_path, len(results))

    # Markdown (skip in dry-run or csv-only mode)
    if not args.dry_run and not args.csv_only:
        md_content = aggregator.generate_markdown(
            results,
            preserve_existing=not args.no_preserve,
        )
        md_path = aggregator.write_markdown(md_content)
        logger.info("Leaderboard: %s", md_path)

    # Summary
    logger.info(
        "Aggregated %d experiment(s) — top: %s (F1=%s)",
        len(results),
        results[0].experiment_id if results else "N/A",
        results[0].val_macro_f1 if results else "N/A",
    )


if __name__ == "__main__":
    main()
