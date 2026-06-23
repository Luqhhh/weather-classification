#!/usr/bin/env python3
"""Analyze confidence distributions from per-sample prediction CSV files."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

import numpy as np


def _load_predictions(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"filename", "true_label", "predicted_label", "confidence", "correct"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    for row in rows:
        row["confidence"] = float(row["confidence"])
        row["correct"] = int(row["correct"])
    return rows


def _describe(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.array(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": round(float(arr.mean()), 6),
        "median": round(float(np.median(arr)), 6),
        "p10": round(float(np.quantile(arr, 0.10)), 6),
        "p25": round(float(np.quantile(arr, 0.25)), 6),
        "p75": round(float(np.quantile(arr, 0.75)), 6),
        "p90": round(float(np.quantile(arr, 0.90)), 6),
    }


def _summary(rows: list[dict]) -> dict:
    confs = [row["confidence"] for row in rows]
    correct = [row["confidence"] for row in rows if row["correct"]]
    wrong = [row["confidence"] for row in rows if not row["correct"]]
    return {
        "num_samples": len(rows),
        "accuracy": round(sum(row["correct"] for row in rows) / max(1, len(rows)), 6),
        "all": _describe(confs),
        "correct": _describe(correct),
        "incorrect": _describe(wrong),
    }


def _write_bins(rows: list[dict], output_path: Path, name: str) -> None:
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "bin_start",
                "bin_end",
                "count",
                "accuracy",
                "mean_confidence",
            ],
        )
        writer.writeheader()
        for lo, hi in zip(bins[:-1], bins[1:]):
            selected = [
                row for row in rows
                if lo <= row["confidence"] < hi or (hi == 1.0 and row["confidence"] <= hi)
            ]
            if selected:
                acc = sum(row["correct"] for row in selected) / len(selected)
                avg_conf = mean(row["confidence"] for row in selected)
            else:
                acc = 0.0
                avg_conf = 0.0
            writer.writerow({
                "name": name,
                "bin_start": lo,
                "bin_end": hi,
                "count": len(selected),
                "accuracy": round(acc, 6),
                "mean_confidence": round(avg_conf, 6),
            })


def _comparison_summary(base_rows: list[dict], comparison_rows: list[dict]) -> dict:
    by_file = {row["filename"]: row for row in comparison_rows}
    fixed = []
    regressed = []
    both_wrong = []
    both_correct = []
    for row in base_rows:
        other = by_file.get(row["filename"])
        if other is None:
            continue
        if not row["correct"] and other["correct"]:
            fixed.append(row)
        elif row["correct"] and not other["correct"]:
            regressed.append(row)
        elif not row["correct"] and not other["correct"]:
            both_wrong.append(row)
        else:
            both_correct.append(row)

    return {
        "matched_samples": len(fixed) + len(regressed) + len(both_wrong) + len(both_correct),
        "fixed_by_comparison": {
            "count": len(fixed),
            "base_confidence": _describe([row["confidence"] for row in fixed]),
        },
        "regressed_by_comparison": {
            "count": len(regressed),
            "base_confidence": _describe([row["confidence"] for row in regressed]),
        },
        "both_wrong": {
            "count": len(both_wrong),
            "base_confidence": _describe([row["confidence"] for row in both_wrong]),
        },
        "both_correct": {
            "count": len(both_correct),
            "base_confidence": _describe([row["confidence"] for row in both_correct]),
        },
    }


def _write_comparison_rows(
    base_rows: list[dict],
    comparison_rows: list[dict],
    output_path: Path,
) -> None:
    by_file = {row["filename"]: row for row in comparison_rows}
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "true_label",
                "base_predicted_label",
                "base_confidence",
                "base_correct",
                "comparison_predicted_label",
                "comparison_confidence",
                "comparison_correct",
                "change_type",
            ],
        )
        writer.writeheader()
        for row in base_rows:
            other = by_file.get(row["filename"])
            if other is None:
                continue
            if not row["correct"] and other["correct"]:
                change_type = "fixed"
            elif row["correct"] and not other["correct"]:
                change_type = "regressed"
            elif not row["correct"] and not other["correct"]:
                change_type = "both_wrong"
            else:
                change_type = "both_correct"
            writer.writerow({
                "filename": row["filename"],
                "true_label": row["true_label"],
                "base_predicted_label": row["predicted_label"],
                "base_confidence": row["confidence"],
                "base_correct": row["correct"],
                "comparison_predicted_label": other["predicted_label"],
                "comparison_confidence": other["confidence"],
                "comparison_correct": other["correct"],
                "change_type": change_type,
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze confidence distributions")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--name", type=str, default="base")
    parser.add_argument("--comparison_predictions", type=str, default=None)
    parser.add_argument("--comparison_name", type=str, default="comparison")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_predictions(Path(args.predictions))
    summary = {args.name: _summary(rows)}
    _write_bins(rows, output_dir / f"{args.name}_confidence_bins.csv", args.name)

    if args.comparison_predictions:
        comparison_rows = _load_predictions(Path(args.comparison_predictions))
        summary[args.comparison_name] = _summary(comparison_rows)
        summary["comparison"] = _comparison_summary(rows, comparison_rows)
        _write_bins(
            comparison_rows,
            output_dir / f"{args.comparison_name}_confidence_bins.csv",
            args.comparison_name,
        )
        _write_comparison_rows(
            rows,
            comparison_rows,
            output_dir / "confidence_comparison.csv",
        )

    with open(output_dir / "confidence_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
