#!/usr/bin/env python3
"""Compute simple image statistics for domain-shift and error-pattern diagnosis."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import WeatherDataset
from data.label_mapping import detect_label_mapping, load_label_mapping


FEATURE_FIELDS = ["brightness", "saturation", "white_ratio", "edge_density"]


def _load_predictions(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


def _feature_stats(image: Image.Image) -> dict[str, float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)

    brightness = float(gray.mean())
    saturation = float(hsv[:, :, 1].mean())
    white_ratio = float(((rgb[:, :, 0] > 0.82) & (rgb[:, :, 1] > 0.82) & (rgb[:, :, 2] > 0.82)).mean())

    grad_y = np.abs(np.diff(gray, axis=0))
    grad_x = np.abs(np.diff(gray, axis=1))
    edge_density = float(
        (grad_y > 0.08).mean() * 0.5 + (grad_x > 0.08).mean() * 0.5
    )

    return {
        "brightness": round(brightness, 6),
        "saturation": round(saturation, 6),
        "white_ratio": round(white_ratio, 6),
        "edge_density": round(edge_density, 6),
    }


def _summarize(rows: list[dict], group_key: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        value = row.get(group_key, "")
        groups[str(value)].append(row)

    summaries = []
    for group, items in sorted(groups.items()):
        summary = {"group_by": group_key, "group": group, "count": len(items)}
        for field in FEATURE_FIELDS:
            values = np.array([float(item[field]) for item in items], dtype=float)
            summary[f"{field}_mean"] = round(float(values.mean()), 6)
            summary[f"{field}_std"] = round(float(values.std()), 6)
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze hand-built image features")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--label_mapping", type=str, default=None)
    parser.add_argument("--predictions", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir).resolve()
    if args.label_mapping and Path(args.label_mapping).exists():
        label_mapper = load_label_mapping(args.label_mapping)
    else:
        label_mapper = detect_label_mapping(data_dir)
    dataset = WeatherDataset(data_dir=data_dir, transform=None, label_mapper=label_mapper)
    predictions = _load_predictions(Path(args.predictions) if args.predictions else None)

    rows = []
    for img_path, label_idx in dataset.images:
        image = dataset._safe_load_image(img_path)
        if image is None:
            continue
        rel_path = str(img_path.resolve().relative_to(data_dir))
        pred = predictions.get(rel_path, {})
        row = {
            "filename": rel_path,
            "true_label": label_mapper.decode(label_idx),
            "predicted_label": pred.get("predicted_label", ""),
            "confidence": pred.get("confidence", ""),
            "correct": pred.get("correct", ""),
        }
        row.update(_feature_stats(image))
        rows.append(row)

    feature_csv = output_dir / "image_features.csv"
    with open(feature_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "filename",
            "true_label",
            "predicted_label",
            "confidence",
            "correct",
            *FEATURE_FIELDS,
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    summaries.extend(_summarize(rows, "true_label"))
    if predictions:
        summaries.extend(_summarize(rows, "correct"))
        summaries.extend(_summarize(rows, "predicted_label"))

    summary_csv = output_dir / "feature_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "group_by",
            "group",
            "count",
            *[f"{field}_{suffix}" for field in FEATURE_FIELDS for suffix in ("mean", "std")],
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    with open(output_dir / "feature_summary.json", "w", encoding="utf-8") as f:
        json.dump({"num_images": len(rows), "summaries": summaries}, f, indent=2, ensure_ascii=False)

    print(f"Wrote {feature_csv}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
