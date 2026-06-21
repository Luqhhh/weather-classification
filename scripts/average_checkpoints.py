#!/usr/bin/env python3
"""Average model weights from training checkpoints."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state(checkpoint: dict) -> Dict[str, torch.Tensor]:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _checkpoint_score(checkpoint: dict, metric: str) -> float | None:
    metrics = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
    value = metrics.get(metric)
    return float(value) if value is not None else None


def _discover_checkpoints(checkpoint_dir: Path, pattern: str) -> List[Path]:
    paths = sorted(checkpoint_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No checkpoints matched {checkpoint_dir}/{pattern}")
    return paths


def _select_checkpoints(
    paths: List[Path],
    metric: str,
    mode: str,
    top_k: int | None,
) -> List[Tuple[Path, dict, float | None]]:
    loaded = [(path, _load_checkpoint(path), None) for path in paths]
    scored = [
        (path, checkpoint, _checkpoint_score(checkpoint, metric))
        for path, checkpoint, _ in loaded
    ]

    if top_k is None or top_k >= len(scored):
        return scored

    reverse = mode == "max"
    if all(score is not None for _, _, score in scored):
        scored = sorted(scored, key=lambda item: item[2], reverse=reverse)
    return scored[:top_k]


def average_states(states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not states:
        raise ValueError("No states to average")

    reference_keys = set(states[0].keys())
    for index, state in enumerate(states[1:], start=2):
        if set(state.keys()) != reference_keys:
            raise ValueError(f"Checkpoint {index} has different state_dict keys")

    averaged: Dict[str, torch.Tensor] = {}
    dtypes = {key: value.dtype for key, value in states[0].items()}

    for key, first_value in states[0].items():
        if torch.is_floating_point(first_value):
            value_sum = first_value.detach().clone().float()
            for state in states[1:]:
                value = state[key]
                if value.shape != first_value.shape:
                    raise ValueError(f"Shape mismatch for {key}: {value.shape} != {first_value.shape}")
                value_sum.add_(value.detach().float())
            averaged[key] = (value_sum / len(states)).to(dtype=dtypes[key])
        else:
            averaged[key] = first_value.detach().clone()

    return averaged


def main() -> None:
    parser = argparse.ArgumentParser(description="Average model checkpoint weights")
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=None)
    parser.add_argument("--pattern", type=str, default="epoch_*.pth")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", type=str, default="val_macro_f1")
    parser.add_argument("--mode", choices=["max", "min"], default="max")
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    if args.checkpoints:
        paths = args.checkpoints
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Checkpoint(s) not found: {missing}")
    elif args.checkpoint_dir:
        paths = _discover_checkpoints(args.checkpoint_dir, args.pattern)
    else:
        raise ValueError("Provide either --checkpoint_dir or --checkpoints")
    selected = _select_checkpoints(paths, args.metric, args.mode, args.top_k)
    states = [_extract_state(checkpoint) for _, checkpoint, _ in selected]
    averaged_state = average_states(states)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = [
        {"path": str(path), args.metric: score}
        for path, _, score in selected
    ]
    torch.save(
        {
            "model_state_dict": averaged_state,
            "averaged_checkpoints": metadata,
            "num_checkpoints": len(selected),
        },
        args.output,
    )

    metadata_path = args.output.with_suffix(".json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "output": str(args.output),
                "metric": args.metric,
                "mode": args.mode,
                "checkpoints": metadata,
            },
            f,
            indent=2,
        )

    logger.info("Averaged %d checkpoints into %s", len(selected), args.output)
    for item in metadata:
        logger.info("  %s: %s=%s", item["path"], args.metric, item[args.metric])


if __name__ == "__main__":
    main()
