#!/usr/bin/env python3
"""
Convert model weights to FP16 (half precision).

Reduces weight file size by ~50% with negligible impact on accuracy
(FP16 → FP32 roundtrip max error ~2e-3, logit difference ~1.5e-4).

Usage:
    python scripts/convert_to_fp16.py --input weights/convnext_tiny_best.pth --output results/convnext_tiny_fp16.pth
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def convert_weights_to_fp16(input_path: Path, output_path: Path) -> None:
    """Load FP32 weights and save as FP16.

    Handles both storage formats:
    - Plain state_dict: ``{layer.weight: tensor, ...}``
    - Wrapped checkpoint: ``{"model_state_dict": {...}, "epoch": ..., ...}``

    Only tensor values are converted to half precision; metadata keys
    (epoch, metrics, etc.) are preserved as-is.
    """
    logger.info(f"Loading weights from {input_path}")
    state = torch.load(input_path, map_location="cpu", weights_only=True)

    original_size = input_path.stat().st_size / (1024 * 1024)

    if isinstance(state, dict):
        # Determine format: wrapped checkpoint or plain state_dict
        if "model_state_dict" in state:
            logger.info("Detected wrapped checkpoint format — converting model weights only")
            state["model_state_dict"] = {
                k: v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                for k, v in state["model_state_dict"].items()
            }
        else:
            # Plain state_dict — check if all values are tensors (model weights)
            tensor_keys = [k for k, v in state.items() if isinstance(v, torch.Tensor)]
            if tensor_keys:
                logger.info(f"Detected plain state_dict format ({len(tensor_keys)} tensor keys)")
                for k in tensor_keys:
                    if state[k].is_floating_point():
                        state[k] = state[k].half()
            else:
                logger.warning("State dict contains no tensors — nothing to convert")
    else:
        logger.error("Unsupported weight file format (expected dict)")
        sys.exit(1)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)

    new_size = output_path.stat().st_size / (1024 * 1024)
    reduction = 100 * (1 - new_size / original_size)
    logger.info(f"Saved FP16 weights to {output_path}")
    logger.info(f"  {original_size:.1f} MB → {new_size:.1f} MB ({reduction:.1f}% reduction)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert model weights to FP16 (half precision)"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to FP32 model weights (.pth)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for FP16 weights (.pth)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    convert_weights_to_fp16(input_path, Path(args.output))


if __name__ == "__main__":
    main()
