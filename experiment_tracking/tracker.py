"""
Experiment result schema and persistence.

Defines the canonical ``results.json`` schema and provides helpers
to build, validate, save, and load it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .git_utils import capture_git_metadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Single experiment result matching the 18-field CSV schema.

    All fields are optional except ``experiment_id`` so the dataclass can
    represent partially-completed experiments (e.g. training-only before
    evaluation runs).
    """

    experiment_id: str
    status: str = "registered"       # "registered" | "partial" | "complete"
    created_at: str = ""

    # Git
    branch: Optional[str] = None
    commit_hash: Optional[str] = None

    # Config
    model: str = ""
    image_size: str = ""
    loss: str = ""
    augmentation: str = ""
    dropout: str = ""
    batch_size: str = ""

    # Core metrics
    val_macro_f1: str = ""
    cloudy_f1: str = ""
    rainy_f1: str = ""
    snowy_f1: str = ""
    sunny_f1: str = ""

    # Extended
    cpu_time_per_image: str = ""
    model_size_mb: str = ""
    submit_check_passed: str = ""
    notes: str = ""

    # Rich context (not flattened into CSV but kept for leaderboard detail)
    git: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    benchmark: Dict[str, Any] = field(default_factory=dict)
    submit_check: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        d = asdict(self)
        # Remove empty top-level flat duplicates (they live inside sub-dicts)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        """Reconstruct from a JSON-loaded dictionary."""
        # Pull known fields; ignore extras
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)

    def flatten_for_csv(self, columns: list[str]) -> Dict[str, str]:
        """Return a flat dict keyed by *columns* suitable for csv.DictWriter."""
        d = self.to_dict()
        flat: Dict[str, str] = {}
        for col in columns:
            val = d.get(col, "")
            if val is None:
                val = ""
            flat[col] = str(val)
        return flat


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
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


class ExperimentTracker:
    """Build and persist ``results.json`` for an experiment."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize_augmentation(aug_cfg: dict | None) -> str:
        """Return a short human-readable augmentation string."""
        if not aug_cfg:
            return "none"
        # Heuristic: if every sub-setting is disabled, call it "none"
        scale = aug_cfg.get("scale", [0.8, 1.0])
        flip = aug_cfg.get("horizontal_flip_prob", 0.5)
        rotation = aug_cfg.get("rotation_degrees", 10)
        jitter = aug_cfg.get("color_jitter", {})
        if (scale == [1.0, 1.0] and flip == 0 and rotation == 0
                and (not jitter or not any(jitter.values()))):
            return "none"
        return "default"

    @staticmethod
    def _extract_config_fields(config: dict) -> Dict[str, str]:
        """Pull key training-config values into flat strings."""
        model_cfg = config.get("model", {})
        training_cfg = config.get("training", {})
        data_cfg = config.get("data", {})
        loss_cfg = training_cfg.get("loss", {})
        aug_cfg = data_cfg.get("augmentation")

        return {
            "model": str(model_cfg.get("name", "")),
            "image_size": str(data_cfg.get("image_size", "")),
            "loss": str(loss_cfg.get("name", "cross_entropy")),
            "augmentation": ExperimentTracker._summarize_augmentation(aug_cfg),
            "dropout": str(model_cfg.get("dropout", "")),
            "batch_size": str(training_cfg.get("batch_size", "")),
        }

    @staticmethod
    def _extract_metrics(evaluation: dict) -> Dict[str, str]:
        """Pull per-class F1 and macro metrics from an evaluation dict."""
        per_class = evaluation.get("per_class", {})
        result: Dict[str, str] = {}

        def _fmt(v) -> str:
            if v is None or v == "":
                return ""
            try:
                return f"{float(v):.4f}"
            except (ValueError, TypeError):
                return str(v)

        result["val_macro_f1"] = _fmt(evaluation.get("macro_f1", ""))
        for cls_name in ("cloudy", "rainy", "snowy", "sunny"):
            entry = per_class.get(cls_name, {})
            if isinstance(entry, dict):
                result[f"{cls_name}_f1"] = _fmt(entry.get("f1", ""))
            else:
                result[f"{cls_name}_f1"] = ""
        return result

    # ------------------------------------------------------------------
    # build / save / load
    # ------------------------------------------------------------------

    def build_result(
        self,
        *,
        config: Optional[dict] = None,
        training_summary: Optional[dict] = None,
        evaluation_metrics: Optional[dict] = None,
        benchmark_results: Optional[dict] = None,
        submit_check_results: Optional[dict] = None,
        notes: str = "",
        experiment_id: Optional[str] = None,
    ) -> ExperimentResult:
        """Assemble a complete ``ExperimentResult``.

        Args:
            config: Merged training config (from YAML).
            training_summary: Dict with ``best_epoch``, ``total_epochs``,
                ``early_stopped``, ``training_time_min``,
                ``best_val_macro_f1``, ``best_val_accuracy``, etc.
            evaluation_metrics: Return value of ``compute_metrics()``.
            benchmark_results: Return value of ``CpuBenchmark.run()`` (or subset).
            submit_check_results: Return value of ``SubmitChecker.run_all_checks()``.
            notes: Free-text experiment notes.
            experiment_id: Explicit ID; auto-generated from ``output_dir`` name
                if not provided.
        """
        config = config or {}
        training_summary = training_summary or {}
        evaluation_metrics = evaluation_metrics or {}
        benchmark_results = benchmark_results or {}
        submit_check_results = submit_check_results or {}

        # --- experiment_id ---
        if experiment_id:
            eid = experiment_id
        else:
            eid = self.output_dir.name

        # --- git ---
        git_meta = capture_git_metadata()

        # --- config fields ---
        config_flat = self._extract_config_fields(config)

        # --- metrics ---
        metrics_flat = self._extract_metrics(evaluation_metrics)

        # --- benchmark ---
        per_image_ms = ""
        model_mb = ""
        if benchmark_results:
            scoring = benchmark_results.get("scoring_estimate", {})
            # per_image_mean from the optimal batch
            optimal_batch = str(benchmark_results.get("optimal_batch_size", 8))
            batch_results = benchmark_results.get("batch_results", {})
            opt_result = batch_results.get(optimal_batch, {})
            per_image_ms = str(opt_result.get("per_image_mean_ms", ""))
            model_mb = str(benchmark_results.get("weight_size_mb", ""))

        # --- submit check ---
        submit_ok = ""
        if submit_check_results:
            submit_ok = str(submit_check_results.get("all_passed", ""))

        # --- status ---
        if evaluation_metrics and benchmark_results and submit_check_results:
            status = "complete"
        elif evaluation_metrics:
            status = "partial"
        else:
            status = "registered"

        # --- artifacts ---
        artifacts: Dict[str, str] = {}
        for cand in (
            "best_model.pth",
            "training_history.csv",
            "training_log.jsonl",
            "config.yaml",
            "confusion_matrix.png",
            "confusion_matrix_final.png",
            "error_samples.csv",
            "error_samples_grid.png",
            "cpu_benchmark.csv",
        ):
            p = self.output_dir / cand
            if p.is_file():
                artifacts[cand] = cand

        return ExperimentResult(
            experiment_id=eid,
            status=status,
            created_at=datetime.now(timezone.utc).isoformat(),
            branch=git_meta.get("branch"),
            commit_hash=git_meta.get("commit_hash"),
            model=config_flat.get("model", ""),
            image_size=config_flat.get("image_size", ""),
            loss=config_flat.get("loss", ""),
            augmentation=config_flat.get("augmentation", ""),
            dropout=config_flat.get("dropout", ""),
            batch_size=config_flat.get("batch_size", ""),
            val_macro_f1=metrics_flat.get("val_macro_f1", ""),
            cloudy_f1=metrics_flat.get("cloudy_f1", ""),
            rainy_f1=metrics_flat.get("rainy_f1", ""),
            snowy_f1=metrics_flat.get("snowy_f1", ""),
            sunny_f1=metrics_flat.get("sunny_f1", ""),
            cpu_time_per_image=per_image_ms,
            model_size_mb=model_mb,
            submit_check_passed=submit_ok,
            notes=notes,
            git=git_meta,
            config=config,
            training=training_summary,
            evaluation=evaluation_metrics,
            benchmark=benchmark_results,
            submit_check=submit_check_results,
            artifacts=artifacts,
        )

    def save(self, result: ExperimentResult) -> Path:
        """Write ``results.json`` to ``self.output_dir``."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        logger.info("Experiment result saved to %s", path)
        return path

    def load(self) -> Optional[ExperimentResult]:
        """Load ``results.json`` from ``self.output_dir`` if it exists."""
        path = self.output_dir / "results.json"
        if not path.is_file():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ExperimentResult.from_dict(data)

    def get_next_experiment_id(self, experiments_dir: str | Path = "experiments") -> str:
        """Auto-increment experiment ID by scanning existing results.

        Returns ``exp_001`` when no prior experiments exist.
        """
        experiments_dir = Path(experiments_dir)
        max_num = 0
        pattern = "exp_"

        # Scan experiments/results.csv if it exists
        csv_path = experiments_dir / "results.csv"
        if csv_path.is_file():
            import csv
            with open(csv_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    eid = row.get("experiment_id", "")
                    if eid.startswith(pattern):
                        try:
                            num = int(eid[len(pattern):])
                            max_num = max(max_num, num)
                        except ValueError:
                            pass

        # Also scan experiments/ subdirectories
        if experiments_dir.is_dir():
            for child in experiments_dir.iterdir():
                if child.is_dir() and child.name.startswith(pattern):
                    rj = child / "results.json"
                    if rj.is_file():
                        try:
                            num = int(child.name[len(pattern):])
                            max_num = max(max_num, num)
                        except ValueError:
                            pass

        return f"exp_{max_num + 1:03d}"
