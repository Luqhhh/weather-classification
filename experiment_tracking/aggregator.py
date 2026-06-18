"""
Experiment aggregator — scans for ``results.json`` files and produces
``results.csv`` and ``leaderboard.md``.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

from .tracker import CSV_COLUMNS, ExperimentResult

logger = logging.getLogger(__name__)

_AUTO_MARKER = "<!-- AUTO-GENERATED -->"


class ExperimentAggregator:
    """Scan output / experiment directories, generate CSV and markdown."""

    def __init__(
        self,
        outputs_dir: str | Path = "outputs",
        experiments_dir: str | Path = "experiments",
    ):
        self.outputs_dir = Path(outputs_dir)
        self.experiments_dir = Path(experiments_dir)

    # ------------------------------------------------------------------
    # scan
    # ------------------------------------------------------------------

    def scan(self) -> List[ExperimentResult]:
        """Find all ``results.json`` files and parse them.

        Searches both ``outputs/`` and ``experiments/``.  Duplicates
        (same experiment_id) are resolved by preferring the version in
        ``experiments/``.
        """
        seen: dict[str, ExperimentResult] = {}

        for base_dir in (self.outputs_dir, self.experiments_dir):
            if not base_dir.is_dir():
                continue
            for child in sorted(base_dir.iterdir()):
                if not child.is_dir():
                    continue
                rj = child / "results.json"
                if not rj.is_file():
                    continue
                try:
                    data = json.loads(rj.read_text(encoding="utf-8"))
                    result = ExperimentResult.from_dict(data)
                    # Prefer later (experiments/ wins over outputs/)
                    seen[result.experiment_id] = result
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    logger.warning("Skipping %s: %s", rj, exc)

        results = sorted(
            seen.values(),
            key=lambda r: _safe_float(r.val_macro_f1),
            reverse=True,
        )
        logger.info("Found %d experiment(s) with results.json", len(results))
        return results

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def generate_csv(self, results: List[ExperimentResult]) -> str:
        """Build CSV content as a string."""
        import io
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r.flatten_for_csv(CSV_COLUMNS))
        return buf.getvalue()

    def write_csv(self, content: str) -> Path:
        """Write ``results.csv`` to the experiments directory."""
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        path = self.experiments_dir / "results.csv"
        path.write_text(content, encoding="utf-8")
        logger.info("Results CSV written to %s", path)
        return path

    # ------------------------------------------------------------------
    # Markdown leaderboard
    # ------------------------------------------------------------------

    def generate_markdown(
        self,
        results: List[ExperimentResult],
        preserve_existing: bool = True,
    ) -> str:
        """Generate ``leaderboard.md`` content.

        When *preserve_existing* is True and ``leaderboard.md`` already
        exists, content above the ``<!-- AUTO-GENERATED -->`` marker is
        retained and the auto table is appended below it.
        """
        preamble = ""
        existing_path = self.experiments_dir / "leaderboard.md"

        if preserve_existing and existing_path.is_file():
            old = existing_path.read_text(encoding="utf-8")
            if _AUTO_MARKER in old:
                preamble = old.split(_AUTO_MARKER)[0].rstrip() + "\n\n"
            else:
                # No marker — keep old content as preamble
                preamble = old.rstrip() + "\n\n"

        new_sections = [
            _AUTO_MARKER,
            "",
            f"> 更新日期：{date.today()} | 排序：val_macro_f1 ↓",
            "",
            "## 排行榜",
            "",
            self._build_summary_table(results),
            "",
            "---",
            "",
        ]

        # Per-experiment detail sections
        for i, r in enumerate(results, 1):
            new_sections.append(self._build_detail_section(i, r))
            new_sections.append("")

        return preamble + "\n".join(new_sections) + "\n"

    def write_markdown(self, content: str) -> Path:
        """Write ``leaderboard.md`` to the experiments directory."""
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        path = self.experiments_dir / "leaderboard.md"
        path.write_text(content, encoding="utf-8")
        logger.info("Leaderboard written to %s", path)
        return path

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------

    def run(self, preserve_existing: bool = True) -> Tuple[Path, Path]:
        """Scan → generate CSV + markdown → write both.

        Returns:
            ``(csv_path, md_path)``
        """
        results = self.scan()
        csv_content = self.generate_csv(results)
        csv_path = self.write_csv(csv_content)
        md_content = self.generate_markdown(results, preserve_existing=preserve_existing)
        md_path = self.write_markdown(md_content)
        return csv_path, md_path

    # ==================================================================
    # Internal builders
    # ==================================================================

    @staticmethod
    def _build_summary_table(results: List[ExperimentResult]) -> str:
        if not results:
            return "| # | Experiment | Model | Loss | Size | Aug | Val F1 | rainy F1 | Best Epoch | CPU Time | Weight |\n"
            "|---|-----------|-------|------|------|-----|--------|----------|------------|----------|--------|\n"
            "*暂无实验数据*"

        header = (
            "| # | Experiment | Model | Loss | Size | Aug | Val F1 | rainy F1 | CPU Time | Weight | Submit |\n"
            "|---|-----------|-------|------|------|-----|--------|----------|----------|--------|--------|"
        )
        rows = [header]
        for i, r in enumerate(results, 1):
            val_f1 = _fmt_f1(r.val_macro_f1)
            rainy_f1 = _fmt_f1(r.rainy_f1)
            cpu = r.cpu_time_per_image or "—"
            weight = r.model_size_mb or "—"
            submit = "✅" if r.submit_check_passed.lower() == "true" else ("—" if not r.submit_check_passed else "❌")
            row = (
                f"| {i} | **{r.experiment_id}** | {r.model or '—'} | {r.loss or '—'} "
                f"| {r.image_size or '—'} | {r.augmentation or '—'} "
                f"| **{val_f1}** | {rainy_f1} | {cpu} | {weight} | {submit} |"
            )
            rows.append(row)
        return "\n".join(rows)

    @staticmethod
    def _build_detail_section(rank: int, r: ExperimentResult) -> str:
        lines = [
            f"## {r.experiment_id}: {r.model} + {r.loss} + {r.image_size}",
            "",
        ]

        # Basic info table
        lines.append("| 字段 | 值 |")
        lines.append("|------|----|")
        lines.append(f"| 排名 | #{rank} |")
        lines.append(f"| 状态 | {r.status} |")
        if r.branch:
            lines.append(f"| 分支 | `{r.branch}` |")
        if r.commit_hash:
            lines.append(f"| Commit | `{r.commit_hash}` |")
        lines.append(f"| 模型 | {r.model or '—'} |")
        lines.append(f"| 输入尺寸 | {r.image_size or '—'} |")
        lines.append(f"| 损失函数 | {r.loss or '—'} |")
        lines.append(f"| 数据增强 | {r.augmentation or '—'} |")
        lines.append(f"| Dropout | {r.dropout or '—'} |")
        lines.append(f"| Batch Size | {r.batch_size or '—'} |")
        lines.append("")

        # Metrics
        if r.val_macro_f1:
            lines.append("### 评估指标")
            lines.append("")
            lines.append("```")
            lines.append(f"Macro F1:  {r.val_macro_f1}")
            for cls_name in ("cloudy", "rainy", "snowy", "sunny"):
                f1_val = getattr(r, f"{cls_name}_f1", "") or "—"
                lines.append(f"  {cls_name}: F1 {f1_val}")
            lines.append("```")
            lines.append("")

        # Benchmark
        if r.cpu_time_per_image or r.model_size_mb:
            lines.append("### 性能")
            lines.append("")
            if r.cpu_time_per_image:
                lines.append(f"- CPU 每张推理: {r.cpu_time_per_image} ms")
            if r.model_size_mb:
                lines.append(f"- 模型大小: {r.model_size_mb} MB")
            if r.submit_check_passed:
                passed = "✅ 通过" if r.submit_check_passed.lower() == "true" else "❌ 未通过"
                lines.append(f"- 提交检查: {passed}")
            lines.append("")

        # Training detail
        training = r.training
        if training:
            lines.append("### 训练信息")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|----|")
            best_ep = training.get("best_epoch", "")
            if isinstance(best_ep, (int, float)) and best_ep >= 0:
                best_ep = int(best_ep) + 1  # 0-indexed → 1-indexed
            lines.append(f"| 最佳轮次 | {best_ep} |")
            lines.append(f"| 总轮次 | {training.get('total_epochs', '—')} |")
            lines.append(f"| 早停 | {'是' if training.get('early_stopped') else '否'} |")
            train_time = training.get("training_time_min", "")
            if train_time:
                lines.append(f"| 训练时长 | {train_time} min |")
            if training.get("best_val_macro_f1"):
                lines.append(f"| 最佳 Val F1 | {training['best_val_macro_f1']} |")
            lines.append("")

        # Notes
        if r.notes:
            lines.append("### 备注")
            lines.append("")
            lines.append(r.notes)
            lines.append("")

        lines.append("---")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _safe_float(value: str) -> float:
    """Parse a string to float, returning 0.0 on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _fmt_f1(value: str) -> str:
    """Format an F1 string for table display."""
    if not value:
        return "—"
    v = _safe_float(value)
    if v == 0.0 and not value.strip():
        return "—"
    return f"{v:.4f}"
