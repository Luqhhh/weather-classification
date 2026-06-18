"""
Experiment tracking and leaderboard generation.

Provides:

- ``capture_git_metadata()`` — collect git branch / commit / dirty status
- ``ExperimentResult`` — dataclass schema for ``results.json``
- ``ExperimentTracker`` — build / save / load ``results.json``
- ``ExperimentAggregator`` — scan experiments, produce ``results.csv`` and ``leaderboard.md``
- ``CSV_COLUMNS`` — the canonical 18-field column list
"""

from .git_utils import capture_git_metadata
from .tracker import CSV_COLUMNS, ExperimentResult, ExperimentTracker
from .aggregator import ExperimentAggregator

__all__ = [
    "capture_git_metadata",
    "CSV_COLUMNS",
    "ExperimentResult",
    "ExperimentTracker",
    "ExperimentAggregator",
]
