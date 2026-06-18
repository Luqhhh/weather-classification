"""
Git metadata capture for experiment tracking.

Provides branch name, commit hash, and working-tree status
so every experiment is reproducible from its source code.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def capture_git_metadata(repo_root: Optional[str] = None) -> dict:
    """Capture git branch, commit hash, and dirty status.

    Args:
        repo_root: Path to the git repository root.  Defaults to the
            parent directory of this file (project root).

    Returns:
        A dict with keys ``branch``, ``commit_hash``, ``commit_message``,
        ``dirty``, and ``dirty_files``.  All values are ``None`` when
        the directory is not inside a git repository or git is unavailable.
    """
    if repo_root is None:
        repo_root = str(Path(__file__).resolve().parent.parent)

    def _git(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "-C", repo_root] + list(args),
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    commit_hash = _git("rev-parse", "HEAD")
    if commit_hash is None:
        logger.debug("Not a git repository or git unavailable at %s", repo_root)
        return {
            "branch": None,
            "commit_hash": None,
            "commit_message": None,
            "dirty": None,
            "dirty_files": None,
        }

    # Short hash for display
    commit_short = _git("rev-parse", "--short", "HEAD")

    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    commit_message = _git("log", "-1", "--format=%s")

    # Dirty check
    status_output = _git("status", "--porcelain")
    dirty = bool(status_output)
    dirty_files = status_output.split("\n")[:10] if dirty else []  # top 10

    return {
        "branch": branch,
        "commit_hash": commit_short or commit_hash[:8],
        "commit_message": commit_message,
        "dirty": dirty,
        "dirty_files": dirty_files,
    }
