from __future__ import annotations

import logging
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_arc_paths() -> None:
    root = repo_root()
    candidates = (
        root / "other_repos" / "arc-agi",
        root / "other_repos" / "ARCEngine",
    )
    for candidate in candidates:
        text = str(candidate)
        if candidate.exists() and text not in sys.path:
            sys.path.insert(0, text)


def build_quiet_arc_logger() -> logging.Logger:
    logger = logging.getLogger("rl_v1.arc_quiet")
    logger.setLevel(logging.ERROR)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def resolve_environments_dir() -> str:
    root = repo_root()
    candidates = (
        root / "other_repos" / "arc-interactive" / "environment_files",
        root / "environment_files",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return str(candidates[-1])
