from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch


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


def set_global_seeds(
    seed: int,
    *,
    deterministic_torch: bool = False,
    cudnn_deterministic: bool = False,
    cudnn_benchmark: bool = True,
) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
