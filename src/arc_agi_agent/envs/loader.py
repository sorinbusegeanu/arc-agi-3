from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional


def _default_environment_roots() -> tuple[Path, ...]:
    base_dir = Path(__file__).resolve().parents[3]
    return (
        base_dir / "other_repos" / "arc-interactive" / "environment_files",
        base_dir / "environment_files",
    )


def _resolve_environment_root(env_id: str, env_root: Optional[str]) -> Optional[str]:
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root))
    current = os.environ.get("ENVIRONMENTS_DIR")
    if current:
        candidates.append(Path(current))
    candidates.extend(_default_environment_roots())
    seen: set[str] = set()
    for candidate in candidates:
        candidate_text = str(candidate)
        if candidate_text in seen:
            continue
        seen.add(candidate_text)
        if _has_local_environment(env_id, candidate_text):
            return candidate_text
    for candidate in candidates:
        candidate_text = str(candidate)
        if candidate_text not in seen:
            continue
        if candidate.is_dir():
            return candidate_text
    return None


def _has_local_environment(env_id: str, env_root: str) -> bool:
    root = Path(env_root) / env_id
    if not root.exists():
        return False
    return any(root.glob("*/metadata.json"))


def make_env(env_id: str, env_root: Optional[str] = None, seed: int = 0, op_mode: str = "normal", render_mode: Optional[str] = None) -> Any:
    resolved_env_root = _resolve_environment_root(env_id, env_root)
    if resolved_env_root:
        os.environ["ENVIRONMENTS_DIR"] = resolved_env_root
    from arc_agi import Arcade, OperationMode

    local_only = bool(resolved_env_root and _has_local_environment(env_id, resolved_env_root))
    resolved_mode = "offline" if local_only else op_mode
    logger = None
    if local_only:
        logger = logging.getLogger("arc_agi.quiet_local")
        logger.setLevel(logging.WARNING)
        logger.handlers.clear()
    arcade = Arcade(operation_mode=OperationMode(resolved_mode), logger=logger)
    env = arcade.make(env_id, seed=int(seed), render_mode=render_mode)
    if env is None:
        raise RuntimeError(f"arcade.make failed for env_id={env_id}")
    return env
