from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional


def _has_local_environment(env_id: str, env_root: str) -> bool:
    root = Path(env_root) / env_id
    if not root.exists():
        return False
    return any(root.glob("*/metadata.json"))


def make_env(env_id: str, env_root: Optional[str] = None, seed: int = 0, op_mode: str = "normal", render_mode: Optional[str] = None) -> Any:
    if env_root:
        os.environ["ENVIRONMENTS_DIR"] = env_root
    from arc_agi import Arcade, OperationMode

    local_only = bool(env_root and _has_local_environment(env_id, env_root))
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
