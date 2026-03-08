from __future__ import annotations

import os
from typing import Any, Optional


def make_env(env_id: str, env_root: Optional[str] = None, seed: int = 0, op_mode: str = "normal", render_mode: Optional[str] = None) -> Any:
    if env_root:
        os.environ["ENVIRONMENTS_DIR"] = env_root
    from arc_agi import Arcade, OperationMode

    arcade = Arcade(operation_mode=OperationMode(op_mode))
    env = arcade.make(env_id, seed=int(seed), render_mode=render_mode)
    if env is None:
        raise RuntimeError(f"arcade.make failed for env_id={env_id}")
    return env
