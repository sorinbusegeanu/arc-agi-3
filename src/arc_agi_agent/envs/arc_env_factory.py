from __future__ import annotations

from typing import Any, Optional

from .loader import make_env


def create_env(
    env_id: str,
    env_root: Optional[str] = None,
    seed: int = 0,
    op_mode: str = "normal",
    render_terminal: bool = False,
) -> Any:
    render_mode = "terminal-fast" if render_terminal else None
    return make_env(env_id=env_id, env_root=env_root, seed=seed, op_mode=op_mode, render_mode=render_mode)
