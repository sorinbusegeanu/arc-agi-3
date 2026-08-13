from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import os

import numpy as np


class ArcGridEnvironment:
    """Clean v7 adapter around the ARC-AGI-3 environment package."""

    def __init__(
        self,
        *,
        game_id: str,
        seed: int = 0,
        env_root: str | None = None,
        op_mode: str = "normal",
        render_mode: str | None = None,
        auto_reset_on_empty_frame: bool = True,
        env_factory: Callable[..., Any] | None = None,
    ) -> None:
        if env_factory is None:
            _ensure_arc_paths()
            from arc_agi_agent.envs.loader import make_env
            env_factory = make_env
        self.env = env_factory(
            env_id=game_id,
            env_root=env_root,
            seed=int(seed),
            op_mode=op_mode,
            render_mode=render_mode,
        )
        self.auto_reset_on_empty_frame = bool(auto_reset_on_empty_frame)
        self.reset_count = 0
        self.skipped_terminal_steps = 0
        self.last_step_was_reset_boundary = False
        self.last_terminal_state: str | None = None
        self.last_outcome_state = "NOT_FINISHED"
        self.last_outcome_polarity = "neutral"
        self.last_levels_completed = 0
        self.level_completed_event = False
        self._last_raw = self.env.reset()
        self._update_from_raw(self._last_raw, reset_boundary=False)

    def observe(self) -> np.ndarray:
        return self._last_grid.copy()

    def reset(self) -> np.ndarray:
        self._last_raw = self.env.reset()
        self.reset_count += 1
        self.last_step_was_reset_boundary = True
        self.last_terminal_state = "explicit_reset"
        self._update_from_raw(self._last_raw, reset_boundary=True)
        return self._last_grid.copy()

    def step(self, action: int) -> np.ndarray:
        try:
            from arcengine import GameAction
            raw = self.env.step(GameAction.from_id(int(action)))
        except ImportError:
            raw = self.env.step(int(action))
        previous_levels = self.last_levels_completed
        self.last_step_was_reset_boundary = False
        self.last_terminal_state = None
        state = _state_name(raw) or "NOT_FINISHED"
        levels = int(getattr(raw, "levels_completed", previous_levels) or 0)
        try:
            grid = _grid_from_raw(raw)
        except ValueError:
            if not self.auto_reset_on_empty_frame:
                raise
            self.last_outcome_state = state if state != "NOT_FINISHED" else "GAME_OVER"
            self.last_levels_completed = levels
            self.level_completed_event = levels > previous_levels
            self.last_outcome_polarity = _polarity(self.last_outcome_state, self.level_completed_event)
            self.last_terminal_state = self.last_outcome_state if self.last_outcome_state in {"WIN", "GAME_OVER"} else None
            self._last_raw = self.env.reset()
            self._last_grid = _grid_from_raw(self._last_raw)
            self.reset_count += 1
            self.skipped_terminal_steps += 1
            self.last_step_was_reset_boundary = True
            return self._last_grid.copy()
        self._last_raw = raw
        self._last_grid = grid
        self.last_outcome_state = state
        self.last_levels_completed = levels
        self.level_completed_event = levels > previous_levels
        self.last_outcome_polarity = _polarity(state, self.level_completed_event)
        self.last_terminal_state = state if state in {"WIN", "GAME_OVER"} else None
        return grid.copy()

    def available_actions(self) -> list[int]:
        actions = getattr(self._last_raw, "available_actions", None)
        if actions:
            return [int(action) for action in actions]
        method = getattr(self.env, "available_actions", None)
        return [] if method is None else [int(action) for action in method()]

    def _update_from_raw(self, raw: Any, *, reset_boundary: bool) -> None:
        self._last_grid = _grid_from_raw(raw)
        self.last_outcome_state = _state_name(raw) or "NOT_FINISHED"
        self.last_levels_completed = int(getattr(raw, "levels_completed", 0) or 0)
        self.level_completed_event = False
        self.last_outcome_polarity = "neutral"
        if not reset_boundary:
            self.last_terminal_state = self.last_outcome_state if self.last_outcome_state in {"WIN", "GAME_OVER"} else None


def registered_game_ids(env_root: str | None = None) -> tuple[str, ...]:
    game_ids: set[str] = set()
    for root in _candidate_environment_roots(env_root):
        if not root.is_dir():
            continue
        for game_dir in root.iterdir():
            if game_dir.is_dir() and any(game_dir.glob("*/metadata.json")):
                game_ids.add(game_dir.name)
    return tuple(sorted(game_ids))


def _ensure_arc_paths() -> None:
    import sys
    repo_root = Path(__file__).resolve().parents[3]
    for relative in ("other_repos/arc-agi", "other_repos/ARCEngine"):
        value = str(repo_root / relative)
        if value not in sys.path:
            sys.path.insert(0, value)


def _candidate_environment_roots(env_root: str | None) -> tuple[Path, ...]:
    repo_root = Path(__file__).resolve().parents[3]
    values = [env_root, os.environ.get("ENVIRONMENTS_DIR"), str(repo_root / "other_repos" / "arc-interactive" / "environment_files"), str(repo_root / "environment_files")]
    result: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        if not value:
            continue
        path = Path(value).expanduser()
        if path not in seen:
            seen.add(path)
            result.append(path)
    return tuple(result)


def _grid_from_raw(raw: Any) -> np.ndarray:
    frame = getattr(raw, "frame", raw)
    if isinstance(frame, list):
        if not frame:
            raise ValueError("raw frame list is empty")
        frame = frame[0]
    grid = np.asarray(frame, dtype=np.int64)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError(f"expected non-empty 2D ARC grid, got shape {grid.shape}")
    return grid


def _state_name(raw: Any) -> str | None:
    state = getattr(raw, "state", None)
    if state is None:
        return None
    return str(getattr(state, "value", state))


def _polarity(state: str, level_completed: bool) -> str:
    if state == "WIN" or level_completed:
        return "positive"
    if state == "GAME_OVER":
        return "negative"
    return "neutral"
