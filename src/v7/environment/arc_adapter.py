from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np


class ArcGridEnvironment:
    """Adapts a local ARC environment using the same external integration as v6."""

    def __init__(self, *, game_id: str, seed: int = 0, env_root: str | None = None, op_mode: str = "normal", render_mode: str | None = None, auto_reset_on_empty_frame: bool = True, env_factory: Callable[..., Any] | None = None) -> None:
        if env_factory is None:
            _ensure_arc_paths()
            from arc_agi_agent.envs.loader import make_env
            env_factory = make_env
        self.env = env_factory(env_id=game_id, env_root=env_root, seed=int(seed), op_mode=op_mode, render_mode=render_mode)
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
        self.last_outcome_state = _state_name(self._last_raw) or "NOT_FINISHED"
        self.last_levels_completed = int(getattr(self._last_raw, "levels_completed", 0) or 0)
        self._last_grid = _grid_from_raw(self._last_raw)

    def observe(self) -> np.ndarray:
        return self._last_grid.copy()

    def reset(self) -> np.ndarray:
        raw = self.env.reset()
        self._last_raw = raw
        self.last_outcome_state = _state_name(raw) or "NOT_FINISHED"
        self.last_outcome_polarity = "neutral"
        self.last_levels_completed = int(getattr(raw, "levels_completed", 0) or 0)
        self.level_completed_event = False
        self._last_grid = _grid_from_raw(raw)
        self.reset_count += 1
        self.last_step_was_reset_boundary = True
        self.last_terminal_state = "explicit_reset"
        return self._last_grid.copy()

    def step(self, action: int) -> np.ndarray:
        from arcengine import GameAction
        self.last_step_was_reset_boundary = False
        self.last_terminal_state = None
        self.level_completed_event = False
        previous_levels = self.last_levels_completed
        raw = self.env.step(GameAction.from_id(int(action)))
        state = _state_name(raw) or "NOT_FINISHED"
        levels = int(getattr(raw, "levels_completed", previous_levels) or 0)
        try:
            self._last_raw = raw
            self._last_grid = _grid_from_raw(raw)
        except ValueError:
            if not self.auto_reset_on_empty_frame:
                raise
            if not _state_name(raw):
                state = "GAME_OVER"
            self.last_outcome_state = state
            self.last_levels_completed = levels
            self.level_completed_event = levels > previous_levels
            self.last_outcome_polarity = _polarity(state, self.level_completed_event)
            self.last_terminal_state = state if state in {"WIN", "GAME_OVER"} else None
            self._last_raw = self.env.reset()
            self.last_outcome_state = state
            self.reset_count += 1
            self.skipped_terminal_steps += 1
            self.last_step_was_reset_boundary = True
            self.last_levels_completed = levels
            self._last_grid = _grid_from_raw(self._last_raw)
            return self._last_grid.copy()
        self.last_outcome_state = state
        self.last_levels_completed = levels
        self.level_completed_event = levels > previous_levels
        self.last_outcome_polarity = _polarity(state, self.level_completed_event)
        self.last_terminal_state = state if state in {"WIN", "GAME_OVER"} else None
        return self._last_grid.copy()

    def available_actions(self) -> list[int]:
        actions = getattr(self._last_raw, "available_actions", None)
        if actions:
            return [int(action) for action in actions]
        if hasattr(self.env, "available_actions"):
            return [int(action) for action in self.env.available_actions()]
        return []


def _ensure_arc_paths() -> None:
    import sys
    repo_root = Path(__file__).resolve().parents[3]
    for relative in ("other_repos/arc-agi", "other_repos/ARCEngine"):
        path = str(repo_root / relative)
        if path not in sys.path:
            sys.path.insert(0, path)


def registered_game_ids(env_root: str | None = None) -> tuple[str, ...]:
    game_ids: set[str] = set()
    for root in _candidate_environment_roots(env_root):
        if not root.exists() or not root.is_dir():
            continue
        for game_dir in root.iterdir():
            if game_dir.is_dir() and any(game_dir.glob("*/metadata.json")):
                game_ids.add(game_dir.name)
    return tuple(sorted(game_ids))


def _candidate_environment_roots(env_root: str | None = None) -> tuple[Path, ...]:
    repo_root = Path(__file__).resolve().parents[3]
    roots: list[Path] = []
    if env_root:
        roots.append(Path(env_root))
    if os.environ.get("ENVIRONMENTS_DIR"):
        roots.append(Path(os.environ["ENVIRONMENTS_DIR"]))
    roots.extend((repo_root / "other_repos" / "arc-interactive" / "environment_files", repo_root / "environment_files"))
    unique: list[Path] = []
    for root in roots:
        root = root.expanduser()
        if root not in unique:
            unique.append(root)
    return tuple(unique)


def _grid_from_raw(raw: Any) -> np.ndarray:
    frame = getattr(raw, "frame", raw)
    if isinstance(frame, list):
        if not frame:
            raise ValueError("raw frame list is empty")
        frame = frame[0]
    array = np.asarray(frame, dtype=int)
    if array.ndim != 2:
        raise ValueError(f"expected 2D ARC grid frame, got shape {array.shape}")
    return array


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
