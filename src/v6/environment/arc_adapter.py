from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


class ArcGridEnvironment:
    """Adapts a local ARC environment to the v6 integer-grid contract."""

    def __init__(
        self,
        *,
        game_id: str,
        seed: int = 0,
        env_root: str | None = None,
        op_mode: str = "normal",
        render_mode: str | None = None,
        auto_reset_on_empty_frame: bool = True,
    ) -> None:
        _ensure_arc_paths()
        from arc_agi_agent.envs.loader import make_env

        self.env = make_env(
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
        self.last_outcome_state: str = "alive"
        self.last_outcome_polarity: str = "neutral"
        self.last_level_number: int | None = None
        self.level_advanced: bool = False
        self._last_raw = self.env.reset()
        self.last_level_number = _level_number(self._last_raw)
        self._last_grid = _grid_from_raw(self._last_raw)

    def observe(self) -> np.ndarray:
        return self._last_grid.copy()

    def reset(self) -> np.ndarray:
        raw = self.env.reset()
        self._last_raw = raw
        self.last_level_number = _level_number(raw)
        self.level_advanced = False
        self._last_grid = _grid_from_raw(self._last_raw)
        self.reset_count += 1
        self.last_step_was_reset_boundary = True
        self.last_terminal_state = "explicit_reset"
        self.last_outcome_state = "alive"
        self.last_outcome_polarity = "neutral"
        return self._last_grid.copy()

    def step(self, action: int) -> np.ndarray:
        from arcengine import GameAction

        self.last_step_was_reset_boundary = False
        self.last_terminal_state = None
        self.level_advanced = False
        previous_level_number = self.last_level_number
        self._last_raw = self.env.step(GameAction.from_id(int(action)))
        current_level_number = _level_number(self._last_raw)
        try:
            self._last_grid = _grid_from_raw(self._last_raw)
            frame_available = True
        except ValueError:
            if not self.auto_reset_on_empty_frame:
                raise
            frame_available = False
            self.last_outcome_state = "end_game"
            self.last_outcome_polarity = "unknown"
            self.last_terminal_state = "end_game"
            raw_after_terminal = self._last_raw
            self._last_raw = self.env.reset()
            self.last_level_number = _level_number(self._last_raw)
            self.level_advanced = False
            self.reset_count += 1
            self.skipped_terminal_steps += 1
            self.last_step_was_reset_boundary = True
            self._last_grid = _grid_from_raw(self._last_raw)
            self._last_raw = raw_after_terminal
            return self._last_grid.copy()
        state_name = (_state_name(self._last_raw) or "").upper()
        if state_name in {"GAME_OVER", "DEAD"}:
            self.last_outcome_state = "dead"
            self.last_outcome_polarity = "negative"
            self.last_terminal_state = state_name or "GAME_OVER"
        elif state_name in {"WIN", "GAME_WON"}:
            self.last_outcome_state = "game_won"
            self.last_outcome_polarity = "positive"
            self.last_terminal_state = state_name or "WIN"
        elif previous_level_number is not None and current_level_number is not None and current_level_number != previous_level_number:
            self.last_outcome_state = "level_advanced"
            self.last_outcome_polarity = "positive"
            self.level_advanced = True
        elif not frame_available:
            self.last_outcome_state = "end_game"
            self.last_outcome_polarity = "unknown"
            self.last_terminal_state = "end_game"
        else:
            self.last_outcome_state = "alive"
            self.last_outcome_polarity = "neutral"
        self.last_level_number = current_level_number
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
            if not game_dir.is_dir():
                continue
            if any(game_dir.glob("*/metadata.json")):
                game_ids.add(game_dir.name)
    return tuple(sorted(game_ids))


def _candidate_environment_roots(env_root: str | None = None) -> tuple[Path, ...]:
    roots: list[Path] = []
    if env_root:
        roots.append(Path(env_root))
    current = os.environ.get("ENVIRONMENTS_DIR")
    if current:
        roots.append(Path(current))
    roots.extend(_default_environment_roots())
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return tuple(unique)


def _default_environment_roots() -> tuple[Path, ...]:
    repo_root = Path(__file__).resolve().parents[3]
    return (
        repo_root / "other_repos" / "arc-interactive" / "environment_files",
        repo_root / "environment_files",
    )


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


def _level_number(raw: Any) -> int | None:
    for name in ("level", "level_number", "level_id", "levels_completed"):
        if hasattr(raw, name):
            try:
                return int(getattr(raw, name))
            except (TypeError, ValueError):
                pass
    return None


def _raw_has_usable_frame(raw: Any) -> bool:
    frame = getattr(raw, "frame", raw)
    if isinstance(frame, list):
        if not frame:
            return False
        frame = frame[0]
    try:
        array = np.asarray(frame)
    except Exception:
        return False
    return bool(array.ndim == 2 and array.size > 0)


def _infer_outcome_fields(raw: Any, *, frame_available: bool) -> tuple[str, str]:
    state_name = (_state_name(raw) or "").upper()
    if state_name == "WIN":
        return "game_won", "positive"
    if state_name == "GAME_OVER":
        return "dead", "negative"
    if frame_available:
        return "alive", "neutral"
    return "end_game", "unknown"
