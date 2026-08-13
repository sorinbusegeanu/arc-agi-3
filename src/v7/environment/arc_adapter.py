from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import os

import numpy as np


class ArcGridEnvironment:
    """Clean v7 adapter around the official ARC-AGI-3 SDK."""

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
        self._arcade = None
        if env_factory is None:
            import arc_agi
            from arc_agi import OperationMode

            modes = {
                "normal": OperationMode.NORMAL,
                "online": OperationMode.ONLINE,
                "offline": OperationMode.OFFLINE,
                "competition": OperationMode.COMPETITION,
            }
            mode = modes.get(str(op_mode).lower())
            if mode is None:
                raise ValueError(f"unknown ARC operation mode: {op_mode}")
            kwargs: dict[str, object] = {"operation_mode": mode}
            if env_root:
                kwargs["environments_dir"] = str(env_root)
            self._arcade = arc_agi.Arcade(**kwargs)
            self.env = self._arcade.make(game_id, seed=int(seed), include_frame_data=True, render_mode=render_mode)
            if self.env is None:
                raise RuntimeError(f"ARC-AGI environment could not be created: {game_id}")
        else:
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
        if self._last_raw is None:
            raise RuntimeError(f"ARC-AGI environment reset failed: {game_id}")
        self._update_from_raw(self._last_raw, reset_boundary=False)

    def observe(self) -> np.ndarray:
        return self._last_grid.copy()

    def reset(self) -> np.ndarray:
        self._last_raw = self.env.reset()
        if self._last_raw is None:
            raise RuntimeError("ARC-AGI environment reset failed")
        self.reset_count += 1
        self.last_step_was_reset_boundary = True
        self.last_terminal_state = "explicit_reset"
        self._update_from_raw(self._last_raw, reset_boundary=True)
        return self._last_grid.copy()

    def step(self, action: int, *, data: dict[str, int] | None = None) -> np.ndarray:
        try:
            from arcengine import GameAction
            game_action = _game_action_from_id(GameAction, int(action))
            raw = self.env.step(game_action, data=data or {})
        except ImportError:
            raw = self.env.step(int(action))
        if raw is None:
            raise RuntimeError("ARC-AGI environment step returned no observation")
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
            if self._last_raw is None:
                raise RuntimeError("ARC-AGI environment reset failed after terminal frame")
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
        action_space = getattr(self.env, "action_space", None)
        if action_space:
            return sorted({_action_id(action) for action in action_space})
        actions = getattr(self._last_raw, "available_actions", None)
        if actions:
            return sorted({_action_id(action) for action in actions})
        method = getattr(self.env, "available_actions", None)
        return [] if method is None else sorted({_action_id(action) for action in method()})

    def action_data(self, action: int, *, rng: Any | None = None) -> dict[str, int]:
        """Return valid default data for complex ARC actions."""
        try:
            from arcengine import GameAction
            game_action = _game_action_from_id(GameAction, int(action))
            if not bool(game_action.is_complex()):
                return {}
        except (ImportError, AttributeError):
            return {}
        height, width = self._last_grid.shape
        if rng is None:
            x, y = width // 2, height // 2
        else:
            x = int(rng.randrange(max(1, width)))
            y = int(rng.randrange(max(1, height)))
        return {"x": x, "y": y}

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
        for metadata in root.rglob("metadata.json"):
            try:
                relative = metadata.relative_to(root)
                if relative.parts:
                    game_ids.add(relative.parts[0])
            except ValueError:
                continue
    return tuple(sorted(game_ids))


def _candidate_environment_roots(env_root: str | None) -> tuple[Path, ...]:
    repo_root = Path(__file__).resolve().parents[3]
    values = [
        env_root,
        os.environ.get("ENVIRONMENTS_DIR"),
        str(repo_root / "environment_files"),
        str(repo_root / "other_repos" / "arc-interactive" / "environment_files"),
    ]
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
    if isinstance(frame, (list, tuple)):
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
    value = getattr(state, "name", None) or getattr(state, "value", state)
    return str(value).upper()


def _action_id(action: Any) -> int:
    value = getattr(action, "value", action)
    try:
        return int(value)
    except (TypeError, ValueError):
        getter = getattr(action, "get_id", None)
        if getter is not None:
            return int(getter())
        raise


def _game_action_from_id(game_action_cls: Any, action_id: int) -> Any:
    from_id = getattr(game_action_cls, "from_id", None)
    if from_id is not None:
        return from_id(int(action_id))
    try:
        return game_action_cls(int(action_id))
    except Exception:
        for action in game_action_cls:
            if _action_id(action) == int(action_id):
                return action
        raise ValueError(f"unknown ARC action id: {action_id}")


def _polarity(state: str, level_completed: bool) -> str:
    if state == "WIN" or level_completed:
        return "positive"
    if state == "GAME_OVER":
        return "negative"
    return "neutral"
