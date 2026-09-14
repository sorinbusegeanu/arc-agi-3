from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from v9.environments.base import StructuralAdapter, _fact, _semantic_id
from v9.environments.contract import BoundaryEvent, BoundaryScope, TaskProgress, WithinActionFrame, WithinActionTrace
from v9.environments.schemas import ActionSchema, EnvironmentIdentity, ObservationSchema
from v9.memory.identity import stable_u64


def _has_environment(root: Path, game_id: str) -> bool:
    return (root / game_id).is_dir() and any((root / game_id).glob("*/metadata.json"))


def _resolve_root(game_id: str, explicit: str | None) -> str | None:
    repository = Path(__file__).resolve().parents[4]
    candidates = [explicit, os.environ.get("ENVIRONMENTS_DIR"), str(repository / "other_repos/arc-interactive/environment_files"), str(repository / "environment_files")]
    paths = [Path(value).expanduser() for value in candidates if value]
    for path in paths:
        if _has_environment(path, game_id):
            return str(path)
    return str(next((path for path in paths if path.is_dir()), "")) or None


def make_arc_environment(game_id: str, *, seed: int, env_root: str | None = None, op_mode: str = "normal", render_mode: str | None = None) -> Any:
    try:
        from arc_agi import Arcade, OperationMode
    except ImportError as exc:
        raise RuntimeError("ARCAdapter requires the arc-agi environment package") from exc
    resolved = _resolve_root(game_id, env_root)
    if resolved:
        os.environ["ENVIRONMENTS_DIR"] = resolved
    local = bool(resolved and _has_environment(Path(resolved), game_id))
    logger = logging.getLogger("v9.arc.local") if local else None
    arcade = Arcade(operation_mode=OperationMode("offline" if local else op_mode), logger=logger)
    environment = arcade.make(game_id, seed=int(seed), render_mode=render_mode)
    if environment is None:
        raise RuntimeError(f"arcade.make failed for {game_id}")
    return environment


def _grid(raw: Any) -> np.ndarray:
    frame = getattr(raw, "frame", raw)
    if isinstance(frame, list):
        if not frame:
            raise ValueError("ARC returned an empty frame")
        frame = frame[0]
    result = np.asarray(frame, dtype=np.int64)
    if result.ndim != 2 or not result.size:
        raise ValueError("ARC observation must be a non-empty two-dimensional grid")
    return result


def _state(raw: Any) -> str:
    value = getattr(raw, "state", "NOT_FINISHED")
    return str(getattr(value, "value", value))


class ARCAdapter(StructuralAdapter):
    def __init__(self, game_id: str, *, seed: int = 0, env_root: str | None = None, env_factory: Callable[..., Any] | None = None) -> None:
        self.game_id, self.seed, self.env_root = str(game_id), int(seed), env_root
        self._env_factory = env_factory or make_arc_environment
        self.env = self._env_factory(self.game_id, seed=self.seed, env_root=env_root)
        self._identity = EnvironmentIdentity("arc", self.game_id, "default", f"seed={seed}")
        self._observation_schema = ObservationSchema("grid", "arc-color-grid")
        self._action_schema = ActionSchema("environment-local", "arc-native-actions")
        self._raw = None
        self._observation = np.zeros((1, 1), dtype=np.int64)
        self._boundary = BoundaryEvent()
        self._last_trace = None
        self._levels = 0
        self._last_state = "NOT_FINISHED"
        self._action_history: list[int] = []
        self.reset()

    def reset(self) -> np.ndarray:
        self._raw = self.env.reset()
        self._observation = _grid(self._raw)
        self._levels = int(getattr(self._raw, "levels_completed", 0) or 0)
        self._last_state = _state(self._raw)
        self._boundary = BoundaryEvent()
        self._last_trace = None
        self._action_history = []
        return self.observe()

    def observe(self) -> np.ndarray:
        return self._observation.copy()

    def semantic_observation(self, observation: Any):
        grid = np.asarray(observation, dtype=np.int64)
        if grid.ndim != 2 or not grid.size:
            return ()
        height, width = grid.shape
        seen = np.zeros_like(grid, dtype=np.bool_)
        facts = []
        for row in range(height):
            for col in range(width):
                color = int(grid[row, col])
                if color == 0 or bool(seen[row, col]):
                    continue
                stack = [(row, col)]
                seen[row, col] = True
                cells = []
                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))
                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                min_r = min(r for r, _ in cells)
                max_r = max(r for r, _ in cells)
                min_c = min(c for _, c in cells)
                max_c = max(c for _, c in cells)
                shape_signature = int(stable_u64(tuple(sorted((r - min_r, c - min_c) for r, c in cells)), person=b"v9-arc-shape"))
                entity = _semantic_id(f"arc:{color}:{min_r}:{min_c}:{shape_signature}")
                facts.append(_fact(2, entity, 6, color, float(color)))
                facts.append(_fact(3, entity, 1, "area", float(len(cells))))
                facts.append(_fact(5, entity, 2, min_r * 4096 + min_c, 1.0))
                facts.append(_fact(5, entity, 3, max_r, float(max_r)))
                facts.append(_fact(5, entity, 4, max_c, float(max_c)))
                facts.append(_fact(3, entity, 1, shape_signature, float(len(cells))))
        for color in sorted(int(value) for value in np.unique(grid) if int(value) != 0):
            count = int(np.count_nonzero(grid == color))
            facts.append(_fact(1, f"arc-color:{color}", 1, color, float(count)))
        return tuple(facts)

    def available_actions(self) -> tuple[int, ...]:
        values = getattr(self._raw, "available_actions", None)
        if values is None:
            callback = getattr(self.env, "available_actions", None)
            values = () if callback is None else callback()
        return tuple(int(value) for value in values or ())

    def step(self, native_action: Any) -> np.ndarray:
        before = self.observe()
        encoded_action = int(native_action)
        try:
            from arcengine import GameAction
            action = GameAction.from_id(encoded_action)
        except ImportError:
            action = encoded_action
        raw = self.env.step(action)
        self._action_history.append(encoded_action)
        state = _state(raw)
        self._last_state = state
        levels = int(getattr(raw, "levels_completed", self._levels) or 0)
        advanced = levels > self._levels
        self._levels = levels
        try:
            after = _grid(raw)
        except ValueError:
            after = _grid(self.env.reset())
        self._raw, self._observation = raw, after
        if state == "WIN":
            self._boundary = BoundaryEvent(BoundaryScope.EPISODE, 1, False)
        elif state == "GAME_OVER":
            self._boundary = BoundaryEvent(BoundaryScope.EPISODE, -1, False)
        elif advanced:
            self._boundary = BoundaryEvent(BoundaryScope.SUBEPISODE, 1, True)
        else:
            self._boundary = BoundaryEvent()
        self._last_trace = WithinActionTrace(before, (WithinActionFrame(after.copy(), 0),), after.copy())
        return self.observe()

    def task_progress(self) -> TaskProgress:
        terminal = self._last_state in {"WIN", "GAME_OVER"}
        return TaskProgress(
            game_id=self.game_id,
            level_id=f"level-{self._levels + (0 if terminal else 1)}",
            level_index=int(self._levels + (0 if terminal else 1)),
            levels_completed=int(self._levels),
            terminal=terminal,
            success=self._last_state == "WIN",
            failure=self._last_state == "GAME_OVER",
            truncated=False,
            score=float(self._levels),
        )

    def capture_state(self) -> dict[str, object]:
        """Capture a reproducible ARC state as seed plus exact action replay."""
        return {
            "schema_version": 1,
            "actions": tuple(self._action_history),
            "observation_signature": int(self.encode_observation(self._observation)),
            "available_actions": tuple(self.available_actions()),
            "levels_completed": int(self._levels),
        }

    def restore_state(self, state: Any) -> None:
        if not isinstance(state, dict) or int(state.get("schema_version", 0)) != 1:
            raise ValueError("unsupported ARC replay snapshot")
        actions = tuple(int(value) for value in state.get("actions", ()))
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
        self.env = self._env_factory(self.game_id, seed=self.seed, env_root=self.env_root)
        self.reset()
        for action in actions:
            if action not in self.available_actions():
                raise RuntimeError("ARC replay snapshot action is no longer available")
            self.step(action)
        if int(self.encode_observation(self._observation)) != int(state["observation_signature"]):
            raise RuntimeError("ARC replay snapshot did not reproduce the captured observation")
        if tuple(self.available_actions()) != tuple(int(value) for value in state["available_actions"]):
            raise RuntimeError("ARC replay snapshot did not reproduce available actions")
        if int(self._levels) != int(state["levels_completed"]):
            raise RuntimeError("ARC replay snapshot did not reproduce level progress")

    def close(self) -> None:
        callback = getattr(self.env, "close", None)
        if callable(callback):
            callback()
