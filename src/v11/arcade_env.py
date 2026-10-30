"""Normalize native ARC-AGI environments for the v10 game tools."""

from __future__ import annotations

import copy
from typing import Any, Callable, Mapping

import numpy as np


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _state_name(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "name", value)).split(".")[-1]


class ArcadeGameEnv:
    """Expose ARC Arcade reset/step observations as GameTools observe/act calls."""

    def __init__(
        self,
        game_id: str,
        seed: int = 0,
        env_root: str | None = None,
        env_factory: Callable[..., Any] | None = None,
    ) -> None:
        if env_factory is None:
            from arc_agi_agent.envs.loader import make_env

            env_factory = make_env
        self.game_id = str(game_id)
        self._env_factory = env_factory
        self._env_root = env_root
        self._env = self._env_factory(self.game_id, env_root=env_root, seed=int(seed), op_mode="offline")
        self._game_action = None
        self._raw_observation: Any = None
        self._seed = int(seed)
        self._raw_observation = self._env.reset()
        if self._raw_observation is None:
            raise RuntimeError(f"ARC environment reset returned no observation for {self.game_id}")

    @staticmethod
    def _frames(frame: Any) -> list[np.ndarray]:
        array = np.asarray(frame)
        if array.ndim == 2:
            return [array]
        if array.ndim == 3 and array.shape[0] > 0:
            return [array[index] for index in range(array.shape[0])]
        raise ValueError(f"ARC observation must contain 2-D grids, got {array.shape}")

    @staticmethod
    def _grid(frame: Any) -> np.ndarray:
        return ArcadeGameEnv._frames(frame)[-1]

    @staticmethod
    def _action_enum() -> Any:
        from arcengine import GameAction

        return GameAction

    def _observation(self, raw: Any) -> dict[str, Any]:
        frame = _field(raw, "frame")
        if frame is None:
            raise ValueError("ARC observation is missing frame")
        available = _field(raw, "available_actions", ()) or ()
        action_enum = self._action_enum()
        legal_actions = []
        for raw_id in available:
            action_id = int(getattr(raw_id, "value", raw_id))
            action = action_enum.from_id(action_id)
            parameters = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
            if str(action.name) == "ACTION6":
                height, width = self._grid(frame).shape
                parameters["properties"] = {
                    "x": {"type": "integer", "minimum": 0, "maximum": width - 1},
                    "y": {"type": "integer", "minimum": 0, "maximum": height - 1},
                }
                parameters["required"] = ["x", "y"]
            legal_actions.append({
                "id": str(action.name), "_environment_id": action_id,
                "parameters": parameters,
            })
        state = _state_name(_field(raw, "state"))
        levels_completed = _field(raw, "levels_completed")
        return {
            "frame": self._grid(frame),
            "legal_actions": legal_actions,
            "game_id": self.game_id,
            "game_state": state,
            "level": _field(raw, "level_index", levels_completed),
            "levels_completed": levels_completed,
            "win_levels": _field(raw, "win_levels"),
            "terminal": state in {"WIN", "GAME_OVER", "LOSE", "LOST"},
        }

    def observe(self) -> dict[str, Any]:
        if self._raw_observation is None:
            raise RuntimeError("ARC environment has no current observation")
        return self._observation(self._raw_observation)

    def act(self, action_id: int | str, args: Mapping[str, Any] | None = None) -> dict[str, Any]:
        action_enum = self._action_enum()
        action = action_enum.from_id(int(action_id))
        data = {"game_id": self.game_id, **dict(args or {})}
        self._raw_observation = self._env.step(action, data=data, reasoning=None)
        if self._raw_observation is None:
            raise RuntimeError("ARC environment step returned no observation")
        return {"state": _state_name(_field(self._raw_observation, "state")),
                "frames": self._frames(_field(self._raw_observation, "frame"))[:-1]}

    def reset(self) -> dict[str, Any]:
        self._raw_observation = self._env.reset()
        if self._raw_observation is None:
            raise RuntimeError("ARC environment reset returned no observation")
        return self.observe()

    def fork(self) -> "ArcadeGameEnv":
        """Create an isolated environment with the same game factory and seed."""
        return ArcadeGameEnv(
            self.game_id, seed=self._seed, env_root=self._env_root,
            env_factory=self._env_factory,
        )

    def fork_current(self) -> "ArcadeGameEnv":
        """Clone this exact live state for an isolated discovery branch."""
        clone = object.__new__(ArcadeGameEnv)
        clone.game_id = self.game_id
        clone._seed = self._seed
        clone._env_root = self._env_root
        clone._env_factory = self._env_factory
        clone._env = copy.deepcopy(self._env)
        clone._raw_observation = copy.deepcopy(self._raw_observation)
        return clone

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()


__all__ = ["ArcadeGameEnv"]
