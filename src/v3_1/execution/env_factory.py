from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

ACTION_NAME_BY_ID = {
    0: "reset",
    1: "up",
    2: "down",
    3: "left",
    4: "right",
    5: "interact",
    6: "click_at",
    7: "undo",
}

ACTION_ID_BY_NAME = {
    "reset": 0,
    "restarts": 0,
    "up": 1,
    "down": 2,
    "left": 3,
    "right": 4,
    "interact": 5,
    "click_at": 6,
    "undo": 7,
    "action1": 1,
    "action2": 2,
    "action3": 3,
    "action4": 4,
    "action5": 5,
    "action6": 6,
    "action7": 7,
}

ACTION_FAMILY_BY_NAME = {
    "up": "move",
    "down": "move",
    "left": "move",
    "right": "move",
    "interact": "interact",
    "click_at": "click_at",
    "undo": "undo",
    "reset": "reset",
}

GRID_DELTA_BY_ACTION_NAME = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_safe_scalar(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_scalar(item) for key, item in value.items()}
    return str(value)


def _discrete_actions(space: object) -> list[dict]:
    if space is None:
        return [normalize_action_lookup(idx) for idx in range(0, 8)]
    if hasattr(space, "n"):
        return [normalize_action_lookup(idx) for idx in range(int(space.n))]
    if isinstance(space, (list, tuple)):
        out = []
        for idx, value in enumerate(space):
            row = normalize_action_lookup(value, available_actions=space)
            row["raw"] = value
            if row["action_id"] is None:
                row["action_id"] = idx
            out.append({"id": row["action_id"], "name": row["action_name"], "family": row["action_family"], "raw": value})
        return out
    row = normalize_action_lookup(0)
    return [{"id": row["action_id"], "name": row["action_name"], "family": row["action_family"]}]


def normalize_action_lookup(action: object, available_actions: object | None = None) -> dict[str, Any]:
    raw_id = None
    raw_name = ""
    if isinstance(action, dict):
        if isinstance(action.get("id"), int):
            raw_id = int(action["id"])
        if action.get("name") is not None:
            raw_name = str(action.get("name")).strip().lower()
    elif isinstance(action, int):
        raw_id = int(action)
    elif isinstance(action, str):
        raw_name = action.strip().lower()
    else:
        action_name = getattr(action, "name", None)
        action_value = getattr(action, "value", None)
        if action_name is not None:
            raw_name = str(action_name).strip().lower()
        if isinstance(action_value, int):
            raw_id = int(action_value)
    if raw_id is None and isinstance(available_actions, list):
        for idx, row in enumerate(available_actions):
            if isinstance(row, dict) and row.get("raw") == action:
                raw_id = int(row.get("id", idx)) if isinstance(row.get("id"), int) else idx
                if row.get("name") is not None:
                    raw_name = str(row.get("name")).strip().lower()
                break
    if raw_id is None and raw_name in ACTION_ID_BY_NAME:
        raw_id = ACTION_ID_BY_NAME[raw_name]
    normalized_name = ACTION_NAME_BY_ID.get(raw_id, raw_name or ("reset" if raw_id == 0 else "unknown"))
    normalized_family = ACTION_FAMILY_BY_NAME.get(normalized_name, "unknown")
    return {
        "action_id": raw_id,
        "action_name": normalized_name if normalized_name else "unknown",
        "action_family": normalized_family,
    }


@dataclass
class NullEnv:
    step_count: int = 0
    width: int = 6
    height: int = 4
    avatar: tuple[int, int] = (1, 1)
    poi: tuple[int, int] = (4, 1)
    done_after: int = 8

    @property
    def action_space(self):
        class _Space:
            n = 6
        return _Space()

    def reset(self, seed: int | None = None):
        del seed
        self.step_count = 0
        self.avatar = (1, 1)
        return self._grid(), {"env": "null"}

    def step(self, action):
        self.step_count += 1
        normalized = normalize_action_lookup(action, available_actions=self.available_actions())
        action_name = str(normalized.get("action_name") or "unknown").lower()
        x, y = self.avatar
        if action_name in GRID_DELTA_BY_ACTION_NAME:
            dx, dy = GRID_DELTA_BY_ACTION_NAME[action_name]
            x += dx
            y += dy
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        self.avatar = (x, y)
        reached = self.avatar == self.poi
        reward = 1.0 if reached else 0.0
        done = reached or self.step_count >= self.done_after
        info = {"avatar": self.avatar, "poi": self.poi, "available_actions": self.available_actions()}
        return self._grid(), reward, done, False, info

    def available_actions(self):
        return _discrete_actions(self.action_space)

    def _grid(self):
        grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        px, py = self.poi
        ax, ay = self.avatar
        grid[py][px] = 2
        grid[ay][ax] = 1
        return grid


@dataclass
class NormalizedEnvAdapter:
    env: object
    env_id: str | None = None
    env_root: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        if hasattr(self.env, "reset"):
            try:
                result = self.env.reset(seed=seed) if seed is not None else self.env.reset()
            except TypeError:
                result = self.env.reset()
            observation, info = self._normalize_reset_result(result)
            info["available_actions"] = self._normalize_available_actions(info.get("available_actions"))
            info.setdefault("env_id", self.env_id)
            return observation, info
        observation = None
        return observation, {"available_actions": self.available_actions(), "env_id": self.env_id}

    def step(self, action: object) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if hasattr(self.env, "step"):
            result = self.env.step(self._normalize_action(action))
            observation, reward, done, truncated, info = self._normalize_step_result(result)
            info["available_actions"] = self._normalize_available_actions(info.get("available_actions"))
            return observation, float(reward or 0.0), bool(done), bool(truncated), info
        return None, 0.0, True, False, {"available_actions": self.available_actions()}

    def available_actions(self) -> list[dict]:
        if hasattr(self.env, "available_actions"):
            actions = self.env.available_actions()
            if isinstance(actions, list):
                normalized = []
                for idx, action in enumerate(actions):
                    row = normalize_action_lookup(action, available_actions=actions)
                    action_id = row["action_id"] if isinstance(row.get("action_id"), int) else idx
                    normalized.append({"id": action_id, "name": row["action_name"], "family": row["action_family"]})
                return normalized
        return _discrete_actions(getattr(self.env, "action_space", None))

    def env_metadata(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "env_root": self.env_root,
            "available_action_count": len(self.available_actions()),
            **self.metadata,
        }

    def _normalize_reset_result(self, result: object) -> tuple[Any, dict[str, Any]]:
        if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], dict):
            return self._normalize_observation(result[0]), dict(result[1])
        info = self._frame_info(result)
        return self._normalize_observation(result), info

    def _normalize_step_result(self, result: object) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if isinstance(result, tuple) and len(result) == 5:
            observation, reward, done, truncated, info = result
            return self._normalize_observation(observation), float(reward or 0.0), bool(done), bool(truncated), dict(info or {})
        if isinstance(result, tuple) and len(result) == 4:
            observation, reward, done, info = result
            return self._normalize_observation(observation), float(reward or 0.0), bool(done), False, dict(info or {})
        info = self._frame_info(result)
        observation = self._normalize_observation(result)
        reward = float(info.get("reward", 0.0) or 0.0)
        done = bool(info.get("done", False))
        truncated = bool(info.get("truncated", False))
        return observation, reward, done, truncated, info

    def _normalize_observation(self, observation: object) -> Any:
        if observation is None:
            return None
        frame = getattr(observation, "frame", observation)
        if isinstance(frame, list):
            if frame and hasattr(frame[0], "tolist"):
                return frame[0].tolist()
            return frame
        if hasattr(frame, "tolist"):
            return frame.tolist()
        return observation

    def _frame_info(self, frame: object) -> dict[str, Any]:
        if frame is None:
            return {}
        available_actions = []
        raw_actions = getattr(frame, "available_actions", None)
        if isinstance(raw_actions, list):
            available_actions = [int(action) for action in raw_actions if isinstance(action, int)]
        done = self._is_terminal_state(getattr(frame, "state", None))
        won = bool(getattr(frame, "win", False))
        reward = float(getattr(frame, "reward", 0.0) or 0.0)
        if won and reward <= 0.0:
            reward = 1.0
        info = {
            "available_actions": available_actions,
            "done": done,
            "win": won,
            "reward": reward,
            "levels_completed": int(getattr(frame, "levels_completed", 0) or 0),
            "win_levels": int(getattr(frame, "win_levels", 0) or 0),
            "state": str(getattr(getattr(frame, "state", None), "name", getattr(frame, "state", ""))),
        }
        model_dump = getattr(frame, "model_dump", None)
        if callable(model_dump):
            info["raw"] = _safe_scalar(model_dump())
        return info

    def _normalize_available_actions(self, actions: object) -> list[dict]:
        if isinstance(actions, list):
            normalized = []
            for idx, action in enumerate(actions):
                row = self._action_row_from_value(action, fallback_id=idx)
                normalized.append(row)
            return normalized
        return self.available_actions()

    def _is_terminal_state(self, state: object) -> bool:
        if state is None:
            return False
        normalized = {
            str(state).upper(),
            str(getattr(state, "name", "")).upper(),
            str(getattr(state, "value", "")).upper(),
        }
        if "NOT_FINISHED" in normalized or "GAMESTATE.NOT_FINISHED" in normalized:
            return False
        return True

    def _normalize_action(self, action: object) -> object:
        try:
            from arcengine import GameAction
        except Exception:
            return action
        if isinstance(action, GameAction):
            return action
        if isinstance(action, dict):
            raw = action.get("raw")
            if isinstance(raw, GameAction):
                return raw
            if isinstance(action.get("name"), str):
                try:
                    return GameAction.from_name(str(action["name"]))
                except Exception:
                    pass
            if isinstance(action.get("id"), int):
                try:
                    return GameAction.from_id(int(action["id"]))
                except Exception:
                    pass
        if isinstance(action, int):
            try:
                return GameAction.from_id(action)
            except Exception:
                return action
        return action

    def _action_row_from_value(self, action: object, *, fallback_id: int) -> dict[str, Any]:
        row = normalize_action_lookup(action)
        action_id = row["action_id"] if isinstance(row.get("action_id"), int) else fallback_id
        return {"id": action_id, "name": row["action_name"], "family": row["action_family"]}


def build_env(factory_path: str | None, *, env_id: str | None = None, env_root: str | None = None, seed: int | None = None, render_terminal: bool = False) -> NormalizedEnvAdapter:
    if factory_path is None:
        return NormalizedEnvAdapter(env=NullEnv(), env_id=env_id, env_root=env_root, metadata={"factory": "null"})
    module_name, func_name = factory_path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, func_name)
    try:
        env = factory(env_id=env_id, env_root=env_root, seed=seed, render_terminal=render_terminal)
    except TypeError:
        env = factory()
    return NormalizedEnvAdapter(env=env, env_id=env_id, env_root=env_root, metadata={"factory": factory_path, "seed": seed, "render_terminal": render_terminal})
