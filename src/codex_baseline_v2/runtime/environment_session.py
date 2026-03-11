from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from codex_baseline_v2.shared.schemas import ActionDescriptorV2, SCHEMA_VERSION


@dataclass(frozen=True)
class StepResultV2:
    observation: Optional[List[List[int]]]
    reward: float
    done: bool
    info: Dict[str, Any]
    available_actions: Optional[List[int]]


class EnvironmentSessionV2:
    """V2-native environment session wrapper for one live game."""

    def __init__(self, env: Any, game_id: str) -> None:
        self._env = env
        self._game_id = game_id
        self._last_info: Dict[str, Any] = {}

    @property
    def game_id(self) -> str:
        return self._game_id

    def reset(self, seed: Optional[int] = None) -> Optional[List[List[int]]]:
        if seed is not None and hasattr(self._env, "seed"):
            self._env.seed(seed)
        obs = self._env.reset()
        self._last_info = {}
        return self._normalize_observation(obs)

    def current_observation(self) -> Optional[List[List[int]]]:
        obs = getattr(self._env, "last_obs", None)
        if obs is None:
            return None
        return self._normalize_observation(obs)

    def step(self, action: ActionDescriptorV2) -> StepResultV2:
        raw_action = self._to_env_action(action)
        step_out = self._env.step(raw_action)
        if isinstance(step_out, tuple):
            if len(step_out) == 4:
                obs, reward, done, info = step_out
            elif len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated) or bool(truncated)
            else:
                raise RuntimeError("Environment step did not return a supported tuple format")
        elif isinstance(step_out, dict):
            obs = step_out.get("observation") or step_out.get("obs")
            reward = step_out.get("reward", 0.0)
            done = step_out.get("done", False)
            info = step_out.get("info", {})
        elif hasattr(step_out, "frame"):
            obs = step_out.frame
            reward = 0.0
            state = getattr(step_out, "state", None)
            done = self._is_terminal_state(state)
            info = step_out.model_dump() if hasattr(step_out, "model_dump") else {"raw": step_out}
        else:
            raise RuntimeError("Environment step did not return (obs, reward, done, info)")
        self._last_info = info if isinstance(info, dict) else {}
        return StepResultV2(
            observation=self._normalize_observation(obs),
            reward=float(reward) if reward is not None else 0.0,
            done=bool(done),
            info=self._normalize_info(info),
            available_actions=self._extract_available_actions(info),
        )

    def available_actions(self) -> Optional[List[int]]:
        avail = self._extract_available_actions(self._last_info)
        if avail:
            return avail
        action_space = getattr(self._env, "action_space", None)
        n = getattr(action_space, "n", None)
        if isinstance(n, int) and n > 0:
            return [int(i) for i in range(n)]
        return [0, 1, 2, 3, 4, 5, 6, 7]

    def progress_status(self) -> Dict[str, Any]:
        return {
            "done": bool(self._last_info.get("done", False)),
            "win": bool(self._last_info.get("win", False)),
            "reward": float(self._last_info.get("reward", 0.0)) if isinstance(self._last_info, dict) else 0.0,
        }

    def _normalize_observation(self, obs: Any) -> Optional[List[List[int]]]:
        if obs is None:
            return None
        if hasattr(obs, "frame"):
            obs = obs.frame
        if isinstance(obs, list) and obs and hasattr(obs[0], "tolist"):
            frame = obs[0].tolist()
            return [[int(v) for v in row] for row in frame]
        if isinstance(obs, list):
            return [[int(v) for v in row] for row in obs]
        if hasattr(obs, "tolist"):
            grid = obs.tolist()
            if isinstance(grid, list):
                return [[int(v) for v in row] for row in grid]
        return None

    def _is_terminal_state(self, state: Any) -> bool:
        if state is None:
            return False
        value = getattr(state, "value", state)
        name = getattr(state, "name", None)
        state_str = str(value if value is not None else state)
        state_name = str(name) if name is not None else ""
        normalized = {state_str.upper(), state_name.upper(), str(state).upper()}
        if {"NOT_FINISHED", "GAMESTATE.NOT_FINISHED"} & normalized:
            return False
        return True

    def _normalize_info(self, info: Any) -> Dict[str, Any]:
        if not isinstance(info, dict):
            return {}
        safe_raw = {}
        for k, v in info.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                safe_raw[k] = v
            elif isinstance(v, list):
                safe_raw[k] = [x if isinstance(x, (int, float, str, bool)) else str(x) for x in v]
            elif isinstance(v, dict):
                safe_raw[k] = {kk: (vv if isinstance(vv, (int, float, str, bool)) else str(vv)) for kk, vv in v.items()}
            else:
                safe_raw[k] = str(v)
        return {
            "reward": safe_raw.get("reward"),
            "progress": safe_raw.get("progress"),
            "done": safe_raw.get("done"),
            "win": safe_raw.get("win"),
            "raw": safe_raw,
        }

    def _extract_available_actions(self, info: Any) -> Optional[List[int]]:
        if not isinstance(info, dict):
            return None
        avail = info.get("available_actions") or info.get("available_actions_sorted")
        if isinstance(avail, list) and all(isinstance(x, int) for x in avail):
            return [int(x) for x in avail]
        return None

    def _to_env_action(self, action: ActionDescriptorV2) -> Any:
        if action.action_type == "coord" and action.coord is not None:
            return {"type": "coord", "x": int(action.coord[0]), "y": int(action.coord[1])}
        if action.action_id is not None:
            return int(action.action_id)
        return 0


def build_action_descriptor(action_id: Optional[int] = None, coord: Optional[Tuple[int, int]] = None) -> ActionDescriptorV2:
    action_type = "coord" if coord is not None else "discrete"
    return ActionDescriptorV2(
        schema_version=SCHEMA_VERSION,
        action_type=action_type,
        action_id=int(action_id) if action_id is not None else None,
        coord=coord,
        raw=None,
    )
