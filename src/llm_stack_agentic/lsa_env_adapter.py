from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Protocol, Tuple

from .lsa_bootstrap_explorer_types import AvailableActionsV1, ObsV1


class EnvAdapter(Protocol):
    def reset(self, episode_id: str, game_id: str, seed: int) -> Any:
        ...

    def step(self, action: Dict[str, Any]) -> Tuple[Any, Optional[float], bool, Any]:
        ...

    def get_available_actions(self, raw_obs: Any) -> Dict[str, Any]:
        ...

    def to_canonical_obs(self, raw_obs: Any) -> Dict[str, Any]:
        ...


class ArcAgiDefaultAdapter:
    def __init__(
        self,
        *,
        op_mode: str = "offline",
        env_dir: Optional[str] = None,
        render_mode: Optional[str] = None,
        save_recording: bool = False,
        scorecard_id: Optional[str] = None,
        quiet: bool = True,
    ) -> None:
        arc_agi_path = "/home/zodrak/zod/other_repos/arc-agi"
        arcengine_path = "/home/zodrak/zod/other_repos/ARCEngine"
        arcengine_pkg = os.path.join(arcengine_path, "arcengine")
        for path in (arc_agi_path, arcengine_path, arcengine_pkg):
            if path not in sys.path:
                sys.path.insert(0, path)

        from arc_agi import Arcade, OperationMode

        if env_dir is not None:
            os.environ["ENVIRONMENTS_DIR"] = env_dir
        elif "ENVIRONMENTS_DIR" not in os.environ:
            os.environ["ENVIRONMENTS_DIR"] = "/home/zodrak/zod/environment_files"

        logger = None
        if quiet:
            import logging

            logger = logging.getLogger("llm_stack_agentic.arcade_quiet")
            logger.handlers.clear()
            logger.addHandler(logging.NullHandler())
            logger.propagate = False
            logger.setLevel(logging.ERROR)

        self._arcade = Arcade(operation_mode=OperationMode(op_mode), logger=logger)
        self._env = None
        self._game_id = None
        self._seed = 0
        self._render_mode = render_mode
        self._save_recording = bool(save_recording)
        self._scorecard_id = scorecard_id

        from arcengine import GameAction, GameState

        self._game_action_enum = GameAction
        self._game_state_enum = GameState
        self._action_ids = [int(a.value) for a in GameAction]
        self._max_action_id = max(self._action_ids) if self._action_ids else 0

    def reset(self, episode_id: str, game_id: str, seed: int) -> Any:
        self._game_id = game_id
        self._seed = int(seed)
        self._env = self._arcade.make(
            game_id,
            seed=seed,
            scorecard_id=self._scorecard_id,
            save_recording=self._save_recording,
            render_mode=self._render_mode,
        )
        if self._env is None:
            raise RuntimeError(f"Failed to create environment for {game_id}")
        raw_obs = self._env.reset()
        if raw_obs is None:
            raise RuntimeError("Environment reset returned None")
        return raw_obs

    def step(self, action: Dict[str, Any]) -> Tuple[Any, Optional[float], bool, Any]:
        if self._env is None:
            raise RuntimeError("Environment not initialized; call reset() first")
        action_type = str(action.get("type", "discrete"))
        action_id = int(action.get("id", 0))
        if action_type == "coord":
            data = {"x": int(action.get("x", 0)), "y": int(action.get("y", 0)), "game_id": self._game_id or ""}
        else:
            data = {"game_id": self._game_id or ""}
        game_action = self._game_action_enum.from_id(action_id)
        raw_obs = self._env.step(game_action, data=data)
        if raw_obs is None:
            raise RuntimeError("Environment step returned None")
        state = getattr(raw_obs, "state", None)
        done = bool(
            state in (self._game_state_enum.WIN, self._game_state_enum.GAME_OVER)
            or str(state) in ("GameState.WIN", "WIN", "GameState.GAME_OVER", "GAME_OVER")
        )
        info = {
            "available_actions": getattr(raw_obs, "available_actions", None),
            "guid": getattr(raw_obs, "guid", None),
            "state": state,
        }
        reward = getattr(raw_obs, "reward", None)
        return raw_obs, reward, bool(done), info

    def get_available_actions(self, raw_obs: Any) -> Dict[str, Any]:
        available = list(getattr(raw_obs, "available_actions", []) or [])
        mask = [False] * (self._max_action_id + 1)
        for action_id in available:
            if 0 <= int(action_id) < len(mask):
                mask[int(action_id)] = True
        coord_id = self._coord_action_id(available)
        return AvailableActionsV1(
            discrete_mask=mask,
            coord_enabled=coord_id is not None,
            coord_action_id=coord_id,
            coord_bounds=None,
            allowed_coords=None,
            tags=None,
        ).to_dict()

    def to_canonical_obs(self, raw_obs: Any) -> Dict[str, Any]:
        grid = []
        frame = getattr(raw_obs, "frame", None)
        if frame:
            grid = frame[0].tolist() if hasattr(frame[0], "tolist") else frame[0]
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        meta = {
            "game_id": getattr(raw_obs, "game_id", None),
            "state": getattr(raw_obs, "state", None),
            "levels_completed": getattr(raw_obs, "levels_completed", None),
            "win_levels": getattr(raw_obs, "win_levels", None),
            "guid": getattr(raw_obs, "guid", None),
        }
        return ObsV1(grid=grid, h=h, w=w, meta=meta).to_dict()

    def _coord_action_id(self, available: list[int]) -> Optional[int]:
        for action in self._game_action_enum:
            if action.is_complex() and int(action.value) in available:
                return int(action.value)
        return None


def default_env_adapter(**kwargs: Any) -> ArcAgiDefaultAdapter:
    return ArcAgiDefaultAdapter(**kwargs)
