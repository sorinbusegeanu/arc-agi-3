from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any
from uuid import uuid4
from pathlib import Path

import torch

from rl_v1.configs.schema import EnvConfig, ModelConfig, RewardConfig
from rl_v1.data.contracts import ObservationPackage, V1Action, action_mask_from_available
from rl_v1.data.preprocessing import frame_to_model_tensor
from rl_v1.utils.runtime import build_quiet_arc_logger, ensure_arc_paths, resolve_environments_dir


class ArcEnvironmentAdapter:
    def __init__(self, env_cfg: EnvConfig, model_cfg: ModelConfig, game_id: str, reward_cfg: RewardConfig | None = None) -> None:
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg
        self.reward_cfg = reward_cfg or RewardConfig()
        self.game_id = game_id
        self._game_id_to_index = {gid: idx for idx, gid in enumerate(env_cfg.game_ids)}
        if self.game_id not in self._game_id_to_index:
            raise ValueError(f"configured game_id not found in env_cfg.game_ids: {self.game_id}")
        self._game_id_index = int(self._game_id_to_index[self.game_id])
        self._seed = int(env_cfg.seed)
        self.env_instance_id = f"env-{uuid4().hex[:10]}"
        self._episode_counter = 0
        self._current_episode_id: str | None = None
        self._arcade = None
        self._env = None
        self._last_obs: ObservationPackage | None = None
        self._previous_frames = []
        self._previous_level_index: int | None = None
        self._deepest_level_index: int | None = None
        self._current_level_index: int | None = None
        self._previous_raw_level_completed = False
        self._previous_raw_game_won = False
        self._previous_levels_completed: int | None = None
        self._steps_in_current_level = 0
        self._previous_state_signature: str | None = None
        self._recent_state_signatures = []
        self._last_action_id = None
        self._same_action_streak = 0
        self._pending_action_id = None
        self._build_arcade()
        self._create_env(self._seed)

    def _build_arcade(self) -> None:
        ensure_arc_paths()
        from arc_agi import Arcade, OperationMode

        environments_dir = self.env_cfg.environments_dir or resolve_environments_dir()
        self._arcade = Arcade(
            environments_dir=environments_dir,
            operation_mode=OperationMode(self.env_cfg.operation_mode),
            logger=build_quiet_arc_logger(),
        )

    def _create_env(self, seed: int) -> None:
        self._env = self._arcade.make(self.game_id, seed=int(seed), save_recording=bool(self.env_cfg.save_recording), render_mode=self.env_cfg.render_mode)
        if self._env is None:
            raise RuntimeError(f"Arcade.make failed for game_id={self.game_id}")

    def reset(self, seed: int | None = None) -> ObservationPackage:
        if seed is not None and int(seed) != self._seed:
            self._seed = int(seed)
            self.close()
            self._create_env(self._seed)
        self._episode_counter += 1
        self._current_episode_id = f"{self.env_instance_id}:{self.game_id}:ep{self._episode_counter}"
        self._previous_frames = []
        self._previous_level_index = None
        self._current_level_index = None
        self._deepest_level_index = None
        self._previous_raw_level_completed = False
        self._previous_raw_game_won = False
        self._previous_levels_completed = None
        self._steps_in_current_level = 0
        self._previous_state_signature = None
        self._recent_state_signatures = []
        self._last_action_id = None
        self._same_action_streak = 0
        self._pending_action_id = None
        obs = self._normalize(self._env.reset(), is_reset=True)
        self._last_obs = obs
        return obs

    def step(self, action_dict: dict[str, Any] | V1Action) -> ObservationPackage:
        from arcengine import GameAction

        action = action_dict if isinstance(action_dict, V1Action) else V1Action(**action_dict)
        payload = None
        if action.action_id == 6:
            if action.x is None or action.y is None:
                raise ValueError("ACTION6 requires explicit x and y payload")
            payload = {"x": int(action.x), "y": int(action.y)}
        self._pending_action_id = int(action.action_id)
        try:
            raw = self._env.step(GameAction.from_id(int(action.action_id)), data=payload, reasoning=None)
            obs = self._normalize(raw, is_reset=False)
        finally:
            self._pending_action_id = None
        if self._last_obs is not None:
            changed = self._last_obs.current_frame.ne(obs.current_frame)
            obs = replace(obs, changed_cell_mask=changed.to(dtype=torch.float32))
            if isinstance(obs.raw_metadata, dict):
                obs.raw_metadata["changed_cell_count"] = int(changed.sum().item())
        elif obs.changed_cell_mask is None:
            obs = replace(obs, changed_cell_mask=torch.zeros_like(obs.current_frame))
        self._last_obs = obs
        return obs

    def close(self) -> None:
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()
        self._env = None

    def _normalize(self, raw: Any, *, is_reset: bool) -> ObservationPackage:
        _ensure_reward_tracking_state(self)
        current_frame, valid_mask = frame_to_model_tensor(raw, canvas_height=self.model_cfg.canvas_height, canvas_width=self.model_cfg.canvas_width)
        prev_1 = self._previous_frames[-1] if len(self._previous_frames) >= 1 else _zeros_like(current_frame)
        prev_2 = self._previous_frames[-2] if len(self._previous_frames) >= 2 else _zeros_like(current_frame)
        self._previous_frames.append(current_frame.clone())
        self._previous_frames = self._previous_frames[-2:]
        available = tuple(int(a) for a in getattr(raw, "available_actions", []) or [])
        state = getattr(getattr(raw, "state", None), "name", str(getattr(raw, "state", "UNKNOWN")))
        current_level_index = _extract_current_level_index(raw, self._env)
        state_signature = _build_state_signature(
            current_frame=current_frame,
            current_level_index=current_level_index,
            available_action_ids=available,
        )

        raw_level_completed = getattr(raw, "level_completed", None)
        raw_levels_completed = getattr(raw, "levels_completed", None)
        if raw_level_completed is not None:
            level_completed = bool(raw_level_completed) and not self._previous_raw_level_completed
        else:
            level_completed = self._previous_level_index is not None and current_level_index > self._previous_level_index
            if not level_completed and raw_levels_completed is not None and self._previous_levels_completed is not None:
                level_completed = int(raw_levels_completed) > int(self._previous_levels_completed)
        self._previous_raw_level_completed = bool(raw_level_completed) if raw_level_completed is not None else False
        self._previous_levels_completed = int(raw_levels_completed) if raw_levels_completed is not None else self._previous_levels_completed

        raw_game_won = getattr(raw, "game_won", None)
        if raw_game_won is None:
            raw_game_won = getattr(raw, "won", None)
        if raw_game_won is None:
            raw_game_won = state == "WIN"
        game_won = bool(raw_game_won) and not self._previous_raw_game_won
        self._previous_raw_game_won = bool(raw_game_won)

        if level_completed and not game_won and self._previous_level_index is not None:
            if current_level_index <= self._previous_level_index:
                raise ValueError(
                    f"invalid level progression for {self._current_episode_id}: "
                    f"prev={self._previous_level_index}, current={current_level_index}, "
                    f"raw={_safe_raw_metadata(raw)}"
                )

        if self._deepest_level_index is None:
            self._deepest_level_index = current_level_index
        else:
            self._deepest_level_index = max(self._deepest_level_index, current_level_index)
        if self._deepest_level_index < current_level_index:
            raise ValueError(
                f"invalid deepest level tracking for {self._current_episode_id}: "
                f"deepest={self._deepest_level_index}, current={current_level_index}, "
                f"raw={_safe_raw_metadata(raw)}"
            )
        self._current_level_index = current_level_index

        if is_reset:
            self._steps_in_current_level = 0

        base_reward = (
            0.0
            if self._steps_in_current_level < int(self.reward_cfg.zero_steps_penalty)
            else float(self.reward_cfg.step_penalty)
        )
        reward = float(base_reward)
        if level_completed:
            reward += float(self.reward_cfg.level_complete_bonus)
        if game_won:
            reward += float(self.reward_cfg.game_win_bonus)
        if self._previous_level_index is not None and current_level_index != self._previous_level_index:
            self._recent_state_signatures = []
        revisit_window_size = int(self.reward_cfg.revisit_window_size)
        revisit_match_count = 0
        repeat_penalty_applied = 0.0
        if bool(self.reward_cfg.repeat_state_penalty_enabled) and revisit_window_size > 0:
            recent_window = self._recent_state_signatures[-revisit_window_size:]
            matching_indices = [idx for idx, signature in enumerate(recent_window) if signature == state_signature]
            revisit_match_count = len(matching_indices)
            if revisit_match_count > 0:
                penalty = float(self.reward_cfg.repeat_state_penalty)
                mode = str(self.reward_cfg.revisit_penalty_mode)
                if mode == "binary":
                    repeat_penalty_applied = penalty
                elif mode == "count":
                    repeat_penalty_applied = penalty * revisit_match_count
                elif mode == "decay":
                    decay = float(self.reward_cfg.revisit_penalty_decay)
                    recent_positions = [len(recent_window) - 1 - idx for idx in matching_indices]
                    recent_positions.sort()
                    repeat_penalty_applied = penalty * sum(decay**k for k in recent_positions)
                else:
                    raise ValueError(f"unsupported revisit_penalty_mode: {self.reward_cfg.revisit_penalty_mode}")
                reward += float(repeat_penalty_applied)

        same_action_penalty_applied = 0.0
        if not is_reset and self._pending_action_id is not None:
            current_action_id = int(self._pending_action_id)
            if self._last_action_id is not None and current_action_id == int(self._last_action_id):
                self._same_action_streak += 1
            else:
                self._same_action_streak = 1
                self._last_action_id = current_action_id
            if self._same_action_streak >= int(self.reward_cfg.same_action_streak_threshold):
                same_action_penalty_applied = float(self.reward_cfg.same_action_streak_penalty)
                reward += same_action_penalty_applied
        elif is_reset:
            self._same_action_streak = 0
            self._last_action_id = None

        self._recent_state_signatures.append(state_signature)
        if revisit_window_size > 0:
            self._recent_state_signatures = self._recent_state_signatures[-revisit_window_size:]
        self._previous_state_signature = state_signature
        if not is_reset and self.env_cfg.debug_log_path:
            _append_reward_debug_log(
                self.env_cfg.debug_log_path,
                episode_id=self._current_episode_id or "",
                env_instance_id=self.env_instance_id,
                game_id=self.game_id,
                current_level_index=current_level_index,
                zero_steps_penalty=int(self.reward_cfg.zero_steps_penalty),
                steps_in_current_level=int(self._steps_in_current_level),
                base_reward=float(base_reward),
                step_penalty=float(self.reward_cfg.step_penalty),
                level_completed=bool(level_completed),
                level_complete_bonus=float(self.reward_cfg.level_complete_bonus),
                game_won=bool(game_won),
                game_win_bonus=float(self.reward_cfg.game_win_bonus),
                repeat_state_penalty_enabled=bool(self.reward_cfg.repeat_state_penalty_enabled),
                revisit_match_count=int(revisit_match_count),
                repeat_state_penalty=float(self.reward_cfg.repeat_state_penalty),
                repeat_penalty_applied=float(repeat_penalty_applied),
                same_action_streak=int(self._same_action_streak),
                same_action_streak_threshold=int(self.reward_cfg.same_action_streak_threshold),
                same_action_streak_penalty=float(self.reward_cfg.same_action_streak_penalty),
                same_action_penalty_applied=float(same_action_penalty_applied),
                total_reward=float(reward),
            )

        terminal = state not in {"NOT_FINISHED", "None", "UNKNOWN"}
        if game_won and not terminal:
            raise ValueError(
                f"invalid game_won transition for {self._current_episode_id}: "
                f"terminal={terminal}, state={state}, raw={_safe_raw_metadata(raw)}"
            )

        if not is_reset:
            if self._previous_level_index is not None and current_level_index != self._previous_level_index:
                self._steps_in_current_level = 0
            else:
                self._steps_in_current_level += 1
        metadata = {
            "game_id": self.game_id,
            "episode_id": self._current_episode_id,
            "env_instance_id": self.env_instance_id,
            "guid": getattr(raw, "guid", None),
            "seed": self._seed,
            "state": state,
            "levels_completed": getattr(raw, "levels_completed", 0),
            "win_levels": getattr(raw, "win_levels", 0),
            "game_id_index": int(self._game_id_index),
            "current_level_index": current_level_index,
            "level_completed": level_completed,
            "game_won": game_won,
            "deepest_level_index": int(self._deepest_level_index),
            "action_input": _serialize_action_input(getattr(raw, "action_input", None)),
            "available_actions": list(available),
            "frame_layers": len(getattr(raw, "frame", []) or []),
            "raw_type": type(raw).__name__,
            "revisit_match_count": int(revisit_match_count),
            "same_action_streak": int(self._same_action_streak),
            "revisit_window_size": int(revisit_window_size),
            "repeat_penalty_applied": float(repeat_penalty_applied),
            "same_action_penalty_applied": float(same_action_penalty_applied),
        }
        obs = ObservationPackage(
            current_frame=current_frame,
            previous_frame_1=prev_1,
            previous_frame_2=prev_2,
            valid_action_mask=action_mask_from_available(available).to(dtype=torch.bool),
            action6_clickable=(6 in available) if available else None,
            raw_metadata=metadata,
            terminal=terminal,
            reward=reward,
            valid_pixel_mask=valid_mask,
            available_action_ids=available,
            game_id=self.game_id,
            game_id_index=int(self._game_id_index),
            current_level_index=current_level_index,
            level_completed=bool(level_completed),
            game_won=bool(game_won),
            deepest_level_index=int(self._deepest_level_index),
            step_count=int(self._steps_in_current_level),
            changed_cell_mask=torch.zeros_like(current_frame),
            raw_response=raw,
        )
        self._previous_level_index = current_level_index
        return obs


def _serialize_action_input(action_input: Any) -> dict[str, Any] | None:
    if action_input is None:
        return None
    action_id = getattr(action_input, "id", None)
    return {"id": getattr(action_id, "value", action_id), "name": getattr(action_id, "name", str(action_id)), "data": getattr(action_input, "data", None), "reasoning": getattr(action_input, "reasoning", None)}


def _zeros_like(tensor):
    import torch

    return torch.zeros_like(tensor)


def _safe_raw_metadata(raw: Any) -> dict[str, Any]:
    keys = (
        "guid",
        "game_id",
        "state",
        "level_completed",
        "levels_completed",
        "current_level_index",
        "level_index",
        "won",
        "game_won",
    )
    out: dict[str, Any] = {}
    for key in keys:
        if hasattr(raw, key):
            out[key] = getattr(raw, key)
    return out


def _build_state_signature(*, current_frame: torch.Tensor, current_level_index: int, available_action_ids: tuple[int, ...]) -> str:
    hasher = hashlib.sha256()
    hasher.update(current_frame.detach().cpu().contiguous().numpy().tobytes())
    hasher.update(int(current_level_index).to_bytes(4, byteorder="little", signed=True))
    hasher.update(bytes(int(a) & 0xFF for a in available_action_ids))
    return hasher.hexdigest()


def _extract_current_level_index(raw: Any, env_wrapper: Any) -> int:
    if hasattr(raw, "current_level_index") and getattr(raw, "current_level_index") is not None:
        return int(getattr(raw, "current_level_index"))
    if hasattr(raw, "level_index") and getattr(raw, "level_index") is not None:
        return int(getattr(raw, "level_index"))
    game = getattr(env_wrapper, "_game", None)
    if game is not None and hasattr(game, "level_index") and getattr(game, "level_index") is not None:
        return int(getattr(game, "level_index"))
    raise ValueError("raw environment response does not expose a usable current level index")


def _ensure_reward_tracking_state(adapter: ArcEnvironmentAdapter) -> None:
    if not hasattr(adapter, "_recent_state_signatures"):
        adapter._recent_state_signatures = []
    if not hasattr(adapter, "_last_action_id"):
        adapter._last_action_id = None
    if not hasattr(adapter, "_same_action_streak"):
        adapter._same_action_streak = 0
    if not hasattr(adapter, "_pending_action_id"):
        adapter._pending_action_id = None
    if not hasattr(adapter, "_game_id_index"):
        adapter._game_id_index = 0
    if not hasattr(adapter, "env_cfg"):
        adapter.env_cfg = type("EnvCfgStub", (), {"debug_log_path": None})()


def _append_reward_debug_log(
    log_path: str,
    *,
    episode_id: str,
    env_instance_id: str,
    game_id: str,
    current_level_index: int,
    zero_steps_penalty: int,
    steps_in_current_level: int,
    base_reward: float,
    step_penalty: float,
    level_completed: bool,
    level_complete_bonus: float,
    game_won: bool,
    game_win_bonus: float,
    repeat_state_penalty_enabled: bool,
    revisit_match_count: int,
    repeat_state_penalty: float,
    repeat_penalty_applied: float,
    same_action_streak: int,
    same_action_streak_threshold: int,
    same_action_streak_penalty: float,
    same_action_penalty_applied: float,
    total_reward: float,
) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (
        f"episode_id({episode_id}), env_id({env_instance_id}), game_id({game_id}), "
        f"current_level_index({current_level_index}), steps_in_current_level({steps_in_current_level}), "
        f"zero_steps_penalty({zero_steps_penalty}), step_penalty({step_penalty}), base_reward({base_reward}), "
        f"level_completed({int(level_completed)}), level_complete_bonus({level_complete_bonus}), "
        f"game_won({int(game_won)}), game_win_bonus({game_win_bonus}), "
        f"repeat_state_penalty_enabled({int(repeat_state_penalty_enabled)}), revisit_match_count({revisit_match_count}), "
        f"repeat_state_penalty({repeat_state_penalty}), repeat_penalty_applied({repeat_penalty_applied}), "
        f"same_action_streak({same_action_streak}), same_action_streak_threshold({same_action_streak_threshold}), "
        f"same_action_streak_penalty({same_action_streak_penalty}), same_action_penalty_applied({same_action_penalty_applied}), "
        f"total_reward({total_reward})\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
