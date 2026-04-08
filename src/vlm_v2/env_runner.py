from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from v3_1.execution.env_factory import build_env

from .action_schema import extract_available_actions, resolve_env_action
from .frame_writer import write_frame_png
from .models import BranchPlan, BranchRunResult, VLMV2Config
from .video_builder import build_episode_video


class EnvSession:
    def __init__(self, cfg: VLMV2Config, *, seed: int) -> None:
        self.cfg = cfg
        self.seed = int(seed)
        self.adapter = build_env(
            cfg.env_factory_path,
            env_id=cfg.env_id,
            env_root=cfg.env_root,
            seed=self.seed,
            render_terminal=cfg.render_terminal,
        )
        self.observation = None
        self.info: dict[str, Any] = {}
        self.done = False
        self.truncated = False
        self.action_rows: list[dict[str, Any]] = []
        self.allowed_actions: list[str] = []
        self.reset()

    def reset(self) -> tuple[Any, dict[str, Any]]:
        self.observation, self.info = self.adapter.reset(seed=self.seed)
        self.done = False
        self.truncated = False
        self.action_rows = self.adapter.available_actions() if hasattr(self.adapter, "available_actions") else []
        self.allowed_actions = extract_available_actions(self.adapter, self.info)
        return self.observation, self.info

    def close(self) -> None:
        if hasattr(self.adapter.env, "close"):
            try:
                self.adapter.env.close()
            except Exception:
                pass


def run_branch(
    *,
    cfg: VLMV2Config,
    seed: int,
    branch: BranchPlan,
    full_actions: list[str],
    output_dir: str,
    max_actions_allowed: int,
) -> BranchRunResult:
    session = EnvSession(cfg, seed=seed)
    out_dir = Path(output_dir)
    frame_dir = out_dir / "frames"
    action_log: list[dict[str, Any]] = []
    total_reward = 0.0
    frame_index = 0
    levels_completed_before = int(session.info.get("levels_completed", 0) or 0)
    write_frame_png(session.observation, str(frame_dir), frame_index)
    frame_index += 1
    executed_actions: list[str] = []
    try:
        for action in full_actions[: max(0, int(max_actions_allowed))]:
            if session.done or session.truncated:
                break
            env_action = resolve_env_action(action, session.action_rows)
            observation, reward, done, truncated, step_info = session.adapter.step(env_action)
            session.observation = observation
            session.info = dict(step_info or {})
            session.done = bool(done)
            session.truncated = bool(truncated)
            session.action_rows = session.adapter.available_actions() if hasattr(session.adapter, "available_actions") else session.action_rows
            updated_actions = extract_available_actions(session.adapter, session.info)
            if updated_actions:
                session.allowed_actions = updated_actions
            executed_actions.append(action)
            total_reward += float(reward or 0.0)
            action_log.append(
                {
                    "step_index": len(executed_actions) - 1,
                    "action": action,
                    "env_action": _safe_json(env_action),
                    "reward": float(reward or 0.0),
                    "done": bool(done),
                    "truncated": bool(truncated),
                    "info": _safe_json(step_info),
                }
            )
            write_frame_png(session.observation, str(frame_dir), frame_index)
            frame_index += 1
            if session.done or session.truncated:
                break
        video_path = build_episode_video(str(frame_dir), fps=cfg.fps, output_name="episode.mp4")
        result = BranchRunResult(
            branch=branch,
            output_dir=str(out_dir),
            frame_dir=str(frame_dir),
            video_path=video_path,
            action_log=action_log,
            executed_actions=list(executed_actions),
            executed_count=len(executed_actions),
            total_reward=float(total_reward),
            done=bool(session.done),
            truncated=bool(session.truncated),
            won=bool(session.info.get("win", False)),
            levels_completed_before=levels_completed_before,
            levels_completed_after=int(session.info.get("levels_completed", levels_completed_before) or 0),
            start_info={"levels_completed": levels_completed_before},
            end_info=dict(session.info),
        )
        with open(out_dir / "branch_result.json", "w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2)
        return result
    finally:
        session.close()


def _safe_json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    return str(value)
