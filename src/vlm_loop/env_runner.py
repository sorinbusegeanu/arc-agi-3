from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from v3_1.execution.env_factory import build_env

from .action_schema import canonicalize_action_name, env_action_name, extract_available_actions, validate_action_sequence
from .frame_writer import write_frame_png
from .models import ActionSequence, EpisodeResult


class EnvSession:
    def __init__(
        self,
        *,
        env_factory_path: str | None,
        env_id: str | None,
        env_root: str | None,
        seed: int,
        render_terminal: bool,
    ) -> None:
        self.seed = seed
        self.env_factory_path = env_factory_path
        self.env_id = env_id
        self.env_root = env_root
        self.render_terminal = render_terminal
        self.adapter = build_env(
            env_factory_path,
            env_id=env_id,
            env_root=env_root,
            seed=seed,
            render_terminal=render_terminal,
        )
        self.observation = None
        self.info: dict[str, Any] = {}
        self.done = False
        self.truncated = False
        self.action_rows: list[dict[str, Any]] = []
        self.allowed_actions: list[str] = []
        self.reset(seed=seed)

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        actual_seed = self.seed if seed is None else seed
        self.observation, self.info = self.adapter.reset(seed=actual_seed)
        self.done = False
        self.truncated = False
        self.action_rows = self.adapter.available_actions() if hasattr(self.adapter, "available_actions") else []
        self.allowed_actions = extract_available_actions(self.adapter) or extract_available_actions_from_info(self.info)
        return self.observation, self.info

    def capture_current_frame(
        self,
        *,
        frame_dir: str,
        frame_index: int,
        episode_frame_dir: str | None = None,
        episode_frame_index: int | None = None,
        append_to_episode: bool = False,
    ) -> tuple[str, int | None]:
        frame_path = write_frame_png(self.observation, frame_dir, frame_index)
        if append_to_episode and episode_frame_dir is not None and episode_frame_index is not None:
            write_frame_png(self.observation, episode_frame_dir, episode_frame_index)
            return frame_path, episode_frame_index + 1
        return frame_path, episode_frame_index

    def run_sequence(
        self,
        *,
        sequence: ActionSequence,
        episode_dir: str,
        episode_frame_dir: str,
        episode_frame_index: int,
        max_steps: int,
    ) -> tuple[EpisodeResult, int]:
        output_dir = Path(episode_dir)
        frame_dir = output_dir / "frames"
        video_path = output_dir / "episode.mp4"
        actions = validate_action_sequence(sequence.actions, allowed_actions=self.allowed_actions, max_length=max_steps)
        action_log: list[dict[str, Any]] = []
        frame_count = 0
        total_reward = 0.0
        self.capture_current_frame(frame_dir=str(frame_dir), frame_index=frame_count)
        frame_count += 1
        current_episode_frame_index = episode_frame_index
        for step_index, action_name in enumerate(actions):
            if self.done or self.truncated:
                break
            env_action = resolve_env_action(action_name, self.action_rows)
            self.observation, reward, self.done, self.truncated, step_info = self.adapter.step(env_action)
            self.info = step_info
            self.action_rows = self.adapter.available_actions() if hasattr(self.adapter, "available_actions") else self.action_rows
            updated_actions = extract_available_actions(self.adapter) or extract_available_actions_from_info(step_info)
            if updated_actions:
                self.allowed_actions = updated_actions
            total_reward += float(reward)
            action_log.append(
                {
                    "step_index": step_index,
                    "action": action_name,
                    "env_action": _safe_json(env_action),
                    "reward": float(reward),
                    "done": bool(self.done),
                    "truncated": bool(self.truncated),
                    "info": _safe_json(step_info),
                }
            )
            write_frame_png(self.observation, str(frame_dir), frame_count)
            write_frame_png(self.observation, episode_frame_dir, current_episode_frame_index)
            frame_count += 1
            current_episode_frame_index += 1
            if self.done or self.truncated:
                break
        result = EpisodeResult(
            episode_id=output_dir.name.replace("seq_", "ep_"),
            sequence_id=sequence.sequence_id,
            actions=actions[: len(action_log)],
            frame_dir=str(frame_dir),
            video_path=str(video_path),
            step_count=len(action_log),
            done=self.done,
            truncated=self.truncated,
            output_dir=str(output_dir),
            total_reward=total_reward,
            action_log=action_log,
            frame_count=frame_count,
        )
        episode_payload = result.to_dict()
        episode_payload["available_actions"] = list(self.allowed_actions)
        with open(output_dir / "episode.json", "w", encoding="utf-8") as handle:
            json.dump(episode_payload, handle, indent=2)
        return result, current_episode_frame_index

    def close(self) -> None:
        if hasattr(self.adapter.env, "close"):
            try:
                self.adapter.env.close()
            except Exception:
                pass


def run_episode(
    *,
    env_factory_path: str | None,
    env_id: str | None,
    env_root: str | None,
    seed: int,
    render_terminal: bool,
    sequence: ActionSequence,
    episode_dir: str,
    max_steps: int,
) -> EpisodeResult:
    session = EnvSession(
        env_factory_path=env_factory_path,
        env_id=env_id,
        env_root=env_root,
        seed=seed,
        render_terminal=render_terminal,
    )
    try:
        episode_frame_dir = str(Path(episode_dir) / "frames")
        session.capture_current_frame(
            frame_dir=episode_frame_dir,
            frame_index=0,
            episode_frame_dir=episode_frame_dir,
            episode_frame_index=0,
            append_to_episode=False,
        )
        result, _ = session.run_sequence(
            sequence=sequence,
            episode_dir=episode_dir,
            episode_frame_dir=episode_frame_dir,
            episode_frame_index=1,
            max_steps=max_steps,
        )
        return result
    finally:
        session.close()


def extract_available_actions_from_info(info: Any) -> list[str]:
    if not isinstance(info, dict):
        return []
    normalized: list[str] = []
    for item in info.get("available_actions", []):
        if isinstance(item, dict):
            candidate = item.get("name", item.get("action_name", item.get("id", item.get("action_id"))))
        else:
            candidate = item
        normalized.append(str(candidate).upper())
    return normalized


def resolve_env_action(action_name: str, action_rows: list[dict[str, Any]]) -> Any:
    target = canonicalize_action_name(action_name)
    for row in action_rows:
        if not isinstance(row, dict):
            continue
        if canonicalize_action_name(row) != target:
            continue
        if row.get("raw") is not None:
            return row.get("raw")
        if row.get("id") is not None:
            return row.get("id")
        if row.get("name") is not None:
            return row.get("name")
    return env_action_name(action_name)


def _safe_json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    return str(value)
