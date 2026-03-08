from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from codex_baseline_v2.adapters.action_adapter import adapt_action
from codex_baseline_v2.adapters.observation_adapter import adapt_observation
from codex_baseline_v2.shared.schemas import SCHEMA_VERSION, TrajectoryEpisodeV2, TrajectoryStepV2
from codex_baseline_v2.shared.state_identity import canonical_state_identity


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_legacy_payload(path: str) -> Dict[str, Any]:
    if path.endswith(".jsonl"):
        rows = _load_jsonl(path)
        return {"schema_version": "TRAJECTORY_BATCH_V1", "episodes": rows}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def convert_episode(episode: Dict[str, Any], episode_idx: int, game_id_override: Optional[str] = None) -> TrajectoryEpisodeV2:
    game_id = str(game_id_override or episode.get("game_id") or "unknown_game")
    episode_id = str(episode.get("episode_id") or f"ep_{episode_idx:05d}")
    steps_raw = episode.get("steps", []) if isinstance(episode.get("steps"), list) else []
    steps: List[TrajectoryStepV2] = []
    for step in steps_raw:
        step_idx = int(step.get("step_idx", len(steps)))
        obs = adapt_observation(step)
        action = adapt_action(step)
        pre_state = canonical_state_identity(obs, include_payload=False)
        steps.append(
            TrajectoryStepV2(
                schema_version=SCHEMA_VERSION,
                game_id=game_id,
                episode_id=episode_id,
                step_idx=step_idx,
                action=action,
                pre_state_hash=pre_state.get("state_hash"),
                post_state_hash=None,
                state_hash_valid=False,
                instruction_id=None,
                target_poi_id=None,
                target_type=None,
                target_geometry=None,
                target_source_round=None,
                reward=float(step.get("reward", step.get("reward_total", 0.0)) or 0.0),
                done=bool(step.get("done", False)),
                observation=obs,
                observation_summary=None,
                info={
                    "raw_step": step,
                    "state_hash_before": step.get("state_hash_before"),
                    "state_hash_after": step.get("state_hash_after"),
                    "state_signature_version": pre_state.get("state_signature_version"),
                },
            )
        )
    return TrajectoryEpisodeV2(
        schema_version=SCHEMA_VERSION,
        game_id=game_id,
        episode_id=episode_id,
        steps=steps,
        done=bool(episode.get("done", False)),
        win=bool(episode.get("win", False)),
        seed=episode.get("seed"),
        metadata={k: v for k, v in episode.items() if k not in {"steps", "done", "win", "seed"}},
    )


def import_legacy_trajectories(payload: Dict[str, Any], game_id_override: Optional[str] = None) -> List[TrajectoryEpisodeV2]:
    episodes = payload.get("episodes", []) if isinstance(payload.get("episodes"), list) else []
    return [convert_episode(ep, idx, game_id_override=game_id_override) for idx, ep in enumerate(episodes)]


def import_legacy_from_path(path: str, game_id_override: Optional[str] = None) -> List[TrajectoryEpisodeV2]:
    payload = load_legacy_payload(path)
    return import_legacy_trajectories(payload, game_id_override=game_id_override)
