from __future__ import annotations

import json
import os
import importlib
import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.pool import Pool
from typing import Any, Dict, List, Optional, Tuple

from codex_baseline_v2.runtime.environment_session import EnvironmentSessionV2, StepResultV2
from codex_baseline_v2.runtime.trajectory_policy import PolicyStateV2, TrajectoryPolicyV2
from codex_baseline_v2.shared.state_identity import canonical_state_identity
from codex_baseline_v2.shared.schemas import ActionDescriptorV2, SCHEMA_VERSION, TrajectoryEpisodeV2, TrajectoryStepV2
from codex_baseline_v2.shared.storage import StoragePathsV2
from codex_baseline_v2.storage.sqlite_intermediates import SQLiteIntermediateStoreV2, sqlite_db_path_for_round


@dataclass
class CollectionConfigV2:
    episodes: int
    max_steps_per_episode: int
    max_steps_per_instruction: int
    seed: Optional[int] = None
    action_repeat_limit: int = 4
    keep_invalid_steps_for_debug: bool = False
    write_raw_copy: bool = False
    keep_observations_in_artifacts: bool = True
    keep_raw_info_in_artifacts: bool = True
    keep_observation_summaries_in_artifacts: bool = True
    storage_backend: str = "files"


class TrajectoryCollectorV2:
    def __init__(self, storage: StoragePathsV2, cfg: CollectionConfigV2) -> None:
        self.storage = storage
        self.cfg = cfg

    def collect_episode(
        self,
        session: EnvironmentSessionV2,
        policy: TrajectoryPolicyV2,
        mode: str,
        instruction: Optional[Any] = None,
        round_id: int = 0,
        episode_idx: int = 0,
    ) -> TrajectoryEpisodeV2:
        obs = session.reset(seed=(self.cfg.seed + episode_idx) if self.cfg.seed is not None else None)
        steps: List[TrajectoryStepV2] = []
        policy_state = PolicyStateV2()
        observed_states_total = 0
        unique_pre_states: set[str] = set()
        unique_post_states: set[str] = set()
        invalid_state_count = 0
        for step_idx in range(self.cfg.max_steps_per_episode):
            avail = session.available_actions()
            if mode in {"random_probe", "unguided_probe"}:
                action = policy.random_action(avail)
            else:
                target_coord = None
                if instruction is not None and getattr(instruction, "target_region", None) is not None:
                    target_coord = instruction.target_region.centroid()
                action = policy.instructed_action(instruction, target_coord, avail, policy_state)
            pre_state = canonical_state_identity(obs, include_payload=False)
            result = session.step(action)
            post_state = canonical_state_identity(result.observation, include_payload=False)
            pre_valid = bool(pre_state.get("valid"))
            post_valid = bool(post_state.get("valid"))
            if pre_valid:
                observed_states_total += 1
                unique_pre_states.add(str(pre_state.get("state_hash")))
            else:
                invalid_state_count += 1
            if post_valid:
                observed_states_total += 1
                unique_post_states.add(str(post_state.get("state_hash")))
            else:
                invalid_state_count += 1
            if (not pre_valid or not post_valid) and not self.cfg.keep_invalid_steps_for_debug:
                break
            step = TrajectoryStepV2(
                schema_version=SCHEMA_VERSION,
                game_id=session.game_id,
                episode_id=f"round{round_id:03d}_ep{episode_idx:05d}",
                step_idx=step_idx,
                action=action,
                pre_state_hash=pre_state.get("state_hash"),
                post_state_hash=post_state.get("state_hash"),
                state_hash_valid=bool(pre_valid and post_valid),
                instruction_id=getattr(instruction, "instruction_id", None),
                target_poi_id=getattr(instruction, "target_poi_id", None),
                target_type=getattr(instruction, "target_type", None),
                target_geometry=getattr(instruction, "target_geometry", None),
                target_source_round=getattr(instruction, "target_source_round", None),
                reward=result.reward,
                done=result.done,
                observation=obs,
                observation_summary=None,
                info={
                    "available_actions": result.available_actions,
                    "step_info": result.info,
                    "collection_mode": mode,
                    "instruction_id": getattr(instruction, "instruction_id", None),
                    "target_poi_id": getattr(instruction, "target_poi_id", None),
                    "target_type": getattr(instruction, "target_type", None),
                    "target_geometry": getattr(instruction, "target_geometry", None).to_dict() if getattr(instruction, "target_geometry", None) is not None else None,
                    "target_source_round": getattr(instruction, "target_source_round", None),
                    "state_signature_version": pre_state.get("state_signature_version"),
                },
            )
            steps.append(step)
            obs = result.observation
            if result.done:
                break
        status = session.progress_status()
        return TrajectoryEpisodeV2(
            schema_version=SCHEMA_VERSION,
            game_id=session.game_id,
            episode_id=f"round{round_id:03d}_ep{episode_idx:05d}",
            steps=steps,
            done=bool(steps and steps[-1].done),
            win=bool(status.get("win", False)),
            seed=(self.cfg.seed + episode_idx) if self.cfg.seed is not None else None,
            metadata={
                "mode": mode,
                "collection_mode": mode,
                "instruction_id": getattr(instruction, "instruction_id", None),
                "target_poi_id": getattr(instruction, "target_poi_id", None),
                "win": bool(status.get("win", False)),
                "state_counters": {
                    "observed_states_total": observed_states_total,
                    "unique_pre_states": len(unique_pre_states),
                    "unique_post_states": len(unique_post_states),
                    "invalid_state_count": invalid_state_count,
                },
            },
        )

    def collect_round(
        self,
        session: EnvironmentSessionV2,
        mode: str,
        instruction: Optional[Any],
        round_id: int,
    ) -> List[TrajectoryEpisodeV2]:
        policy = TrajectoryPolicyV2(seed=self.cfg.seed)
        episodes = []
        for ep_idx in range(self.cfg.episodes):
            episodes.append(self.collect_episode(session, policy, mode, instruction, round_id, ep_idx))
        return episodes

    def collect_round_parallel(
        self,
        env_factory_path: str,
        env_id: str,
        env_root: str,
        mode: str,
        instruction: Optional[Any],
        round_id: int,
        workers: int,
        pool: Optional[Pool] = None,
    ) -> List[TrajectoryEpisodeV2]:
        payloads = [
            (
                env_factory_path,
                env_id,
                env_root,
                self.cfg,
                mode,
                instruction,
                round_id,
                ep_idx,
            )
            for ep_idx in range(self.cfg.episodes)
        ]
        if pool is not None:
            return pool.map(_collect_episode_worker, payloads)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as local_pool:
            return local_pool.map(_collect_episode_worker, payloads)

    def collect_round_parallel_iter(
        self,
        env_factory_path: str,
        env_id: str,
        env_root: str,
        mode: str,
        instruction: Optional[Any],
        round_id: int,
        workers: int,
        pool: Optional[Pool] = None,
    ):
        payloads = [
            (
                env_factory_path,
                env_id,
                env_root,
                self.cfg,
                mode,
                instruction,
                round_id,
                ep_idx,
            )
            for ep_idx in range(self.cfg.episodes)
        ]
        if pool is not None:
            for episode in pool.imap_unordered(_collect_episode_worker, payloads):
                yield episode
            return
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as local_pool:
            for episode in local_pool.imap_unordered(_collect_episode_worker, payloads):
                yield episode

    def write_artifacts(self, game_id: str, round_id: int, episodes: List[TrajectoryEpisodeV2]) -> None:
        paths = self.storage.ensure_round_dirs(game_id, round_id)
        raw_path = os.path.join(paths["raw_trajectories"], "episodes.jsonl")
        norm_path = os.path.join(paths["normalized_trajectories"], "episodes.jsonl")
        counters = _compute_round_state_counters(episodes)
        counters_path = os.path.join(paths["round_root"], "state_counters.json")
        lines = [json.dumps(_episode_to_artifact_dict(ep, self.cfg), sort_keys=True) + "\n" for ep in episodes]
        with open(norm_path, "w", encoding="utf-8") as handle:
            handle.writelines(lines)
        if self.cfg.write_raw_copy:
            with open(raw_path, "w", encoding="utf-8") as handle:
                handle.writelines(lines)
        with open(counters_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(counters, sort_keys=True))
        if self.cfg.storage_backend == "sqlite":
            sqlite_store = SQLiteIntermediateStoreV2(sqlite_db_path_for_round(self.storage.root, game_id, round_id))
            sqlite_store.write_episode_batch(game_id, round_id, episodes)


def _collect_episode_worker(payload: Tuple[str, str, str, CollectionConfigV2, str, Optional[Any], int, int]) -> TrajectoryEpisodeV2:
    env_factory_path, env_id, env_root, cfg, mode, instruction, round_id, ep_idx = payload
    module_name, func_name = env_factory_path.rsplit(":", 1)
    mod = importlib.import_module(module_name)
    env_factory = getattr(mod, func_name)
    try:
        env = env_factory(env_id=env_id, env_root=env_root, seed=cfg.seed + ep_idx)
    except TypeError:
        env = env_factory()
    session = EnvironmentSessionV2(env, env_id)
    policy = TrajectoryPolicyV2(seed=cfg.seed + ep_idx)
    collector = TrajectoryCollectorV2(StoragePathsV2(""), cfg)
    return collector.collect_episode(session, policy, mode, instruction, round_id, ep_idx)


def _compute_round_state_counters(episodes: List[TrajectoryEpisodeV2]) -> Dict[str, int]:
    observed_states_total = 0
    invalid_state_count = 0
    unique_pre_states: set[str] = set()
    unique_post_states: set[str] = set()
    for ep in episodes:
        for step in ep.steps:
            if step.pre_state_hash:
                unique_pre_states.add(step.pre_state_hash)
                observed_states_total += 1
            else:
                invalid_state_count += 1
            if step.post_state_hash:
                unique_post_states.add(step.post_state_hash)
                observed_states_total += 1
            else:
                invalid_state_count += 1
    return {
        "observed_states_total": int(observed_states_total),
        "unique_pre_states": int(len(unique_pre_states)),
        "unique_post_states": int(len(unique_post_states)),
        "invalid_state_count": int(invalid_state_count),
    }


def _episode_to_artifact_dict(episode: TrajectoryEpisodeV2, cfg: CollectionConfigV2) -> Dict[str, Any]:
    payload = episode.to_dict()
    for step in payload.get("steps", []):
        if not cfg.keep_observations_in_artifacts:
            step["observation"] = None
        if not cfg.keep_observation_summaries_in_artifacts:
            step["observation_summary"] = None
        if not cfg.keep_raw_info_in_artifacts and isinstance(step.get("info"), dict):
            info = dict(step["info"])
            info.pop("raw", None)
            info.pop("step_info", None)
            step["info"] = info
    return payload
