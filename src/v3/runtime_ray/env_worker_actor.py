from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

from codex_baseline_v2.executor.online_executor import run_offline_local_execution
from codex_baseline_v2.runtime.environment_session import EnvironmentSessionV2
from codex_baseline_v2.runtime.trajectory_collector import CollectionConfigV2, TrajectoryCollectorV2
from codex_baseline_v2.runtime.trajectory_policy import TrajectoryPolicyV2
from codex_baseline_v2.shared.schemas import ControllerInstructionV2, ExecutorOutcomeV2
from codex_baseline_v2.shared.storage import StoragePathsV2

from .messages import ExecutorOutcome, ExecutorRequest, RawEpisode, RawStep


def _load_env_factory(path: str):
    module_name, func_name = path.rsplit(":", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, func_name)


def _episode_to_raw(episode, round_id: int, mode: str, worker_id: str) -> RawEpisode:
    return RawEpisode(
        game_id=episode.game_id,
        episode_id=episode.episode_id,
        round_id=round_id,
        mode=mode,
        env_worker_id=worker_id,
        steps=[
            RawStep(
                game_id=episode.game_id,
                episode_id=episode.episode_id,
                step_idx=step.step_idx,
                action=step.action.to_dict(),
                reward=step.reward,
                done=step.done,
                observation=step.observation,
                info=dict(step.info or {}),
            )
            for step in episode.steps
        ],
        done=episode.done,
        win=episode.win,
        seed=episode.seed,
        metadata=dict(episode.metadata or {}),
    )


class EnvWorkerActor:
    def __init__(self, worker_id: str, env_factory: Optional[str] = None, env_id: Optional[str] = None, env_root: Optional[str] = None, collection_cfg: Optional[Dict[str, Any]] = None) -> None:
        self.worker_id = worker_id
        self.env_factory_path = env_factory
        self.env_id = env_id
        self.env_root = env_root
        self.collection_cfg = collection_cfg or {}
        self._env = None
        self._session = None
        if env_factory and env_id and env_root:
            self._ensure_session()

    def _ensure_session(self) -> EnvironmentSessionV2:
        if self._session is not None:
            return self._session
        env_factory = _load_env_factory(self.env_factory_path)
        try:
            self._env = env_factory(env_id=self.env_id, env_root=self.env_root)
        except TypeError:
            self._env = env_factory()
        self._session = EnvironmentSessionV2(self._env, self.env_id or "unknown_game")
        return self._session

    def collect_probe_episode(self, game_id: str, round_id: int, episode_idx: int, mode: str = "random_probe") -> RawEpisode:
        session = self._ensure_session()
        cfg = CollectionConfigV2(
            episodes=1,
            max_steps_per_episode=int(self.collection_cfg.get("max_steps_per_episode", 40)),
            max_steps_per_instruction=int(self.collection_cfg.get("max_steps_per_instruction", 40)),
            seed=self.collection_cfg.get("seed"),
            action_repeat_limit=int(self.collection_cfg.get("action_repeat_limit", 4)),
        )
        collector = TrajectoryCollectorV2(StoragePathsV2(""), cfg)
        policy = TrajectoryPolicyV2(seed=cfg.seed)
        episode = collector.collect_episode(session, policy, mode, None, round_id, episode_idx)
        return _episode_to_raw(episode, round_id, mode, self.worker_id)

    def execute_directed(self, request: ExecutorRequest) -> ExecutorOutcome:
        session = self._ensure_session()
        instruction = ControllerInstructionV2.from_dict(request.instruction or {})
        outcome: ExecutorOutcomeV2 = run_offline_local_execution(
            session=session,
            instruction=instruction,
            max_steps=int(request.max_steps),
            episode_id=f"ray_round{request.round_id:03d}_ep{request.episode_idx:05d}",
        )
        raw_episode = RawEpisode(
            game_id=request.game_id,
            episode_id=outcome.episode_id,
            round_id=request.round_id,
            mode=request.mode,
            env_worker_id=self.worker_id,
            steps=[],
            done=outcome.done,
            win=bool(outcome.reward_delta and outcome.reward_delta > 0),
            metadata={"selected_skill_id": request.selected_skill_id},
        )
        return ExecutorOutcome(
            game_id=request.game_id,
            round_id=request.round_id,
            episode_idx=request.episode_idx,
            raw_episode=raw_episode,
            execution_outcome=outcome.to_dict(),
            negative_planning_feedback=bool(getattr(outcome, "negative_planning_feedback", False)),
            termination_reason=outcome.outcome_summary,
        )
