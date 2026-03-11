from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RayRuntimeConfig:
    coordinator_actors: int = 1
    blackboard_actors: int = 1
    memory_actors: int = 1
    planner_actors: int = 1
    storage_actors: int = 1
    ranker_actors: int = 1
    env_workers: int = 4
    episode_analyzer_workers: int = 4
    planning_helper_workers: int = 4
    local_mode: bool = False


def init_ray_local(local_mode: bool = False) -> None:
    import ray

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            local_mode=local_mode,
            include_dashboard=False,
            logging_level="ERROR",
            _node_ip_address="127.0.0.1",
        )


def default_runtime_config(workers: int = 4) -> RayRuntimeConfig:
    return RayRuntimeConfig(
        env_workers=workers,
        episode_analyzer_workers=max(1, workers),
        planning_helper_workers=max(1, workers),
        local_mode=True,
    )


def bootstrap_runtime(config: RayRuntimeConfig, *, actor_kwargs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    import ray

    from .analysis_pool import EpisodeAnalyzerWorker
    from .blackboard_actor import BlackboardActor
    from .coordinator_actor import CoordinatorActor
    from .env_worker_actor import EnvWorkerActor
    from .helper_pool import PlanningHelperWorker
    from .planner_actor import PlannerActor
    from .ranker_actor import RankerActor
    from .skill_memory_actor import SkillMemoryActor
    from .storage_actor import StorageActor

    init_ray_local(local_mode=config.local_mode)
    actor_kwargs = actor_kwargs or {}
    actors = {
        "coordinator": ray.remote(CoordinatorActor).options(name="v3_coordinator").remote(**actor_kwargs.get("coordinator", {})),
        "blackboard": ray.remote(BlackboardActor).options(name="v3_blackboard").remote(**actor_kwargs.get("blackboard", {})),
        "memory": ray.remote(SkillMemoryActor).options(name="v3_memory").remote(**actor_kwargs.get("memory", {})),
        "planner": ray.remote(PlannerActor).options(name="v3_planner").remote(**actor_kwargs.get("planner", {})),
        "storage": ray.remote(StorageActor).options(name="v3_storage").remote(**actor_kwargs.get("storage", {})),
    }
    actors["ranker"] = ray.remote(RankerActor).options(name="v3_ranker").remote(**actor_kwargs.get("ranker", {})) if config.ranker_actors > 0 else None
    env_workers = [ray.remote(EnvWorkerActor).remote(**actor_kwargs.get("env_worker", {}), worker_id=f"env_{idx:03d}") for idx in range(config.env_workers)]
    analyzer_workers = [ray.remote(EpisodeAnalyzerWorker).remote(**actor_kwargs.get("episode_analyzer", {}), worker_id=f"an_{idx:03d}") for idx in range(config.episode_analyzer_workers)]
    helper_workers = [ray.remote(PlanningHelperWorker).remote(**actor_kwargs.get("planning_helper", {}), worker_id=f"help_{idx:03d}") for idx in range(config.planning_helper_workers)]
    return {
        "actors": actors,
        "env_workers": env_workers,
        "episode_analyzer_workers": analyzer_workers,
        "planning_helper_workers": helper_workers,
    }
