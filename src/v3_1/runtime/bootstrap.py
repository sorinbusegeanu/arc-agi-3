from __future__ import annotations

import ray

from v3_1.agents.analysis_worker import AnalysisWorker
from v3_1.agents.blackboard_agent import BlackboardAgent
from v3_1.agents.env_worker_agent import EnvWorkerAgent
from v3_1.agents.memory_agent import MemoryAgent
from v3_1.agents.planner_agent import PlannerAgent
from v3_1.agents.planning_helper_worker import PlanningHelperWorker
from v3_1.agents.ranker_agent import RankerAgent
from v3_1.agents.storage_agent import StorageAgent
from v3_1.config.runtime import runtime_resources
from v3_1.learning.ranker_state import RankerState


def _maybe_init_ray(config) -> None:
    if not ray.is_initialized():
        init_kwargs = {
            "namespace": config.ray.namespace,
            "ignore_reinit_error": True,
            "include_dashboard": False,
            "log_to_driver": False,
            "local_mode": config.ray.local_mode,
        }
        if config.ray.temp_dir:
            init_kwargs["_temp_dir"] = config.ray.temp_dir
        ray.init(**init_kwargs)


def bootstrap_services(config, *, session_id: str, game_id: str):
    _maybe_init_ray(config)
    resources = runtime_resources(config)

    blackboard = BlackboardAgent.options(name=f"{session_id}:blackboard", lifetime="detached", num_cpus=config.ray.service_cpus).remote(session_id, game_id)
    memory = MemoryAgent.options(name=f"{session_id}:memory", lifetime="detached", num_cpus=config.ray.service_cpus).remote(session_id)
    planner = PlannerAgent.options(name=f"{session_id}:planner", lifetime="detached", num_cpus=config.ray.service_cpus).remote(config.planning)
    ranker = (
        RankerAgent.options(name=f"{session_id}:ranker", lifetime="detached", num_cpus=config.ray.service_cpus).remote(RankerState())
        if config.feature_flags.enable_ranker
        else None
    )
    sqlite_path = f"{config.storage.root_dir}/{session_id.replace(':', '_')}/manifests.sqlite" if config.storage.export_sqlite else None
    storage = StorageAgent.options(name=f"{session_id}:storage", lifetime="detached", num_cpus=config.ray.service_cpus).remote(
        root_dir=config.storage.root_dir,
        sqlite_path=sqlite_path,
    )

    env_workers = [
        EnvWorkerAgent.options(num_cpus=config.ray.worker_cpus).remote(
            f"env_worker:{idx}",
            env_factory=config.environment.env_factory,
            env_id=config.environment.env_id,
            env_root=config.environment.env_root,
            seed=config.environment.seed + idx,
        )
        for idx in range(max(1, resources.env_workers))
    ]
    analysis_workers = [
        AnalysisWorker.options(num_cpus=config.ray.worker_cpus).remote()
        for _ in range(max(1, resources.analysis_workers))
    ]
    helper_workers = [
        PlanningHelperWorker.options(num_cpus=config.ray.worker_cpus).remote()
        for _ in range(max(1, resources.planning_helper_workers or 1))
    ]

    return {
        "blackboard": blackboard,
        "memory": memory,
        "planner": planner,
        "ranker": ranker,
        "storage": storage,
        "env_workers": env_workers,
        "analysis_workers": analysis_workers,
        "helper_workers": helper_workers,
    }
