from __future__ import annotations

import ray

from v3_1.agents.analysis_worker import analyze_episode_task
from v3_1.agents.blackboard_agent import BlackboardAgent
from v3_1.agents.env_worker_agent import EnvWorkerAgent
from v3_1.agents.memory_agent import MemoryAgent
from v3_1.agents.mechanic_graph_agent import MechanicGraphAgent
from v3_1.agents.planner_agent import PlannerAgent
from v3_1.agents.planning_helper_worker import run_helper_task
from v3_1.agents.ranker_agent import RankerAgent
from v3_1.agents.storage_agent import StorageAgent
from v3_1.config.runtime import runtime_resources
from v3_1.learning.ranker_state import RankerState
from v3_1.llm.local_adapter_openai_compat import OpenAICompatLocalLLMAdapter
from v3_1.llm.local_adapter_stub import StubLocalLLMAdapter
from v3_1.mechanics.hypothesis_registry import HypothesisRegistry
from v3_1.storage.paths import get_persistent_memory_db_path


def _maybe_init_ray(config) -> None:
    if not ray.is_initialized():
        init_kwargs = {
            "namespace": config.ray.namespace,
            "ignore_reinit_error": True,
            "include_dashboard": False,
            "log_to_driver": False,
            "local_mode": config.ray.local_mode,
        }
        if config.ray.address:
            init_kwargs["address"] = config.ray.address
        if config.ray.temp_dir:
            init_kwargs["_temp_dir"] = config.ray.temp_dir
        ray.init(**init_kwargs)


def bootstrap_services(config, *, session_id: str, game_id: str, render_terminal: bool = False):
    _maybe_init_ray(config)
    resources = runtime_resources(config)
    persistent_db_path = (
        config.storage.persistent_memory_db_path_override
        if config.storage.enable_persistent_memory and config.storage.persistent_memory_db_path_override
        else str(get_persistent_memory_db_path(config.storage.root_dir)) if config.storage.enable_persistent_memory else None
    )

    llm_cfg = config.hypothesis_generation
    if not bool(getattr(llm_cfg, "enable_llm", False)):
        llm_reasoner_adapter = StubLocalLLMAdapter()
    elif str(getattr(llm_cfg, "llm_provider", "") or "").strip().lower() in {"openai_compat", "openai-compatible", "openai"}:
        llm_reasoner_adapter = OpenAICompatLocalLLMAdapter(
            base_url=str(getattr(llm_cfg, "llm_base_url", "") or ""),
            model_name=str(getattr(llm_cfg, "llm_model_name", "") or ""),
            api_key_env=str(getattr(llm_cfg, "llm_api_key_env", "") or ""),
            timeout_sec=float(getattr(llm_cfg, "llm_timeout_sec", 5.0) or 5.0),
            retry_limit=int(getattr(llm_cfg, "llm_retry_limit", 0) or 0),
            emit_raw_debug=bool(getattr(llm_cfg, "llm_emit_raw_debug", False)),
        )
    else:
        llm_reasoner_adapter = StubLocalLLMAdapter()

    blackboard = BlackboardAgent.options(name=f"{session_id}:blackboard", num_cpus=config.ray.service_cpus).remote(
        session_id,
        game_id,
        config.storage.max_blackboard_consequences,
    )
    memory = MemoryAgent.options(name=f"{session_id}:memory", num_cpus=config.ray.service_cpus).remote(
        session_id,
        load_persistent_priors_on_session_start=config.storage.load_persistent_priors_on_session_start,
    )
    mechanic_graph = MechanicGraphAgent.options(name=f"{session_id}:mechanic_graph", num_cpus=config.ray.service_cpus).remote(
        session_id,
        game_id,
    )
    planner = PlannerAgent.options(name=f"{session_id}:planner", num_cpus=config.ray.service_cpus).remote(config.planning)
    ranker = (
        RankerAgent.options(name=f"{session_id}:ranker", num_cpus=config.ray.service_cpus).remote(RankerState())
        if config.feature_flags.enable_ranker
        else None
    )
    sqlite_path = f"{config.storage.root_dir}/{session_id.replace(':', '_')}/manifests.sqlite" if config.storage.export_sqlite else None
    storage = StorageAgent.options(name=f"{session_id}:storage", num_cpus=config.ray.service_cpus).remote(
        root_dir=config.storage.root_dir,
        sqlite_path=sqlite_path,
        persistent_memory_db_path=persistent_db_path,
        persistence_flags={
            "persist_skill_stats": config.storage.persist_skill_stats,
            "persist_candidate_outcomes": config.storage.persist_candidate_outcomes,
            "persist_failure_patterns": config.storage.persist_failure_patterns,
            "persist_recovery_patterns": config.storage.persist_recovery_patterns,
            "persist_poi_patterns": config.storage.persist_poi_patterns,
            "persist_trigger_patterns": config.storage.persist_trigger_patterns,
            "persist_consequence_patterns": config.storage.persist_consequence_patterns,
            "persist_entity_signatures": config.storage.persist_entity_signatures,
            "persist_area_signatures": config.storage.persist_area_signatures,
            "persist_mechanic_hypotheses": config.storage.persist_mechanic_hypotheses,
            "persist_ranker_state": config.storage.persist_ranker_state,
        },
    )

    env_workers = [
        EnvWorkerAgent.options(num_cpus=config.ray.worker_cpus).remote(
            f"env_worker:{idx}",
            env_factory=config.environment.env_factory,
            env_id=config.environment.env_id,
            env_root=config.environment.env_root,
            seed=config.environment.seed + idx,
            render_terminal=render_terminal,
        )
        for idx in range(max(1, resources.env_workers))
    ]
    return {
        "blackboard": blackboard,
        "memory": memory,
        "mechanic_graph": mechanic_graph,
        "planner": planner,
        "ranker": ranker,
        "storage": storage,
        "env_workers": env_workers,
        "analysis_task": analyze_episode_task,
        "helper_task": run_helper_task,
        "hypothesis_registry": HypothesisRegistry(),
        "llm_reasoner_adapter": llm_reasoner_adapter,
        "persistent_memory_db_path": persistent_db_path,
    }
