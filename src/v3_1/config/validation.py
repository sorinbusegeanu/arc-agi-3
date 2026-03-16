from __future__ import annotations

from pathlib import Path

from v3_1.config.schema import V31Config
from v3_1.contracts.errors import InvalidConfigurationError


def validate_config(config: V31Config) -> None:
    if config.runtime.max_rounds < 1:
        raise InvalidConfigurationError("runtime.max_rounds must be >= 1")
    if config.runtime.no_progress_budget < 1:
        raise InvalidConfigurationError("runtime.no_progress_budget must be >= 1")
    if config.ray.env_workers < 1:
        raise InvalidConfigurationError("ray.env_workers must be >= 1")
    if config.ray.analysis_workers < 1:
        raise InvalidConfigurationError("ray.analysis_workers must be >= 1")
    if config.ray.planning_helper_workers < 0:
        raise InvalidConfigurationError("ray.planning_helper_workers must be >= 0")
    if config.environment.probe_steps < 1 or config.environment.directed_steps < 1:
        raise InvalidConfigurationError("environment step budgets must be >= 1")
    if config.storage.root_dir.strip() == "":
        raise InvalidConfigurationError("storage.root_dir must not be empty")
    if config.storage.persistent_memory_flush_every_n_rounds < 0:
        raise InvalidConfigurationError("storage.persistent_memory_flush_every_n_rounds must be >= 0")
    if config.storage.max_blackboard_consequences < 1:
        raise InvalidConfigurationError("storage.max_blackboard_consequences must be >= 1")
    if config.storage.enable_persistent_memory and config.storage.persistent_memory_db_path_override:
        db_parent = Path(config.storage.persistent_memory_db_path_override).expanduser().resolve().parent
        if not db_parent.exists():
            raise InvalidConfigurationError("storage.persistent_memory_db_path_override parent must exist")
    if not config.storage.enable_persistent_memory and config.storage.load_persistent_priors_on_session_start:
        raise InvalidConfigurationError("persistent priors cannot load when persistent memory is disabled")
    durable_flags = [
        config.storage.persist_skill_stats,
        config.storage.persist_candidate_outcomes,
        config.storage.persist_failure_patterns,
        config.storage.persist_recovery_patterns,
        config.storage.persist_poi_patterns,
        config.storage.persist_trigger_patterns,
        config.storage.persist_consequence_patterns,
        config.storage.persist_entity_signatures,
        config.storage.persist_area_signatures,
        config.storage.persist_mechanic_hypotheses,
        config.storage.persist_ranker_state,
    ]
    if config.storage.enable_persistent_memory and not any(durable_flags):
        raise InvalidConfigurationError("at least one durable persistence family must be enabled when persistent memory is enabled")
    if config.planning.max_candidates < 1:
        raise InvalidConfigurationError("planning.max_candidates must be >= 1")
    if config.planning.trace_level not in {"minimal", "debug", "full"}:
        raise InvalidConfigurationError("planning.trace_level must be one of: minimal, debug, full")
    if config.memory.retry_limit < 1:
        raise InvalidConfigurationError("memory.retry_limit must be >= 1")
    if config.memory.cooldown_rounds < 0:
        raise InvalidConfigurationError("memory.cooldown_rounds must be >= 0")
    if config.hypothesis_generation.deterministic_rule_window_steps < 1:
        raise InvalidConfigurationError("hypothesis_generation.deterministic_rule_window_steps must be >= 1")
    if config.hypothesis_generation.llm_call_budget_per_round < 0:
        raise InvalidConfigurationError("hypothesis_generation.llm_call_budget_per_round must be >= 0")
    if config.hypothesis_generation.llm_retry_limit < 0:
        raise InvalidConfigurationError("hypothesis_generation.llm_retry_limit must be >= 0")
    if config.hypothesis_generation.llm_max_output_tokens < 1:
        raise InvalidConfigurationError("hypothesis_generation.llm_max_output_tokens must be >= 1")
    if config.hypothesis_generation.llm_timeout_sec <= 0:
        raise InvalidConfigurationError("hypothesis_generation.llm_timeout_sec must be > 0")
    if not (0.0 <= config.hypothesis_generation.llm_temperature <= 1.0):
        raise InvalidConfigurationError("hypothesis_generation.llm_temperature must be in [0,1]")
    if not (0.0 <= config.hypothesis_generation.llm_confidence_cap <= 1.0):
        raise InvalidConfigurationError("hypothesis_generation.llm_confidence_cap must be in [0,1]")
