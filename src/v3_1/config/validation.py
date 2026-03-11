from __future__ import annotations

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
    if config.planning.max_candidates < 1:
        raise InvalidConfigurationError("planning.max_candidates must be >= 1")
    if config.memory.retry_limit < 1:
        raise InvalidConfigurationError("memory.retry_limit must be >= 1")
    if config.memory.cooldown_rounds < 0:
        raise InvalidConfigurationError("memory.cooldown_rounds must be >= 0")
