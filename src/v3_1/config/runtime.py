from __future__ import annotations

from dataclasses import dataclass

from v3_1.config.schema import V31Config


@dataclass(frozen=True)
class RuntimeResources:
    env_workers: int
    analysis_workers: int
    planning_helper_workers: int


def runtime_resources(config: V31Config) -> RuntimeResources:
    return RuntimeResources(
        env_workers=max(1, config.ray.env_workers),
        analysis_workers=max(1, config.ray.analysis_workers),
        planning_helper_workers=max(0, config.ray.planning_helper_workers),
    )

