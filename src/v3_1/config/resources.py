from __future__ import annotations

from dataclasses import dataclass

from v3_1.config.schema import V31Config


@dataclass(frozen=True)
class ResourceLayout:
    coordinator_cpus: float
    service_cpus: float
    worker_cpus: float


def build_resource_layout(config: V31Config) -> ResourceLayout:
    return ResourceLayout(
        coordinator_cpus=config.ray.coordinator_cpus,
        service_cpus=config.ray.service_cpus,
        worker_cpus=config.ray.worker_cpus,
    )

