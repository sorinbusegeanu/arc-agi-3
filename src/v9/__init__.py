"""Independent ARC-AGI-3 Hydra memory runtime."""

from v9.environments.contract import EnvironmentCognitionAdapter
from v9.memory.identity import EventUid, MemoryUid
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig, ScientificConfig, ScientificConfigId
from v9.runtime.residency import install_bounded_residency

# v9.7.9 exposes a full-metrics alias for residency diagnostics while preserving
# the existing metrics() API used by the runtime and dashboard.
if not hasattr(ContinuousMemoryRuntime, "full_metrics"):
    ContinuousMemoryRuntime.full_metrics = ContinuousMemoryRuntime.metrics
install_bounded_residency(ContinuousMemoryRuntime)

__all__ = [
    "ContinuousMemoryRuntime",
    "EnvironmentCognitionAdapter",
    "EventUid",
    "MemoryUid",
    "RuntimeConfig",
    "ScientificConfig",
    "ScientificConfigId",
]
