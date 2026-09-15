"""Independent ARC-AGI-3 Hydra memory runtime."""

from v9.environments.contract import EnvironmentCognitionAdapter
from v9.memory.identity import EventUid, MemoryUid
from v9.runtime import RuntimeConfig, ScientificConfig, ScientificConfigId
from v9.runtime.completed_runtime import CompletedContinuousMemoryRuntime
import v9.runtime as _runtime_package

# Keep `from v9.runtime import ContinuousMemoryRuntime` and `from v9 import
# ContinuousMemoryRuntime` behaviorally identical. Python initializes `v9`
# before submodule imports, so installing the completed class here gives the CLI,
# tests and external callers one authoritative v9.7.8 runtime.
ContinuousMemoryRuntime = CompletedContinuousMemoryRuntime
_runtime_package.ContinuousMemoryRuntime = CompletedContinuousMemoryRuntime

__all__ = [
    "ContinuousMemoryRuntime",
    "EnvironmentCognitionAdapter",
    "EventUid",
    "MemoryUid",
    "RuntimeConfig",
    "ScientificConfig",
    "ScientificConfigId",
]
