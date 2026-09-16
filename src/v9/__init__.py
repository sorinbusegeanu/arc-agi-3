"""Independent ARC-AGI-3 Hydra memory runtime."""

from v9.environments.contract import EnvironmentCognitionAdapter
from v9.memory.identity import EventUid, MemoryUid
from v9.runtime import RuntimeConfig, ScientificConfig, ScientificConfigId
from v9.runtime.v978_conformant import V978ContinuousMemoryRuntime
import v9.runtime as _runtime_package

# One authoritative v9.7.8 runtime for CLI, tests and external callers.
ContinuousMemoryRuntime = V978ContinuousMemoryRuntime
_runtime_package.ContinuousMemoryRuntime = V978ContinuousMemoryRuntime

# HGT uses canonical SymbolOccurrence payloads as the only SYMBOL authority.
from v9.hgt import training as _hgt_training
from v9.hgt.canonical_symbol_graph import install as _install_canonical_symbol_graph
_install_canonical_symbol_graph(_hgt_training)
from v9.hgt.v978_symbol_features import install as _install_v978_symbol_features
_install_v978_symbol_features(_hgt_training)

__all__ = [
    "ContinuousMemoryRuntime",
    "EnvironmentCognitionAdapter",
    "EventUid",
    "MemoryUid",
    "RuntimeConfig",
    "ScientificConfig",
    "ScientificConfigId",
]
