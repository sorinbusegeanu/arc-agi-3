"""Independent ARC-AGI-3 Hydra memory runtime."""

from v9.environments.contract import EnvironmentCognitionAdapter
from v9.memory.identity import EventUid, MemoryUid
from v9.runtime import RuntimeConfig, ScientificConfig, ScientificConfigId
from v9.runtime.v978_conformant import V978ContinuousMemoryRuntime
from v9.runtime.multiprocess import EncodedTransition
from v9.runtime.derivation_publication import install_bounded_derivation_publication
from v9.runtime.residency import install_bounded_residency
from v9.runtime.reset_memory import install_reset_memory
from v9.runtime.integration_repairs import install_integration_repairs
import v9.runtime as _runtime_package

# One authoritative v9.7.9 runtime: v9.7.8 symbolic grounding plus bounded
# developmental M0/M1 residency.
ContinuousMemoryRuntime = V978ContinuousMemoryRuntime
if not hasattr(ContinuousMemoryRuntime, "full_metrics"):
    ContinuousMemoryRuntime.full_metrics = ContinuousMemoryRuntime.metrics
install_bounded_derivation_publication(ContinuousMemoryRuntime)
install_bounded_residency(ContinuousMemoryRuntime)
install_reset_memory(ContinuousMemoryRuntime)
install_integration_repairs(ContinuousMemoryRuntime)
_runtime_package.ContinuousMemoryRuntime = ContinuousMemoryRuntime

# Terminal status is derived from the three authoritative boundary fields.
if not hasattr(EncodedTransition, "done"):
    EncodedTransition.done = property(
        lambda self: bool(self.task_success or self.task_failure or self.task_truncated)
    )

# High-throughput actors route through the passive-capture adapter factory so
# symbol timestamps reflect observation/action order rather than post-hoc labels.
from v9.runtime import multiprocess as _multiprocess
from v9.environments.passive_capture import install_process_factory_route as _install_process_factory_route
_install_process_factory_route(_multiprocess)

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
