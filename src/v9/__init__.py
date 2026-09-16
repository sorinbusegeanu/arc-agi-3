"""Independent ARC-AGI-3 Hydra memory runtime."""

from v9.environments.contract import EnvironmentCognitionAdapter
from v9.memory.identity import EventUid, MemoryUid
from v9.runtime import RuntimeConfig, ScientificConfig, ScientificConfigId
from v9.runtime.v978_conformant import V978ContinuousMemoryRuntime
from v9.runtime.multiprocess import EncodedTransition
from v9.runtime.derivation_publication import install_bounded_derivation_publication
from v9.runtime.residency import ResidentMemoryManager, install_bounded_residency
from v9.runtime.bounded_indexes import install_bounded_indexes
from v9.runtime.bounded_index_cleanup import install_bounded_index_cleanup
from v9.runtime.reset_memory import install_reset_memory
from v9.runtime.integration_repairs import install_integration_repairs
from v9.runtime.pipeline_service_v2 import MemoryPipelineServiceV2
from v9.runtime.environment_viability import install_environment_viability
from v9.runtime.viability_confidence import install_viability_confidence
from v9.runtime.grounding_action_index import install_grounding_action_index
from v9.runtime.actor_policy_cache import install_actor_policy_cache
from v9.runtime.publication import CanonicalGraph
import v9.runtime as _runtime_package

# One authoritative v9.7.9 runtime: v9.7.8 symbolic grounding plus bounded
# developmental residency, bounded graph indexes, and environment viability.
ContinuousMemoryRuntime = V978ContinuousMemoryRuntime
if not hasattr(ContinuousMemoryRuntime, "full_metrics"):
    ContinuousMemoryRuntime.full_metrics = ContinuousMemoryRuntime.metrics
install_bounded_derivation_publication(ContinuousMemoryRuntime)
install_bounded_residency(ContinuousMemoryRuntime)
install_bounded_indexes(ContinuousMemoryRuntime)
install_bounded_index_cleanup(CanonicalGraph)
install_reset_memory(ContinuousMemoryRuntime)
install_integration_repairs(ContinuousMemoryRuntime)
install_environment_viability(ContinuousMemoryRuntime, MemoryPipelineServiceV2)
install_viability_confidence(ContinuousMemoryRuntime)
install_grounding_action_index(ContinuousMemoryRuntime)
install_actor_policy_cache(ContinuousMemoryRuntime)
_runtime_package.ContinuousMemoryRuntime = ContinuousMemoryRuntime

# --actors is sampler-process parallelism, not merely a cap on one-job-per-game
# scheduling. Split per-game budgets when fewer jobs than actor slots exist.
from v9.runtime import parallel_memory_coordinator_v2 as _parallel_memory_coordinator_v2
from v9.runtime.actor_job_parallelism import install_actor_job_parallelism as _install_actor_job_parallelism
_install_actor_job_parallelism(_parallel_memory_coordinator_v2)

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
from v9.runtime.adaptive_exploration import install_adaptive_exploration as _install_adaptive_exploration
_install_adaptive_exploration(_multiprocess)
from v9.runtime.actor_production_telemetry import install_actor_production_telemetry as _install_actor_production_telemetry
_install_actor_production_telemetry(_multiprocess.ProcessTopology)

# HGT uses canonical SymbolOccurrence payloads as the only SYMBOL authority.
from v9.hgt import training as _hgt_training
import v9.hgt as _hgt_package
from v9.hgt.canonical_symbol_graph import install as _install_canonical_symbol_graph
_install_canonical_symbol_graph(_hgt_training)
from v9.hgt.v978_symbol_features import install as _install_v978_symbol_features
_install_v978_symbol_features(_hgt_training)

# Existing governor states now actively shrink graph/HGT working sets and
# backpressure new memory-worker work without changing scientific config ID.
from v9.runtime.pressure_control import install_pressure_control as _install_pressure_control
_install_pressure_control(
    ContinuousMemoryRuntime,
    CanonicalGraph,
    ResidentMemoryManager,
    _hgt_training,
    MemoryPipelineServiceV2,
)

# Final correctness layer runs after every prior compatibility/performance
# wrapper so restore, deletion, viability, IPC and snapshot invariants have one
# authoritative implementation.
import v9.runtime.runtime_integrity as _runtime_integrity_module
from v9.runtime.runtime_integrity import install_runtime_integrity as _install_runtime_integrity
_install_runtime_integrity(
    ContinuousMemoryRuntime,
    CanonicalGraph,
    MemoryPipelineServiceV2,
    _hgt_training,
)
from v9.runtime.runtime_integrity_followup import install as _install_runtime_integrity_followup
_install_runtime_integrity_followup(_runtime_integrity_module, MemoryPipelineServiceV2)

# Unique M0/grounded-M1 and first-seen normalized M1 rows are append-only.
# Publish them continuously in bounded canonical batches instead of accumulating
# a second large deferred publication debt. Repeated M1N support remains dirty
# and coalesced until the normal flush boundary.
from v9.runtime.inline_lowlevel_publication import install_inline_lowlevel_publication as _install_inline_lowlevel_publication
_install_inline_lowlevel_publication(ContinuousMemoryRuntime)

# Metrics layers compose several mutable runtime registries. Keep each complete
# dashboard/metrics read on one authoritative runtime cut so publication cannot
# resize graph dictionaries during telemetry traversal.
from v9.runtime.metrics_concurrency import install_metrics_concurrency as _install_metrics_concurrency
_install_metrics_concurrency(ContinuousMemoryRuntime)

# Canonical commit remains strictly ordered, but it runs independently from the
# coordinator's queue-draining loop so actor publication and worker preparation
# continue while the authoritative graph mutation is in progress.
from v9.runtime.publication_throughput import install_publication_throughput as _install_publication_throughput
_install_publication_throughput(MemoryPipelineServiceV2)

_hgt_package.train_hgt_epoch = _hgt_training.train_hgt_epoch
from v9.runtime import epoch_runner as _epoch_runner
_epoch_runner.train_hgt_epoch = _hgt_training.train_hgt_epoch
_runtime_package.ContinuousMemoryRuntime = ContinuousMemoryRuntime

__all__ = [
    "ContinuousMemoryRuntime",
    "EnvironmentCognitionAdapter",
    "EventUid",
    "MemoryUid",
    "RuntimeConfig",
    "ScientificConfig",
    "ScientificConfigId",
]
