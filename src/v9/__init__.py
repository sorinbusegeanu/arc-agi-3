"""Independent ARC-AGI-3 Hydra memory runtime."""

from v9.environments.contract import EnvironmentCognitionAdapter
from v9.memory.identity import EventUid, MemoryUid
from v9.runtime import (
    LearnedDevelopmentalFeedbackProfile,
    RuntimeConfig,
    ScientificConfig,
    ScientificConfigId,
    ScientificVisibilityMode,
)
from v9.runtime.v978_conformant import V978ContinuousMemoryRuntime
from v9.runtime.derivation_publication import install_bounded_derivation_publication
from v9.runtime.residency import ResidentMemoryManager, install_bounded_residency
from v9.runtime.bounded_indexes import install_bounded_indexes
from v9.runtime.bounded_index_cleanup import install_bounded_index_cleanup
from v9.runtime.reset_memory import install_reset_memory
from v9.runtime.integration_repairs import install_integration_repairs
from v9.runtime.parallel_memory_coordinator import MemoryPipelineService
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
install_environment_viability(ContinuousMemoryRuntime, MemoryPipelineService)
install_viability_confidence(ContinuousMemoryRuntime)
install_grounding_action_index(ContinuousMemoryRuntime)
install_actor_policy_cache(ContinuousMemoryRuntime)
_runtime_package.ContinuousMemoryRuntime = ContinuousMemoryRuntime

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
    MemoryPipelineService,
)

# Final correctness layer runs after every prior compatibility/performance
# wrapper so restore, deletion, viability, IPC and snapshot invariants have one
# authoritative implementation.
import v9.runtime.runtime_integrity as _runtime_integrity_module
from v9.runtime.runtime_integrity import install_runtime_integrity as _install_runtime_integrity
_install_runtime_integrity(
    ContinuousMemoryRuntime,
    CanonicalGraph,
    MemoryPipelineService,
    _hgt_training,
)
from v9.runtime.runtime_integrity_followup import install as _install_runtime_integrity_followup
_install_runtime_integrity_followup(_runtime_integrity_module, MemoryPipelineService)

# Unique M0/grounded-M1 and first-seen normalized M1 rows are append-only.
# Publish them continuously in bounded canonical batches instead of accumulating
# a second large deferred publication debt. Repeated M1N support remains dirty
# and coalesced until the normal flush boundary.
from v9.runtime.inline_lowlevel_publication import install_inline_lowlevel_publication as _install_inline_lowlevel_publication
_install_inline_lowlevel_publication(ContinuousMemoryRuntime)

# Canonical commit remains strictly ordered, but it runs independently from the
# coordinator's queue-draining loop so actor publication and worker preparation
# continue while the authoritative graph mutation is in progress.
from v9.runtime.publication_throughput import install_publication_throughput as _install_publication_throughput
_install_publication_throughput(MemoryPipelineService)

_hgt_package.train_hgt_epoch = _hgt_training.train_hgt_epoch
from v9.runtime import epoch_runner as _epoch_runner
_epoch_runner.train_hgt_epoch = _hgt_training.train_hgt_epoch
from v9.runtime.post_sampling_progress import install as _install_post_sampling_progress
_install_post_sampling_progress(_epoch_runner, ContinuousMemoryRuntime)
from v9.runtime.concurrent_transfer_validation import install as _install_concurrent_transfer_validation
_install_concurrent_transfer_validation(_epoch_runner)
from v9.runtime.concurrent_validation_startup import install as _install_concurrent_validation_startup
_install_concurrent_validation_startup()
_runtime_package.ContinuousMemoryRuntime = ContinuousMemoryRuntime

_LAZY_EXPORTS = {
    name: ("v9.runtime", name)
    for name in (
        "CanonicalStateHandle", "CanonicalStore", "CanonicalTransaction",
        "CanonicalCommitWAL", "DevelopmentalCut", "EpochInferenceView",
        "PersistenceFrontiers", "PolicyProjection", "PolicyVersion",
        "SignatureIndexStore", "StorageGovernor", "StorageGovernorStatus",
        "TrainingEvidenceManifest", "TrainingEvidenceRecord", "TransportBatchBundle",
        "TransportSlabDescriptor", "TransportSlabPool", "TransactionOverlay",
        "WALRecoveryResult",
    )
}
_LAZY_EXPORTS.update({
    name: ("v9.research.experiment_manifest", name)
    for name in (
        "ExperimentManifest", "ExperimentManifestId", "GroundingCondition",
        "InteractionOpportunityManifest", "ReasoningCondition", "ScientificEvidenceId",
        "StructuralPriorProfile", "TrialManifest", "TrialSpec",
    )
})
_LAZY_EXPORTS.update({
    "DevelopmentalMilestoneLedger": ("v9.research.prediction_registry", "DevelopmentalMilestoneLedger"),
    "ResearchPredictionRegistry": ("v9.research.prediction_registry", "ResearchPredictionRegistry"),
})


def __getattr__(name: str):
    import importlib

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value
    return value

__all__ = [
    "ContinuousMemoryRuntime",
    "CanonicalStateHandle",
    "CanonicalStore",
    "CanonicalTransaction",
    "CanonicalCommitWAL",
    "DevelopmentalCut",
    "DevelopmentalMilestoneLedger",
    "EpochInferenceView",
    "EnvironmentCognitionAdapter",
    "EventUid",
    "MemoryUid",
    "RuntimeConfig",
    "ScientificConfig",
    "ScientificConfigId",
    "ScientificVisibilityMode",
    "LearnedDevelopmentalFeedbackProfile",
    "PersistenceFrontiers",
    "PolicyProjection",
    "PolicyVersion",
    "ExperimentManifest",
    "ExperimentManifestId",
    "GroundingCondition",
    "InteractionOpportunityManifest",
    "ReasoningCondition",
    "ScientificEvidenceId",
    "StructuralPriorProfile",
    "TrialManifest",
    "TrialSpec",
    "TransactionOverlay",
    "SignatureIndexStore",
    "StorageGovernor",
    "StorageGovernorStatus",
    "TrainingEvidenceManifest",
    "TrainingEvidenceRecord",
    "TransportBatchBundle",
    "TransportSlabDescriptor",
    "TransportSlabPool",
    "WALRecoveryResult",
    "ResearchPredictionRegistry",
]
