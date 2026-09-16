"""Independent ARC-AGI-3 Hydra memory runtime."""

from v9.environments.contract import EnvironmentCognitionAdapter
from v9.memory.identity import EventUid, MemoryUid
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig, ScientificConfig, ScientificConfigId
from v9.runtime.multiprocess import EncodedTransition
from v9.runtime.residency import install_bounded_residency
from v9.runtime.reset_memory import install_reset_memory

# v9.7.9 exposes a full-metrics alias for residency diagnostics while preserving
# the existing metrics() API used by the runtime and dashboard.
if not hasattr(ContinuousMemoryRuntime, "full_metrics"):
    ContinuousMemoryRuntime.full_metrics = ContinuousMemoryRuntime.metrics
install_bounded_residency(ContinuousMemoryRuntime)
install_reset_memory(ContinuousMemoryRuntime)

# Encoded transitions expose one derived terminal boundary bit for legacy
# cognition paths while keeping success/failure/truncation as the source fields.
if not hasattr(EncodedTransition, "done"):
    EncodedTransition.done = property(
        lambda self: bool(self.task_success or self.task_failure or self.task_truncated)
    )

# Keep the reference prepared-batch path telemetry-equivalent to the optimized
# canonical commit path for passive symbolic observations.
_prepared_batch = ContinuousMemoryRuntime.apply_prepared_ingestion_batch
if not getattr(_prepared_batch, "_v979_symbol_telemetry", False):
    def _v979_prepared_batch(self, rows):
        prepared_rows = tuple(rows)
        before = int(self.telemetry.get("symbol_occurrences", 0))
        result = _prepared_batch(self, prepared_rows)
        expected = sum(len(getattr(row, "symbol_occurrences", ())) for row in prepared_rows)
        observed = int(self.telemetry.get("symbol_occurrences", 0)) - before
        if expected > observed:
            self.telemetry["symbol_occurrences"] = before + expected
        return result

    _v979_prepared_batch._v979_symbol_telemetry = True
    ContinuousMemoryRuntime.apply_prepared_ingestion_batch = _v979_prepared_batch

__all__ = [
    "ContinuousMemoryRuntime",
    "EnvironmentCognitionAdapter",
    "EventUid",
    "MemoryUid",
    "RuntimeConfig",
    "ScientificConfig",
    "ScientificConfigId",
]
