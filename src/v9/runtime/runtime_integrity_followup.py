from __future__ import annotations

import inspect
from typing import Any


def preserve_pipeline_viability_dispatch(pipeline_cls: type) -> None:
    """Undo only the final dispatch suppression wrapper.

    Pipeline v2 applies CommitPlan rows directly through canonical_commit, so the
    environment-viability observation belongs to the existing dispatch wrapper.
    The public runtime.apply_prepared_ingestion_batch path has its own observation
    wrapper and does not pass through pipeline dispatch; the two paths are
    alternatives, not duplicate observations of the same transition.
    """
    current = pipeline_cls.dispatch_transition
    try:
        nonlocals = inspect.getclosurevars(current).nonlocals
    except (TypeError, ValueError):
        return
    original = nonlocals.get("original_dispatch")
    if callable(original):
        pipeline_cls.dispatch_transition = original


def cleanup_owned_shared_memory(names: set[str]) -> None:
    """Release worker-owned shared-memory segments that were never consumed."""
    from multiprocessing import shared_memory

    for name in tuple(names):
        try:
            segment = shared_memory.SharedMemory(name=name, create=False)
        except FileNotFoundError:
            continue
        try:
            segment.close()
            segment.unlink()
        except FileNotFoundError:
            pass


def install(runtime_integrity_module: Any, pipeline_cls: type) -> None:
    preserve_pipeline_viability_dispatch(pipeline_cls)
    runtime_integrity_module._VIABILITY_PROFILE_LIMIT = 256
    runtime_integrity_module._cleanup_owned_shared_memory = cleanup_owned_shared_memory

    # memory_pipeline.py owns the spawn-safe ingest and derivation worker targets.
    # Integrity follow-up only preserves viability dispatch and cleanup helpers.
