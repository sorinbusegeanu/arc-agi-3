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


def ingest_batch_worker_main(task_queue: Any, result_queue: Any) -> None:
    """Spawn-safe ingestion worker with crash-time shared-memory cleanup."""
    from v9.runtime import memory_pipeline_v2
    from v9.runtime.multiprocess import WorkerStop

    owned: set[str] = set()
    try:
        while True:
            item = task_queue.get()
            if isinstance(item, WorkerStop):
                return
            if not isinstance(item, memory_pipeline_v2.IngestionBatchTask):
                continue
            try:
                result = memory_pipeline_v2.prepare_commit_batch(item)
                descriptor = memory_pipeline_v2.publish_shared_batch(
                    result,
                    start_sequence=result.start_sequence,
                    end_sequence=result.end_sequence,
                    rows=len(result.rows),
                )
                owned.add(descriptor.name)
                result_queue.put(("ingest_batch_shm", result.start_sequence, result.end_sequence, descriptor))
            except BaseException as exc:
                result_queue.put(("worker_error", "ingest", int(item.start_sequence), repr(exc)))
    finally:
        cleanup_owned_shared_memory(owned)


def derivation_batch_worker_main(task_queue: Any, result_queue: Any) -> None:
    """Spawn-safe derivation worker with crash-time shared-memory cleanup."""
    from v9.runtime import memory_pipeline_v2
    from v9.runtime.multiprocess import WorkerStop

    owned: set[str] = set()
    try:
        while True:
            item = task_queue.get()
            if isinstance(item, WorkerStop):
                return
            if not isinstance(item, memory_pipeline_v2.DerivationBatchTask):
                continue
            try:
                results = tuple(memory_pipeline_v2.derive_memory(task) for task in item.tasks)
                descriptor = memory_pipeline_v2.publish_shared_batch(
                    results,
                    start_sequence=item.start_task_id,
                    end_sequence=item.end_task_id,
                    rows=len(results),
                )
                owned.add(descriptor.name)
                result_queue.put(("derivation_batch_shm", item.start_task_id, item.end_task_id, descriptor))
            except BaseException as exc:
                result_queue.put(("worker_error", "derivation", int(item.start_task_id), repr(exc)))
    finally:
        cleanup_owned_shared_memory(owned)


def install(runtime_integrity_module: Any, pipeline_cls: type) -> None:
    preserve_pipeline_viability_dispatch(pipeline_cls)
    runtime_integrity_module._VIABILITY_PROFILE_LIMIT = 256
    runtime_integrity_module._cleanup_owned_shared_memory = cleanup_owned_shared_memory

    # Multiprocessing spawn can only pickle module-level callables. The first
    # integrity layer installs local closures to add crash cleanup; replace only
    # those worker targets with equivalent top-level functions here.
    from v9.runtime import memory_pipeline_v2
    from v9.runtime import memory_worker_topology_v2

    memory_pipeline_v2.ingest_batch_worker_main = ingest_batch_worker_main
    memory_pipeline_v2.derivation_batch_worker_main = derivation_batch_worker_main
    memory_worker_topology_v2.ingest_batch_worker_main = ingest_batch_worker_main
    memory_worker_topology_v2.derivation_batch_worker_main = derivation_batch_worker_main
