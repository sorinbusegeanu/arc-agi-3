from __future__ import annotations

import queue
from types import SimpleNamespace

from v9.runtime.derivation_merge import DerivationLeaseManager, DerivationTaskIdentity
from v9.runtime.memory_pipeline import DerivationTask
from v9.runtime.parallel_memory_coordinator import MemoryPipelineService


def test_expired_lease_retries_with_new_epoch_and_attempt() -> None:
    now = [0.0]
    manager = DerivationLeaseManager(pending_limit=2, inflight_limit=1, clock=lambda: now[0])
    identity = DerivationTaskIdentity(7, 10, 1)
    assert manager.enqueue(identity)
    first = manager.lease(duration_seconds=1.0)[0]
    now[0] = 2.0
    assert manager.expire() == (identity,)
    second = manager.lease(duration_seconds=1.0)[0]
    assert second.lease_epoch > first.lease_epoch
    assert second.attempt == 2
    assert not manager.complete(first, "stale")
    assert manager.complete(second, "accepted")


def test_failed_lease_retries_immediately_with_new_attempt() -> None:
    manager = DerivationLeaseManager()
    identity = DerivationTaskIdentity(8, 11, 1)
    manager.enqueue(identity)
    first = manager.lease()[0]
    assert manager.retry(first)
    second = manager.lease()[0]
    assert second.attempt == 2
    assert second.lease_epoch > first.lease_epoch


def test_unrelated_completed_derivation_does_not_wait_for_straggler() -> None:
    applied: list[int] = []

    class Runtime:
        watermark = 0
        config = SimpleNamespace(canonical_transaction_max_input_bytes=1024)

        def apply_derivation_results_batch(self, rows) -> None:
            applied.extend(int(row.task_id) for row in rows)

    memory = SimpleNamespace(
        ingest_workers=1,
        derivation_queue=queue.Queue(),
        derivation_result_queue=queue.Queue(),
    )
    service = MemoryPipelineService(Runtime(), memory, ingest_queue_capacity=8)
    try:
        for signature in (10, 20):
            service._consider_candidate(
                DerivationTask(0, signature, (), 2, (), 1)
            )
        assert service.pump_derivation_tasks()
        batch = memory.derivation_queue.get_nowait()
        first, second = batch.tasks
        service.derive_results[second.task_id] = SimpleNamespace(
            task_id=second.task_id,
            structural_signature=second.structural_signature,
            support=second.support,
        )
        assert service.apply_derivation_ready()
        assert applied == [second.task_id]
        assert first.task_id in service._derivation_lease_by_task

        memory.derivation_result_queue.put(
            ("derivation_task_error", first.task_id, "injected")
        )
        assert service.drain_derivation_results()
        assert service.derivation_retries == 1
        assert service.pump_derivation_tasks()
        retried = memory.derivation_queue.get_nowait().tasks[0]
        assert retried.structural_signature == first.structural_signature
        assert retried.task_id != first.task_id
    finally:
        service.shutdown_parallel_pipeline()


def test_pending_derivation_is_a_live_drain_path_when_workers_are_alive() -> None:
    class Process:
        def is_alive(self) -> bool:
            return True

    class Runtime:
        watermark = 0
        config = SimpleNamespace(canonical_transaction_max_input_bytes=1024)

    memory = SimpleNamespace(
        ingest_workers=1,
        derivation_queue=queue.Queue(maxsize=1),
        derivation_result_queue=queue.Queue(),
        derivation_processes=[Process()],
        queue_depths=lambda: {
            "derivation_queue_depth": 1,
            "derivation_result_queue_depth": 0,
        },
    )
    service = MemoryPipelineService(Runtime(), memory, ingest_queue_capacity=8)
    try:
        # Saturate the worker queue so leased work cannot be dispatched yet.
        memory.derivation_queue.put(object())
        for signature in (101, 102):
            service._consider_candidate(DerivationTask(0, signature, (), 2, (), 1))

        assert not service.pump_derivation_tasks()
        pending, inflight, _ = service.derivation_leases.counts
        assert pending == 0
        assert inflight == 2
        assert service.pending_derivation
        assert service.derivation_workers_alive()
        token = service.derivation_drain_progress_token()
        assert token[3] == 2
    finally:
        service.shutdown_parallel_pipeline()


def test_unleased_pending_derivation_is_not_retryable_but_remains_valid_work() -> None:
    class Process:
        def is_alive(self) -> bool:
            return True

    class Runtime:
        watermark = 0
        config = SimpleNamespace(canonical_transaction_max_input_bytes=1024)

    memory = SimpleNamespace(
        ingest_workers=1,
        derivation_queue=queue.Queue(),
        derivation_result_queue=queue.Queue(),
        derivation_processes=[Process()],
        queue_depths=lambda: {
            "derivation_queue_depth": 0,
            "derivation_result_queue_depth": 0,
        },
    )
    service = MemoryPipelineService(Runtime(), memory, ingest_queue_capacity=8)
    try:
        identity = DerivationTaskIdentity(201, 2, 1)
        service._derivation_candidates[identity] = DerivationTask(0, 201, (), 2, (), 1)
        assert service.derivation_leases.enqueue(identity)
        assert service.derivation_leases.counts[:2] == (1, 0)
        assert service._derivation_lease_by_task == {}
        assert service.derivation_workers_alive()
        assert service.derivation_drain_progress_token()[0] == 1
    finally:
        service.shutdown_parallel_pipeline()
