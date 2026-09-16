from __future__ import annotations

import queue
import time
from threading import Thread
from typing import Any

from .canonical_commit import apply_canonical_commit_batch
from .parallel_memory_coordinator import _adaptive_canonical_batch_size
from .shared_batch_transport import SharedBatchDescriptor, consume_shared_batch


_INGEST_TASK_BATCH_SIZE = 2048
_MIN_PUBLICATION_INTAKE_HIGH_WATER = 4096
_MAX_PUBLICATION_INTAKE_HIGH_WATER = 16384
_RESULT_QUEUE_DRAIN_BATCHES = 64


def _batch_rows(value: Any) -> int:
    if isinstance(value, SharedBatchDescriptor):
        return int(value.rows)
    return len(value.rows)


def _batch_end_sequence(value: Any) -> int:
    if isinstance(value, SharedBatchDescriptor):
        return int(value.end_sequence)
    return int(value.end_sequence)


def _prepared_rows_waiting(service: Any) -> int:
    return sum(_batch_rows(value) for value in service.ingest_results.values())


def _finish_canonical_commit(service: Any) -> bool:
    thread = getattr(service, "_canonical_commit_thread", None)
    if thread is None or thread.is_alive():
        return False
    thread.join()
    service.runtime._canonical_commit_inflight = False
    error = getattr(service, "_canonical_commit_error", None)
    if error is not None:
        service._canonical_commit_thread = None
        service._canonical_commit_error = None
        raise error

    result, elapsed, decode_ms, count = service._canonical_commit_result
    service._canonical_commit_thread = None
    service._canonical_commit_result = None
    service._canonical_commit_events = 0
    service.canonical_apply_seconds += float(elapsed)
    service.canonical_apply_events += int(count)
    service.ingest_result_decode_ms += float(decode_ms)
    service.last_canonical_ingest_batch = int(count)
    service.ingested += int(count)
    service._canonical_commit_batches += 1

    for candidate in result.derivation_candidates:
        service._consider_candidate(candidate)

    backlog = max(
        len(service.pending_ingest),
        _prepared_rows_waiting(service),
        max(0, int(service.sampled) - int(service.ingested)),
    )
    service.canonical_batch_size = _adaptive_canonical_batch_size(
        service.canonical_batch_size,
        backlog,
    )
    return True


def _submit_canonical_commit(service: Any) -> bool:
    if getattr(service, "_canonical_commit_thread", None) is not None:
        return False

    transport_batches: list[Any] = []
    row_count = 0
    while service.ingest_apply in service.ingest_results:
        value = service.ingest_results[service.ingest_apply]
        rows = _batch_rows(value)
        if transport_batches and row_count + rows > service.canonical_batch_size:
            break
        service.ingest_results.pop(service.ingest_apply)
        transport_batches.append(value)
        row_count += rows
        service.ingest_apply = _batch_end_sequence(value) + 1
        if row_count >= service.canonical_batch_size:
            break
    if not transport_batches:
        return False

    frozen_batches = tuple(transport_batches)
    service._canonical_commit_events = int(row_count)
    service._canonical_commit_result = None
    service._canonical_commit_error = None
    service.runtime._canonical_commit_inflight = True

    def run() -> None:
        plans: list[Any] = []
        decode_ms = 0.0
        try:
            for value in frozen_batches:
                if isinstance(value, SharedBatchDescriptor):
                    batch, batch_decode_ms = consume_shared_batch(value)
                    decode_ms += float(batch_decode_ms)
                else:
                    batch = value
                plans.extend(batch.rows)
            commit_started = time.perf_counter()
            result = apply_canonical_commit_batch(service.runtime, tuple(plans))
            commit_elapsed = time.perf_counter() - commit_started
        except BaseException as exc:
            service._canonical_commit_error = exc
            return
        service._canonical_commit_result = (
            result,
            commit_elapsed,
            decode_ms,
            len(plans),
        )

    thread = Thread(target=run, name="v9-canonical-commit", daemon=True)
    service._canonical_commit_thread = thread
    thread.start()
    return True


def install_publication_throughput(pipeline_cls: type) -> None:
    """Keep publication intake independent from preparation decoding and canonical mutation."""
    if getattr(pipeline_cls, "_publication_throughput_installed", False):
        return

    original_init = pipeline_cls.__init__
    original_diagnostics = pipeline_cls.diagnostics

    def init(self: Any, runtime: Any, memory: Any, *, ingest_queue_capacity: int) -> None:
        original_init(self, runtime, memory, ingest_queue_capacity=ingest_queue_capacity)
        self.ipc_batch_size = _INGEST_TASK_BATCH_SIZE
        ingest_workers = max(1, int(getattr(memory, "ingest_workers", 1)))
        self.ingest_local_high_water = min(
            _MAX_PUBLICATION_INTAKE_HIGH_WATER,
            max(
                _MIN_PUBLICATION_INTAKE_HIGH_WATER,
                self.ipc_batch_size * ingest_workers,
            ),
        )
        self._canonical_commit_thread = None
        self._canonical_commit_result = None
        self._canonical_commit_error = None
        self._canonical_commit_events = 0
        self._canonical_commit_batches = 0
        self.runtime._canonical_commit_inflight = False

    def drain_ingest_results(self: Any, *, block: bool = False, timeout: float = 0.0) -> bool:
        progressed = False
        started = time.perf_counter()
        first = True
        for _ in range(_RESULT_QUEUE_DRAIN_BATCHES):
            try:
                item = (
                    self.memory.ingest_result_queue.get(timeout=timeout)
                    if block and first
                    else self.memory.ingest_result_queue.get_nowait()
                )
            except queue.Empty:
                break
            first = False
            if item[0] == "worker_error":
                raise RuntimeError(f"{item[1]} worker task {item[2]} failed: {item[3]}")
            if item[0] == "ingest_batch_shm":
                descriptor = item[3]
                self.ingest_results[int(item[1])] = descriptor
                self.ingest_result_batches += 1
                self.ingest_result_rows += int(descriptor.rows)
                self.ingest_result_bytes += int(descriptor.size)
                self.ingest_result_encode_ms += float(descriptor.encode_ms)
                progressed = True
                continue
            if item[0] == "ingest_batch":
                batch = item[3]
                self.ingest_results[int(item[1])] = batch
                self.ingest_result_batches += 1
                self.ingest_result_rows += len(batch.rows)
                progressed = True
        self.ingest_result_drain_seconds += time.perf_counter() - started
        return progressed

    def apply_ingest_ready(self: Any) -> bool:
        progressed = _finish_canonical_commit(self)
        if self._canonical_commit_thread is not None:
            return progressed
        return _submit_canonical_commit(self) or progressed

    def service(self: Any) -> bool:
        progressed = _finish_canonical_commit(self)

        # Match the fast pre-9.7.8 order: feed workers first, then consume results.
        progressed = self.pump_ingest_tasks() or progressed
        progressed = self.pump_derivation_tasks() or progressed
        progressed = self.drain_ingest_results() or progressed
        progressed = self.drain_derivation_results() or progressed

        # Canonical mutation and ingest-result decoding run off the coordinator.
        if self._canonical_commit_thread is None:
            progressed = self.apply_derivation_ready() or progressed
        progressed = self.apply_ingest_ready() or progressed

        # Refill worker queues after result draining freed capacity.
        progressed = self.pump_ingest_tasks() or progressed
        progressed = self.pump_derivation_tasks() or progressed
        return progressed

    def block_for_result(self: Any, timeout: float = 0.05) -> bool:
        if _finish_canonical_commit(self):
            return True
        thread = self._canonical_commit_thread
        if thread is not None:
            thread.join(timeout=max(0.0, float(timeout)))
            return _finish_canonical_commit(self)
        if self.drain_ingest_results(block=True, timeout=float(timeout)):
            self.apply_ingest_ready()
            return True
        return self.drain_derivation_results(block=True, timeout=float(timeout))

    def diagnostics(self: Any) -> dict[str, float | int]:
        result = dict(original_diagnostics(self))
        result.update(
            {
                "canonical_commit_inflight": int(self._canonical_commit_thread is not None),
                "canonical_commit_batches": int(self._canonical_commit_batches),
                "canonical_commit_pending_events": int(self._canonical_commit_events),
                "publication_intake_high_water": int(self.ingest_local_high_water),
                "ingest_task_batch_size": int(self.ipc_batch_size),
                "prepared_ingest_batches_waiting": int(len(self.ingest_results)),
                "prepared_ingest_rows_waiting": int(_prepared_rows_waiting(self)),
                "coordinator_ingest_decode_ms": 0.0,
            }
        )
        return result

    pipeline_cls.__init__ = init
    pipeline_cls.drain_ingest_results = drain_ingest_results
    pipeline_cls.apply_ingest_ready = apply_ingest_ready
    pipeline_cls.service = service
    pipeline_cls.block_for_result = block_for_result
    pipeline_cls.diagnostics = diagnostics
    pipeline_cls._publication_throughput_installed = True
