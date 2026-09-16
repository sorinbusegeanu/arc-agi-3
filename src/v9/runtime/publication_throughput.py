from __future__ import annotations

import time
from threading import Thread
from typing import Any

from .canonical_commit import apply_canonical_commit_batch
from .parallel_memory_coordinator import _adaptive_canonical_batch_size


_MIN_PUBLICATION_INTAKE_HIGH_WATER = 1024
_MAX_PUBLICATION_INTAKE_HIGH_WATER = 4096


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

    result, elapsed = service._canonical_commit_result
    count = int(service._canonical_commit_events)
    service._canonical_commit_thread = None
    service._canonical_commit_result = None
    service._canonical_commit_events = 0
    service.canonical_apply_seconds += float(elapsed)
    service.canonical_apply_events += count
    service.last_canonical_ingest_batch = count
    service.ingested += count
    service._canonical_commit_batches += 1

    for candidate in result.derivation_candidates:
        service._consider_candidate(candidate)

    backlog = max(
        len(service.pending_ingest),
        len(service.ingest_results),
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

    plans: list[Any] = []
    while service.ingest_apply in service.ingest_results:
        batch = service.ingest_results[service.ingest_apply]
        if plans and len(plans) + len(batch.rows) > service.canonical_batch_size:
            break
        service.ingest_results.pop(service.ingest_apply)
        plans.extend(batch.rows)
        service.ingest_apply = int(batch.end_sequence) + 1
        if len(plans) >= service.canonical_batch_size:
            break
    if not plans:
        return False

    frozen_plans = tuple(plans)
    service._canonical_commit_events = len(frozen_plans)
    service._canonical_commit_result = None
    service._canonical_commit_error = None
    service.runtime._canonical_commit_inflight = True

    def run() -> None:
        started = time.perf_counter()
        try:
            result = apply_canonical_commit_batch(service.runtime, frozen_plans)
        except BaseException as exc:
            service._canonical_commit_error = exc
            return
        service._canonical_commit_result = (result, time.perf_counter() - started)

    thread = Thread(target=run, name="v9-canonical-commit", daemon=True)
    service._canonical_commit_thread = thread
    thread.start()
    return True


def install_publication_throughput(pipeline_cls: type) -> None:
    """Overlap canonical commit with bounded publication intake and worker preparation."""
    if getattr(pipeline_cls, "_publication_throughput_installed", False):
        return

    original_init = pipeline_cls.__init__
    original_diagnostics = pipeline_cls.diagnostics

    def init(self: Any, runtime: Any, memory: Any, *, ingest_queue_capacity: int) -> None:
        original_init(self, runtime, memory, ingest_queue_capacity=ingest_queue_capacity)
        capacity = max(1, int(ingest_queue_capacity))
        self.ingest_local_high_water = min(
            _MAX_PUBLICATION_INTAKE_HIGH_WATER,
            max(
                int(self.ingest_local_high_water),
                _MIN_PUBLICATION_INTAKE_HIGH_WATER,
                capacity * 4,
            ),
        )
        self._canonical_commit_thread = None
        self._canonical_commit_result = None
        self._canonical_commit_error = None
        self._canonical_commit_events = 0
        self._canonical_commit_batches = 0
        self.runtime._canonical_commit_inflight = False

    def apply_ingest_ready(self: Any) -> bool:
        progressed = _finish_canonical_commit(self)
        if self._canonical_commit_thread is not None:
            return progressed
        return _submit_canonical_commit(self) or progressed

    def service(self: Any) -> bool:
        progressed = _finish_canonical_commit(self)
        progressed = self.pump_ingest_tasks() or progressed
        progressed = self.pump_derivation_tasks() or progressed
        progressed = self.drain_ingest_results() or progressed
        progressed = self.drain_derivation_results() or progressed

        if self._canonical_commit_thread is None:
            progressed = self.apply_derivation_ready() or progressed
        progressed = self.apply_ingest_ready() or progressed
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
                "prepared_ingest_batches_waiting": int(len(self.ingest_results)),
            }
        )
        return result

    pipeline_cls.__init__ = init
    pipeline_cls.apply_ingest_ready = apply_ingest_ready
    pipeline_cls.service = service
    pipeline_cls.block_for_result = block_for_result
    pipeline_cls.diagnostics = diagnostics
    pipeline_cls._publication_throughput_installed = True
