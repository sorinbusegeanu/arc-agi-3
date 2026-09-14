from __future__ import annotations

from collections import deque
from dataclasses import replace
from itertools import islice
import queue
import time
from typing import Any

from .canonical_commit import apply_canonical_commit_batch
from .memory_pipeline import DerivationResult, IngestionTask
from .memory_pipeline_v2 import IngestionBatchTask, PreparedCommitBatch
from .parallel_memory_coordinator import _adaptive_canonical_batch_size
from .shared_batch_transport import consume_shared_batch


class MemoryPipelineServiceV2:
    def __init__(self, runtime: Any, memory: Any, *, ingest_queue_capacity: int) -> None:
        self.runtime = runtime
        self.memory = memory
        self.pending_ingest: deque[IngestionTask] = deque()
        self.pending_derivation: deque[Any] = deque()
        self.ingest_results: dict[int, PreparedCommitBatch] = {}
        self.derive_results: dict[int, DerivationResult] = {}
        self.inflight: set[int] = set()
        self.waiting_candidates: dict[int, Any] = {}
        self.last_support: dict[int, int] = {}
        self.sampled = 0
        self.ingested = 0
        self.derived = 0
        self.ingest_sequence = 1
        self.ingest_apply = 1
        self.derive_task_id = 1
        self.derive_apply = 1
        self.watermark_cursor = int(runtime.watermark)
        self.ingest_local_high_water = max(1024, int(ingest_queue_capacity) * 2)
        self.ipc_batch_size = 2048
        self.canonical_batch_size = 256
        self.last_canonical_ingest_batch = 0
        self.canonical_apply_seconds = 0.0
        self.canonical_apply_events = 0
        self.ingest_result_drain_seconds = 0.0
        self.ingest_result_batches = 0
        self.ingest_result_bytes = 0
        self.ingest_result_encode_ms = 0.0
        self.ingest_result_decode_ms = 0.0

    def dispatch_transition(self, transition: Any) -> None:
        self.sampled += 1
        canonical_sequence = self.runtime.reserve_producer_sequence(
            int(transition.actor_id), int(transition.producer_sequence)
        )
        if canonical_sequence != int(transition.producer_sequence):
            transition = replace(transition, producer_sequence=canonical_sequence)
        sequence = self.ingest_sequence
        self.ingest_sequence += 1
        self.watermark_cursor += 1
        self.pending_ingest.append(IngestionTask(sequence, self.watermark_cursor, transition))
        self.watermark_cursor += len(tuple(transition.symbols))

    def pump_ingest_tasks(self) -> bool:
        if not self.pending_ingest:
            return False
        count = min(self.ipc_batch_size, len(self.pending_ingest))
        tasks = tuple(islice(self.pending_ingest, 0, count))
        batch = IngestionBatchTask(tasks[0].sequence, tasks[-1].sequence, tasks)
        try:
            self.memory.ingest_queue.put_nowait(batch)
        except queue.Full:
            return False
        for _ in range(count):
            self.pending_ingest.popleft()
        return True

    def _consider_candidate(self, candidate: Any) -> None:
        signature = int(candidate.structural_signature)
        support = int(candidate.support)
        previous = int(self.last_support.get(signature, 0))
        if signature in self.inflight:
            current = self.waiting_candidates.get(signature)
            if current is None or int(current.support) < support:
                self.waiting_candidates[signature] = candidate
            return
        if support <= previous:
            return
        if previous >= 2 and support < previous * 2:
            return
        task = replace(candidate, task_id=self.derive_task_id)
        self.derive_task_id += 1
        self.inflight.add(signature)
        self.pending_derivation.append(task)

    def pump_derivation_tasks(self) -> bool:
        progressed = False
        for _ in range(256):
            if not self.pending_derivation:
                break
            try:
                self.memory.derivation_queue.put_nowait(self.pending_derivation[0])
            except queue.Full:
                break
            self.pending_derivation.popleft()
            progressed = True
        return progressed

    def drain_ingest_results(self, *, block: bool = False, timeout: float = 0.0) -> bool:
        progressed = False
        started = time.perf_counter()
        first = True
        for _ in range(64):
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
                batch, decode_ms = consume_shared_batch(descriptor)
                self.ingest_results[int(item[1])] = batch
                self.ingest_result_batches += 1
                self.ingest_result_bytes += int(descriptor.size)
                self.ingest_result_encode_ms += float(descriptor.encode_ms)
                self.ingest_result_decode_ms += float(decode_ms)
                progressed = True
                continue
            if item[0] == "ingest_batch":
                self.ingest_results[int(item[1])] = item[3]
                self.ingest_result_batches += 1
                progressed = True
        self.ingest_result_drain_seconds += time.perf_counter() - started
        return progressed

    def apply_ingest_ready(self) -> bool:
        plans: list[Any] = []
        while self.ingest_apply in self.ingest_results:
            batch = self.ingest_results[self.ingest_apply]
            if plans and len(plans) + len(batch.rows) > self.canonical_batch_size:
                break
            self.ingest_results.pop(self.ingest_apply)
            plans.extend(batch.rows)
            self.ingest_apply = int(batch.end_sequence) + 1
            if len(plans) >= self.canonical_batch_size:
                break
        if not plans:
            return False
        started = time.perf_counter()
        result = apply_canonical_commit_batch(self.runtime, plans)
        elapsed = time.perf_counter() - started
        self.canonical_apply_seconds += elapsed
        self.canonical_apply_events += len(plans)
        self.ingested += len(plans)
        for candidate in result.derivation_candidates:
            self._consider_candidate(candidate)
        self.last_canonical_ingest_batch = len(plans)
        backlog = max(
            len(self.pending_ingest),
            sum(len(batch.rows) for batch in self.ingest_results.values()),
            max(0, self.sampled - self.ingested),
        )
        self.canonical_batch_size = _adaptive_canonical_batch_size(
            self.canonical_batch_size, backlog
        )
        return True

    def drain_derivation_results(self, *, block: bool = False, timeout: float = 0.0) -> bool:
        progressed = False
        first = True
        for _ in range(128):
            try:
                item = (
                    self.memory.derivation_result_queue.get(timeout=timeout)
                    if block and first
                    else self.memory.derivation_result_queue.get_nowait()
                )
            except queue.Empty:
                break
            first = False
            if item[0] == "worker_error":
                raise RuntimeError(f"{item[1]} worker task {item[2]} failed: {item[3]}")
            if item[0] == "derivation":
                self.derive_results[int(item[1])] = item[2]
                progressed = True
        return progressed

    def apply_derivation_ready(self) -> bool:
        batch: list[DerivationResult] = []
        while self.derive_apply in self.derive_results and len(batch) < 128:
            batch.append(self.derive_results.pop(self.derive_apply))
            self.derive_apply += 1
        if not batch:
            return False
        self.runtime.apply_derivation_results_batch(batch)
        for result in batch:
            signature = int(result.structural_signature)
            self.last_support[signature] = max(
                int(self.last_support.get(signature, 0)), int(result.support)
            )
            self.inflight.discard(signature)
            self.derived += 1
            waiting = self.waiting_candidates.pop(signature, None)
            if waiting is not None:
                self._consider_candidate(waiting)
        return True

    def service(self) -> bool:
        progressed = self.pump_ingest_tasks()
        progressed = self.drain_ingest_results() or progressed
        progressed = self.apply_ingest_ready() or progressed
        progressed = self.pump_ingest_tasks() or progressed
        progressed = self.pump_derivation_tasks() or progressed
        progressed = self.drain_derivation_results() or progressed
        progressed = self.apply_derivation_ready() or progressed
        progressed = self.pump_derivation_tasks() or progressed
        return progressed

    def block_for_result(self, timeout: float = 0.05) -> bool:
        progressed = self.drain_ingest_results(block=True, timeout=timeout)
        progressed = self.apply_ingest_ready() or progressed
        if progressed:
            return True
        progressed = self.drain_derivation_results(block=True, timeout=timeout)
        return self.apply_derivation_ready() or progressed

    def diagnostics(self) -> dict[str, float | int]:
        return {
            "sampled_steps": self.sampled,
            "ingested_steps": self.ingested,
            "derivations_applied": self.derived,
            "sampling_backlog": max(0, self.sampled - self.ingested),
            "coordinator_pending_ingest": len(self.pending_ingest),
            "coordinator_pending_derivation": len(self.pending_derivation),
            "derivation_inflight": len(self.inflight),
            "canonical_apply_rate": self.canonical_apply_events / max(1e-9, self.canonical_apply_seconds),
            "canonical_apply_latency_ms": 1000.0 * self.canonical_apply_seconds / max(1, self.canonical_apply_events),
            "canonical_batch_size": self.canonical_batch_size,
            "canonical_batch_last_applied": self.last_canonical_ingest_batch,
            "ingest_result_batches": self.ingest_result_batches,
            "ingest_result_drain_ms_per_batch": 1000.0 * self.ingest_result_drain_seconds / max(1, self.ingest_result_batches),
            "ingest_result_encode_ms_per_batch": self.ingest_result_encode_ms / max(1, self.ingest_result_batches),
            "ingest_result_decode_ms_per_batch": self.ingest_result_decode_ms / max(1, self.ingest_result_batches),
            "ingest_result_bytes_per_batch": self.ingest_result_bytes / max(1, self.ingest_result_batches),
            "ingest_ipc_batch_size": self.ipc_batch_size,
        }
