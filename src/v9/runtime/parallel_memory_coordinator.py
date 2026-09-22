from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from itertools import islice
import queue
import time
from threading import Thread
from typing import Any

from .actor_job_parallelism import expand_jobs_for_actor_limit
from .canonical_commit import apply_canonical_commit_batch
from .derivation_merge import DerivationLease, DerivationLeaseManager, DerivationTaskIdentity
from .memory_pipeline import DerivationBatchTask, DerivationResult, IngestionBatchTask, IngestionTask, PreparedCommitBatch
from .memory_worker_topology import MemoryWorkerTopology
from .multiprocess import ActorDone, ActorError, ProcessTopology
from .shared_batch_transport import (
    SlabOwnership,
    TransportSlabDescriptor,
    TransportSlabPool,
    consume_shared_batch,
    decode_transport_rows,
    decode_transport_value,
    encode_transport_rows,
)


_ACTOR_COMPLETION_GRACE_SECONDS = 2.0
_PIPELINE_DRAIN_STALL_SECONDS = 30.0
_CANONICAL_BATCH_MIN = 256
_CANONICAL_BATCH_MAX = 4096
_INGEST_RESULT_DECODE_BATCHES = 2
_INGEST_RESULT_DECODE_BUDGET_SECONDS = 0.005
_PUBLICATION_BATCH_MIN = 1024
_PUBLICATION_BATCH_MAX = 8192
_PUBLICATION_PRIORITY_DRAINS = 4
_HGT_WRITER_BATCHES = 8


def _adaptive_publication_batch_size(backlog: int) -> int:
    backlog = max(0, int(backlog))
    if backlog >= 25_000:
        return _PUBLICATION_BATCH_MAX
    if backlog >= 10_000:
        return 4096
    if backlog >= 2_000:
        return 2048
    return _PUBLICATION_BATCH_MIN


def _drain_publication_batch(
    publication_queue: Any,
    budget: int,
    *,
    transport_pool: TransportSlabPool | None = None,
    preserve_shard_done: bool = True,
    first_timeout: float = 0.0,
    byte_budget: int | None = None,
    carried_bytes: list[int] | None = None,
) -> tuple[tuple[Any, ...], tuple[int, ...]]:
    transitions: list[Any] = []
    shard_done: list[int] = []
    marker_items: list[Any] = []
    payload_bytes = 0
    for index in range(max(0, int(budget))):
        if transitions and byte_budget is not None and payload_bytes >= int(byte_budget):
            break
        try:
            if index == 0 and float(first_timeout) > 0.0:
                item = publication_queue.get(timeout=float(first_timeout))
            else:
                item = publication_queue.get_nowait()
        except queue.Empty:
            break
        if item[0] == "transition":
            transitions.append(item[3])
            payload_bytes += len(encode_transport_rows((item[3],)))
        elif item[0] == "transition_batch":
            transitions.extend(item[3].transitions)
            payload_bytes += len(encode_transport_rows(tuple(item[3].transitions)))
        elif item[0] == "transition_slab":
            if transport_pool is None:
                raise RuntimeError("transport slab publication requires its owning pool")
            descriptor = TransportSlabDescriptor.unpack(item[3])
            payload_bytes += int(descriptor.length)
            transport_pool.transfer_to_coordinator(descriptor)
            try:
                transitions.extend(
                    decode_transport_rows(transport_pool.read(descriptor), descriptor)
                )
            finally:
                transport_pool.release(descriptor, owner=SlabOwnership.COORDINATOR_OWNED)
        elif item[0] == "shard_error":
            raise RuntimeError(str(item[3]))
        elif item[0] == "shard_done":
            shard_done.append(int(item[1]))
            if preserve_shard_done:
                marker_items.append(item)
        if len(transitions) >= int(budget):
            break
    if preserve_shard_done:
        for item in marker_items:
            publication_queue.put(item)
    if carried_bytes is not None:
        carried_bytes.append(int(payload_bytes))
    return tuple(transitions), tuple(shard_done)


def _consume_transport_value(pool: TransportSlabPool, packed: bytes) -> tuple[Any, TransportSlabDescriptor, float]:
    descriptor = TransportSlabDescriptor.unpack(packed)
    pool.transfer_to_coordinator(descriptor)
    started = time.perf_counter()
    try:
        value = decode_transport_value(pool.read(descriptor))
    finally:
        pool.release(descriptor, owner=SlabOwnership.COORDINATOR_OWNED)
    return value, descriptor, 1000.0 * (time.perf_counter() - started)


class _AsyncHGTWriter:
    def __init__(self, dataset: Any, *, max_batches: int = _HGT_WRITER_BATCHES) -> None:
        self.dataset = dataset
        self.queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(max_batches)))
        self.error: BaseException | None = None
        self.append_seconds = 0.0
        self.rows = 0
        self.thread = Thread(target=self._run, name="v9-hgt-dataset-writer", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                return
            try:
                started = time.perf_counter()
                self.dataset.append_batch(item)
                self.append_seconds += time.perf_counter() - started
                self.rows += len(item)
            except BaseException as exc:
                self.error = exc
                return

    def _raise_if_failed(self) -> None:
        if self.error is not None:
            raise RuntimeError("HGT dataset writer failed") from self.error

    def submit(self, transitions: tuple[Any, ...]) -> None:
        if not transitions:
            return
        while True:
            self._raise_if_failed()
            try:
                self.queue.put(tuple(transitions), timeout=0.05)
                return
            except queue.Full:
                continue

    def close(self) -> None:
        if not self.thread.is_alive():
            self._raise_if_failed()
            return
        while True:
            self._raise_if_failed()
            try:
                self.queue.put(None, timeout=0.05)
                break
            except queue.Full:
                continue
        self.thread.join()
        self._raise_if_failed()


def _adaptive_canonical_batch_size(current: int, backlog: int) -> int:
    current = max(_CANONICAL_BATCH_MIN, min(_CANONICAL_BATCH_MAX, int(current)))
    backlog = max(0, int(backlog))
    if backlog >= current * 2 and current < _CANONICAL_BATCH_MAX:
        return min(_CANONICAL_BATCH_MAX, current * 2)
    if backlog <= current // 2 and current > _CANONICAL_BATCH_MIN:
        return max(_CANONICAL_BATCH_MIN, current // 2)
    return current


@dataclass(frozen=True, slots=True)
class ProcessActorResult:
    actor_id: int
    game_id: str
    steps: int
    positive_boundaries: int
    negative_boundaries: int
    episode_boundaries: int
    resets: int
    policy_refreshes: int = 0
    task_successes: int = 0
    task_failures: int = 0
    task_truncations: int = 0
    levels_completed: int = 0
    epoch_inference_view_id: str = ""


def _reconcile_actor_liveness(
    active: dict[int, tuple[int, Any]],
    clean_exit_without_done: dict[int, float],
    *,
    now: float | None = None,
    grace_seconds: float = _ACTOR_COMPLETION_GRACE_SECONDS,
) -> int:
    current_time = time.monotonic() if now is None else float(now)
    missing = 0
    for actor_id, (_, process) in tuple(active.items()):
        exitcode = process.exitcode
        if exitcode is None:
            clean_exit_without_done.pop(actor_id, None)
            continue
        if exitcode != 0:
            raise RuntimeError(f"actor process {actor_id} exited before completion with code {exitcode}")
        missing += 1
        first_seen = clean_exit_without_done.setdefault(actor_id, current_time)
        if current_time - first_seen >= float(grace_seconds):
            process_name = getattr(process, "name", f"actor-{actor_id}")
            raise RuntimeError(
                f"actor {actor_id} ({process_name}) exited cleanly but no ActorDone was received "
                f"within {float(grace_seconds):.1f}s; actor completion queue protocol failed"
            )
    return missing


class MemoryPipelineService:
    def __init__(self, runtime: Any, memory: Any, *, ingest_queue_capacity: int) -> None:
        self.runtime = runtime
        self.memory = memory
        self.pending_ingest: deque[IngestionTask] = deque()
        self.pending_ingest_batches: deque[IngestionBatchTask] = deque()
        self.pending_ingest_batch_rows = 0
        self.pending_ingest_batch_bytes = 0
        self.pending_derivation: deque[Any] = deque()
        self.derivation_leases = DerivationLeaseManager(
            pending_limit=4096,
            inflight_limit=256,
            completed_limit=2048,
        )
        self._derivation_candidates: dict[DerivationTaskIdentity, Any] = {}
        self._derivation_lease_by_task: dict[int, DerivationLease] = {}
        self.derivation_retries = 0
        self.derivation_stale_results = 0
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
        runtime_config = getattr(runtime, "config", None)
        self.ingest_local_byte_high_water = max(
            1,
            int(getattr(runtime_config, "canonical_transaction_max_input_bytes", 64 * 1024 * 1024)),
        )
        self.outstanding_ingest_bytes = 0
        self.ipc_batch_size = 256
        self.canonical_batch_size = 256
        self.last_canonical_ingest_batch = 0
        self.canonical_apply_seconds = 0.0
        self.canonical_apply_events = 0
        self.ingest_result_drain_seconds = 0.0
        self.ingest_result_batches = 0
        self.ingest_result_rows = 0
        self.ingest_result_bytes = 0
        self.ingest_result_encode_ms = 0.0
        self.ingest_result_decode_ms = 0.0
        self.producer_sequence_batch_seconds = 0.0
        self.watermark_allocation_seconds = 0.0
        self.ingestion_task_build_seconds = 0.0
        self.publication_to_ingest_queue_seconds = 0.0
        self.publication_dispatch_seconds = 0.0
        self.publication_dispatch_rows = 0
        self.publication_batches = 0

    @staticmethod
    def _split_input_bytes(total_bytes: int, rows: int) -> tuple[int, ...]:
        if rows <= 0 or total_bytes < 0:
            raise ValueError("input byte allocation requires non-negative bytes and positive rows")
        base, remainder = divmod(int(total_bytes), int(rows))
        return tuple(base + int(index < remainder) for index in range(int(rows)))

    def _check_ingest_admission(self, rows: int, input_bytes: int) -> None:
        outstanding_rows = max(0, int(self.sampled) - int(self.ingested))
        if outstanding_rows + int(rows) > int(self.ingest_local_high_water):
            raise RuntimeError(
                f"publication batch exceeds bounded ingest high-water: outstanding={outstanding_rows} "
                f"incoming={rows} high_water={self.ingest_local_high_water}"
            )
        if self.outstanding_ingest_bytes + int(input_bytes) > int(self.ingest_local_byte_high_water):
            raise RuntimeError(
                "publication batch exceeds bounded ingest byte high-water: "
                f"outstanding={self.outstanding_ingest_bytes} incoming={input_bytes} "
                f"high_water={self.ingest_local_byte_high_water}"
            )

    def release_ingest_input_bytes(self, input_bytes: int) -> None:
        self.outstanding_ingest_bytes -= int(input_bytes)
        if self.outstanding_ingest_bytes < 0:
            raise RuntimeError("outstanding ingest byte accounting underflow")

    def dispatch_transition(self, transition: Any) -> None:
        input_bytes = len(encode_transport_rows((transition,)))
        self._check_ingest_admission(1, input_bytes)
        self.sampled += 1
        canonical_sequence = self.runtime.reserve_producer_sequence(int(transition.actor_id), int(transition.producer_sequence))
        if canonical_sequence != int(transition.producer_sequence):
            transition = replace(transition, producer_sequence=canonical_sequence)
        scientific = self.runtime.config.scientific
        if not bool(scientific.symbolic_grounding_enabled) and tuple(transition.symbols):
            transition = replace(transition, symbols=())
        sequence = self.ingest_sequence
        self.ingest_sequence += 1
        self.watermark_cursor += 1
        symbol_limit = min(int(scientific.symbol_budget_per_window), int(scientific.max_symbol_facts_per_window))
        symbol_count = min(len(tuple(transition.symbols)), symbol_limit)
        self.pending_ingest.append(
            IngestionTask(
                sequence,
                self.watermark_cursor,
                transition,
                symbol_limit,
                int(scientific.symbol_payload_bytes),
                int(scientific.max_cross_modal_facts_per_macro_event),
                str(scientific.symbol_deduplication_policy),
                int(scientific.symbol_window_time_span),
                str(scientific.symbol_codec_name),
                int(scientific.symbol_codec_version),
                input_bytes,
            )
        )
        self.outstanding_ingest_bytes += input_bytes
        self.watermark_cursor += symbol_count

    def dispatch_transitions_batch(
        self,
        transitions: tuple[Any, ...],
        *,
        carried_bytes: int | None = None,
    ) -> int:
        transitions = tuple(transitions)
        if not transitions:
            return 0
        input_bytes = (
            len(encode_transport_rows(transitions))
            if carried_bytes is None
            else int(carried_bytes)
        )
        if input_bytes <= 0:
            raise ValueError("non-empty publication batches require positive carried bytes")
        self._check_ingest_admission(len(transitions), input_bytes)

        dispatch_started = time.perf_counter()
        sequence_started = time.perf_counter()
        reserve_batch = getattr(self.runtime, "reserve_producer_sequences_batch", None)
        requests = tuple((int(row.actor_id), int(row.producer_sequence)) for row in transitions)
        if callable(reserve_batch):
            canonical_sequences = tuple(reserve_batch(requests))
        else:
            canonical_sequences = tuple(
                self.runtime.reserve_producer_sequence(producer_id, proposed)
                for producer_id, proposed in requests
            )
        if len(canonical_sequences) != len(transitions):
            raise RuntimeError("producer sequence batch reservation length mismatch")
        self.producer_sequence_batch_seconds += time.perf_counter() - sequence_started

        scientific = self.runtime.config.scientific
        grounded = bool(scientific.symbolic_grounding_enabled)
        prepared: list[Any] = []
        for transition, canonical_sequence in zip(transitions, canonical_sequences):
            if int(canonical_sequence) != int(transition.producer_sequence):
                transition = replace(transition, producer_sequence=int(canonical_sequence))
            if not grounded and tuple(transition.symbols):
                transition = replace(transition, symbols=())
            prepared.append(transition)

        symbol_limit = min(int(scientific.symbol_budget_per_window), int(scientific.max_symbol_facts_per_window))
        watermark_started = time.perf_counter()
        cursor = int(self.watermark_cursor)
        first_sequence = int(self.ingest_sequence)
        watermarks: list[int] = []
        for transition in prepared:
            cursor += 1
            watermarks.append(cursor)
            cursor += min(len(tuple(transition.symbols)), symbol_limit)
        self.watermark_cursor = cursor
        self.ingest_sequence += len(prepared)
        self.watermark_allocation_seconds += time.perf_counter() - watermark_started

        build_started = time.perf_counter()
        task_input_bytes = self._split_input_bytes(input_bytes, len(prepared))
        tasks = tuple(
            IngestionTask(
                first_sequence + index,
                watermarks[index],
                transition,
                symbol_limit,
                int(scientific.symbol_payload_bytes),
                int(scientific.max_cross_modal_facts_per_macro_event),
                str(scientific.symbol_deduplication_policy),
                int(scientific.symbol_window_time_span),
                str(scientific.symbol_codec_name),
                int(scientific.symbol_codec_version),
                task_input_bytes[index],
            )
            for index, transition in enumerate(prepared)
        )
        batches = tuple(
            IngestionBatchTask(
                chunk[0].sequence,
                chunk[-1].sequence,
                chunk,
                task_input_bytes[start : start + len(chunk)],
            )
            for start in range(0, len(tasks), max(1, int(self.ipc_batch_size)))
            for chunk in (tasks[start : start + max(1, int(self.ipc_batch_size))],)
        )
        self.ingestion_task_build_seconds += time.perf_counter() - build_started

        self.sampled += len(tasks)
        enqueue_started = time.perf_counter()
        pending_mode = bool(self.pending_ingest_batches)
        for batch in batches:
            if not pending_mode:
                try:
                    self.memory.ingest_queue.put_nowait(batch)
                    continue
                except queue.Full:
                    pending_mode = True
            rows = len(batch.tasks)
            if self.pending_ingest_batch_rows + rows > int(self.ingest_local_high_water):
                raise RuntimeError("bounded pending ingest-batch buffer exceeded")
            if self.pending_ingest_batch_bytes + batch.input_bytes > int(self.ingest_local_byte_high_water):
                raise RuntimeError("bounded pending ingest-batch byte buffer exceeded")
            self.pending_ingest_batches.append(batch)
            self.pending_ingest_batch_rows += rows
            self.pending_ingest_batch_bytes += batch.input_bytes
        self.outstanding_ingest_bytes += input_bytes
        self.publication_to_ingest_queue_seconds += time.perf_counter() - enqueue_started
        self.publication_dispatch_seconds += time.perf_counter() - dispatch_started
        self.publication_dispatch_rows += len(tasks)
        self.publication_batches += 1
        return len(tasks)

    def pump_ingest_tasks(self) -> bool:
        if self.pending_ingest_batches:
            batch = self.pending_ingest_batches[0]
            try:
                self.memory.ingest_queue.put_nowait(batch)
            except queue.Full:
                return False
            self.pending_ingest_batches.popleft()
            self.pending_ingest_batch_rows -= len(batch.tasks)
            self.pending_ingest_batch_bytes -= int(batch.input_bytes)
            if self.pending_ingest_batch_rows < 0 or self.pending_ingest_batch_bytes < 0:
                raise RuntimeError("pending ingest-batch accounting underflow")
            return True
        if not self.pending_ingest:
            return False
        count = min(self.ipc_batch_size, len(self.pending_ingest))
        tasks = tuple(islice(self.pending_ingest, 0, count))
        task_bytes = tuple(int(task.input_bytes) for task in tasks)
        batch = IngestionBatchTask(tasks[0].sequence, tasks[-1].sequence, tasks, task_bytes)
        try:
            self.memory.ingest_queue.put_nowait(batch)
        except queue.Full:
            return False
        for _ in range(count):
            self.pending_ingest.popleft()
        return True

    def _consider_candidate(self, candidate: Any) -> None:
        from .scientific_modes import ScientificVisibilityMode

        if (
            getattr(
                getattr(self.runtime.config, "scientific", None),
                "scientific_visibility_mode",
                ScientificVisibilityMode.ASYNC_DEVELOPMENT,
            )
            is ScientificVisibilityMode.MATCHED_REASONING
        ):
            # Matched epochs derive from the selected branch's immutable
            # post-sampling cut.  Dispatching these candidates now would let
            # worker completion timing publish M2-M4 during sampling.
            self.runtime.set_telemetry_gauge(
                "matched_derivation_candidates_deferred_to_cut", 1
            )
            return
        signature = int(candidate.structural_signature)
        support = int(candidate.support)
        previous = int(self.last_support.get(signature, 0))
        if support <= previous:
            return
        if previous >= 2 and support < previous * 2:
            return
        identity = DerivationTaskIdentity(signature, support, 1)
        self._derivation_candidates[identity] = candidate
        self.derivation_leases.enqueue(identity)

    def _expire_derivation_leases(self) -> bool:
        expired = set(self.derivation_leases.expire())
        if not expired:
            return False
        expired_tasks = {
            task_id
            for task_id, lease in self._derivation_lease_by_task.items()
            if lease.identity in expired
        }
        for task_id in expired_tasks:
            self._derivation_lease_by_task.pop(task_id, None)
        if expired_tasks:
            self.pending_derivation = deque(
                task for task in self.pending_derivation
                if int(task.task_id) not in expired_tasks
            )
        for identity in expired:
            if not any(
                lease.identity.signature == identity.signature
                for lease in self._derivation_lease_by_task.values()
            ):
                self.inflight.discard(identity.signature)
        self.derivation_retries += len(expired)
        return True

    def _retry_derivation_task(self, task_id: int) -> bool:
        lease = self._derivation_lease_by_task.pop(int(task_id), None)
        if lease is None:
            self.derivation_stale_results += 1
            return False
        retried = self.derivation_leases.retry(lease)
        if retried:
            self.derivation_retries += 1
        if not any(
            current.identity.signature == lease.identity.signature
            for current in self._derivation_lease_by_task.values()
        ):
            self.inflight.discard(lease.identity.signature)
        return retried

    def derivation_drain_progress_token(self) -> tuple[int, ...]:
        lease_pending, lease_inflight, lease_completed = self.derivation_leases.counts
        queue_depths = self.memory.queue_depths() if hasattr(self.memory, "queue_depths") else {}
        return (
            int(lease_pending),
            int(lease_inflight),
            int(lease_completed),
            int(len(self.pending_derivation)),
            int(len(self._derivation_lease_by_task)),
            int(len(self.derive_results)),
            int(self.derived),
            int(self.derivation_retries),
            int(queue_depths.get("derivation_queue_depth", -1)),
            int(queue_depths.get("derivation_result_queue_depth", -1)),
        )

    def derivation_workers_alive(self) -> bool:
        processes = tuple(getattr(self.memory, "derivation_processes", ()) or ())
        return bool(processes) and any(process.is_alive() for process in processes)

    def pump_derivation_tasks(self) -> bool:
        progressed = self._expire_derivation_leases()
        if not self.pending_derivation:
            for lease in self.derivation_leases.lease(count=64):
                candidate = self._derivation_candidates.get(lease.identity)
                if candidate is None:
                    raise RuntimeError("leased derivation identity has no candidate payload")
                task = replace(candidate, task_id=self.derive_task_id)
                self.derive_task_id += 1
                self._derivation_lease_by_task[int(task.task_id)] = lease
                self.inflight.add(lease.identity.signature)
                self.pending_derivation.append(task)
        if not self.pending_derivation:
            return progressed
        count = min(64, len(self.pending_derivation))
        tasks = tuple(islice(self.pending_derivation, 0, count))
        batch = DerivationBatchTask(int(tasks[0].task_id), int(tasks[-1].task_id), tasks)
        try:
            self.memory.derivation_queue.put_nowait(batch)
        except queue.Full:
            return progressed
        for _ in range(count):
            self.pending_derivation.popleft()
        return True

    def drain_ingest_results(self, *, block: bool = False, timeout: float = 0.0) -> bool:
        progressed = False
        started = time.perf_counter()
        first = True
        for _ in range(_INGEST_RESULT_DECODE_BATCHES):
            try:
                item = self.memory.ingest_result_queue.get(timeout=timeout) if block and first else self.memory.ingest_result_queue.get_nowait()
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
                self.ingest_result_rows += int(descriptor.rows)
                self.ingest_result_bytes += int(descriptor.size)
                self.ingest_result_encode_ms += float(descriptor.encode_ms)
                self.ingest_result_decode_ms += float(decode_ms)
                progressed = True
            elif item[0] == "ingest_batch_slab":
                batch, descriptor, decode_ms = _consume_transport_value(
                    self.memory.ingest_result_pool,
                    item[4],
                )
                self.ingest_results[int(item[1])] = batch
                self.ingest_result_batches += 1
                self.ingest_result_rows += int(descriptor.rows)
                self.ingest_result_bytes += int(descriptor.length)
                self.ingest_result_encode_ms += float(item[5])
                self.ingest_result_decode_ms += float(decode_ms)
                progressed = True
            elif item[0] == "ingest_batch":
                batch = item[3]
                self.ingest_results[int(item[1])] = batch
                self.ingest_result_batches += 1
                self.ingest_result_rows += len(batch.rows)
                progressed = True
            if progressed and time.perf_counter() - started >= _INGEST_RESULT_DECODE_BUDGET_SECONDS:
                break
        self.ingest_result_drain_seconds += time.perf_counter() - started
        return progressed

    def apply_ingest_ready(self) -> bool:
        plans: list[Any] = []
        input_bytes = 0
        transaction_row_limit = min(
            int(self.canonical_batch_size),
            int(getattr(self.runtime.config, "canonical_transaction_max_rows", 1024)),
        )
        while self.ingest_apply in self.ingest_results:
            batch = self.ingest_results[self.ingest_apply]
            if plans and len(plans) + len(batch.rows) > transaction_row_limit:
                break
            self.ingest_results.pop(self.ingest_apply)
            plans.extend(batch.rows)
            input_bytes += int(getattr(batch, "input_bytes", 0))
            self.ingest_apply = int(batch.end_sequence) + 1
            if len(plans) >= transaction_row_limit:
                break
        if not plans:
            return False
        started = time.perf_counter()
        result = apply_canonical_commit_batch(self.runtime, plans, input_bytes=input_bytes)
        elapsed = time.perf_counter() - started
        self.canonical_apply_seconds += elapsed
        self.canonical_apply_events += len(plans)
        self.last_canonical_ingest_batch = len(plans)
        self.ingested += len(plans)
        self.release_ingest_input_bytes(input_bytes)

        # Continuous canonical ingestion bypasses runtime.apply_prepared_ingestion_batch,
        # so residency maintenance must be serviced here as well. The manager keeps
        # each visit bounded and internally enforces the configured insertion interval.
        resident_manager = getattr(self.runtime, "_resident_memory", None)
        if resident_manager is not None:
            resident_manager.maybe_compact()

        for candidate in result.derivation_candidates:
            self._consider_candidate(candidate)
        backlog = max(len(self.pending_ingest), len(self.ingest_results), max(0, self.sampled - self.ingested))
        self.canonical_batch_size = _adaptive_canonical_batch_size(self.canonical_batch_size, backlog)
        return True

    def drain_derivation_results(self, *, block: bool = False, timeout: float = 0.0) -> bool:
        progressed = False
        first = True
        for _ in range(16):
            try:
                item = self.memory.derivation_result_queue.get(timeout=timeout) if block and first else self.memory.derivation_result_queue.get_nowait()
            except queue.Empty:
                break
            first = False
            if item[0] == "worker_error":
                raise RuntimeError(f"{item[1]} worker task {item[2]} failed: {item[3]}")
            if item[0] == "derivation_task_error":
                progressed = self._retry_derivation_task(int(item[1])) or progressed
                continue
            if item[0] == "derivation_batch_shm":
                descriptor = item[3]
                batch, _decode_ms = consume_shared_batch(descriptor)
                for row in batch:
                    self.derive_results[int(row.task_id)] = row
                progressed = True
            elif item[0] == "derivation_batch_slab":
                batch, _descriptor, _decode_ms = _consume_transport_value(
                    self.memory.derivation_result_pool,
                    item[3],
                )
                for row in batch:
                    self.derive_results[int(row.task_id)] = row
                progressed = True
            elif item[0] == "derivation_batch":
                for row in item[3]:
                    self.derive_results[int(row.task_id)] = row
                progressed = True
        return progressed

    def apply_derivation_ready(self) -> bool:
        if not self.derive_results:
            return False
        progressed = False
        handled = False
        for task_id in sorted(tuple(self.derive_results)):
            handled = True
            row = self.derive_results.pop(task_id)
            lease = self._derivation_lease_by_task.get(task_id)
            if lease is None or not self.derivation_leases.accepts(lease):
                self.derivation_stale_results += 1
                continue
            newer_support = max(
                (
                    identity.target_support
                    for identity in self._derivation_candidates
                    if identity.signature == lease.identity.signature
                ),
                default=lease.identity.target_support,
            )
            if lease.identity.target_support < newer_support:
                self.derivation_leases.complete(lease, "STALE_RESULT")
                self._derivation_lease_by_task.pop(task_id, None)
                self._derivation_candidates.pop(lease.identity, None)
                self.derivation_stale_results += 1
                if not any(
                    current.identity.signature == lease.identity.signature
                    for current in self._derivation_lease_by_task.values()
                ):
                    self.inflight.discard(lease.identity.signature)
                continue
            self.runtime.apply_derivation_results_batch((row,))
            if not self.derivation_leases.complete(lease, row):
                raise RuntimeError("derivation lease changed during canonical publication")
            self._derivation_lease_by_task.pop(task_id, None)
            self.derived += 1
            progressed = True
            signature = int(row.structural_signature)
            self.last_support[signature] = int(row.support)
            self._derivation_candidates.pop(lease.identity, None)
            if not any(
                current.identity.signature == signature
                for current in self._derivation_lease_by_task.values()
            ):
                self.inflight.discard(signature)
        self.derive_apply = max(self.derive_apply, max(self.derive_results, default=0) + 1)
        return progressed or handled

    def service(self) -> bool:
        progressed = self.pump_ingest_tasks()
        progressed = self.pump_derivation_tasks() or progressed
        progressed = self.drain_ingest_results() or progressed
        progressed = self.apply_ingest_ready() or progressed
        progressed = self.drain_derivation_results() or progressed
        progressed = self.apply_derivation_ready() or progressed
        return progressed

    def block_for_result(self, timeout: float = 0.05) -> bool:
        if self.drain_ingest_results(block=True, timeout=float(timeout)):
            return True
        return self.drain_derivation_results(block=True, timeout=float(timeout))

    def diagnostics(self) -> dict[str, float | int]:
        rows = max(1, self.ingest_result_rows)
        lease_pending, lease_inflight, lease_completed = self.derivation_leases.counts
        return {
            "sampled_steps": self.sampled,
            "ingested_steps": self.ingested,
            "sampling_backlog": max(0, self.sampled - self.ingested),
            "canonical_apply_latency_ms": 1000.0 * self.canonical_apply_seconds / max(1, self.canonical_apply_events),
            "canonical_batch_size": self.last_canonical_ingest_batch,
            "canonical_batch_target": self.canonical_batch_size,
            "ingest_result_batches": self.ingest_result_batches,
            "ingest_result_rows": self.ingest_result_rows,
            "ingest_result_bytes": self.ingest_result_bytes,
            "ingest_result_encode_ms": self.ingest_result_encode_ms,
            "ingest_result_decode_ms": self.ingest_result_decode_ms,
            "ingest_result_drain_ms": 1000.0 * self.ingest_result_drain_seconds,
            "ipc_bytes_per_transition": self.ingest_result_bytes / rows,
            "decode_ms_per_transition": self.ingest_result_decode_ms / rows,
            "pending_ingest_batches": len(self.pending_ingest_batches),
            "pending_ingest_batch_rows": self.pending_ingest_batch_rows,
            "pending_ingest_batch_bytes": self.pending_ingest_batch_bytes,
            "outstanding_ingest_bytes": self.outstanding_ingest_bytes,
            "ingest_byte_high_water": self.ingest_local_byte_high_water,
            "producer_sequence_batch_ms": 1000.0 * self.producer_sequence_batch_seconds,
            "watermark_allocation_ms": 1000.0 * self.watermark_allocation_seconds,
            "ingestion_task_build_ms": 1000.0 * self.ingestion_task_build_seconds,
            "publication_to_ingest_queue_ms": 1000.0 * self.publication_to_ingest_queue_seconds,
            "publication_dispatch_ms": 1000.0 * self.publication_dispatch_seconds,
            "publication_dispatch_ms_per_transition": 1000.0 * self.publication_dispatch_seconds / max(1, self.publication_dispatch_rows),
            "publication_batches": self.publication_batches,
            "derivation_lease_pending": lease_pending,
            "derivation_lease_inflight": lease_inflight,
            "derivation_lease_completed": lease_completed,
            "derivation_lease_retries": self.derivation_retries,
            "derivation_stale_results": self.derivation_stale_results,
        }


def run_parallel_memory_jobs(
    runtime: Any,
    jobs: list[tuple[int, Any, int, int]],
    *,
    actor_limit: int,
    stage_workers: int,
    shards: int,
    queue_capacity: int,
    epsilon: float,
    stagnation_by_game: dict[str, float] | None = None,
    env_root: str | None,
    alfred_backend_factory: str | None,
    start_method: str | None,
    progress_interval_seconds: float,
    ingest_workers: int,
    derivation_workers: int,
    ingest_queue_capacity: int,
    derivation_queue_capacity: int,
    publication_queue_capacity: int,
    actor_view_refresh_steps: int = 64,
    actor_view_refresh_ms: float = 250.0,
    allow_policy_refresh: bool = True,
    bound_epoch_view: Any | None = None,
    hgt_dataset: Any | None = None,
    evidence_branch: str = "",
    evaluation_only: bool = False,
) -> list[ProcessActorResult]:
    original_job_count = len(jobs)
    jobs = expand_jobs_for_actor_limit(jobs, actor_limit)
    runtime.set_telemetry_gauge("sampling_jobs_before_actor_split", original_job_count)
    runtime.set_telemetry_gauge("sampling_jobs_after_actor_split", len(jobs))
    target_actor_slots = min(int(actor_limit), len(jobs))
    topology = ProcessTopology(
        actors=target_actor_slots,
        stage_workers=int(stage_workers),
        shards=int(shards),
        queue_capacity=max(int(queue_capacity), int(publication_queue_capacity)),
        start_method=start_method,
    )
    topology.start_workers()
    memory = MemoryWorkerTopology(
        topology.ctx,
        ingest_workers=int(ingest_workers),
        derivation_workers=int(derivation_workers),
        ingest_queue_capacity=int(ingest_queue_capacity),
        derivation_queue_capacity=int(derivation_queue_capacity),
        ingest_result_queue_capacity=max(int(publication_queue_capacity), 1024),
        derivation_result_queue_capacity=max(1024, int(derivation_queue_capacity)),
    )
    memory.start()
    pipeline = MemoryPipelineService(runtime, memory, ingest_queue_capacity=int(ingest_queue_capacity))

    pending = deque(jobs)
    active: dict[int, tuple[int, Any]] = {}
    free_slots = list(range(topology.actors))
    results: list[ProcessActorResult] = []
    clean_exit_without_done: dict[int, float] = {}
    grounded_influence_total = 0
    peak_active_actors = 0
    initial_policy = (
        bound_epoch_view.policy_projection.snapshot
        if bound_epoch_view is not None
        else runtime.actor_policy_snapshot()
    )
    bound_epoch_view_id = "" if bound_epoch_view is None else str(bound_epoch_view.identity.checksum)
    published_policy_generation = int(initial_policy.generation)
    next_policy_publish = time.monotonic() + max(0.01, float(actor_view_refresh_ms) / 1000.0)
    run_nonce = int(runtime.watermark)
    requested_steps = sum(int(row[2]) for row in jobs)
    started_at = time.monotonic()
    next_progress = started_at + max(1.0, float(progress_interval_seconds))
    clean_shutdown = False
    dataset_start_count = int(getattr(hgt_dataset, "count", 0)) if hgt_dataset is not None else 0
    if evaluation_only and hgt_dataset is not None:
        raise ValueError("evaluation-only jobs cannot write an HGT training dataset")
    hgt_writer = _AsyncHGTWriter(hgt_dataset) if hgt_dataset is not None else None
    publication_queue_drain_seconds = 0.0
    publication_queue_drain_rows = 0
    publication_dispatch_seconds = 0.0
    publication_batches = 0
    last_publication_batch_size = 0

    runtime.set_telemetry_gauge("actor_slots_target", target_actor_slots)
    runtime.set_telemetry_gauge("actor_process_start_method", topology.actor_start_method)
    tracked_shm_bytes = topology.tracked_shm_bytes + memory.tracked_shm_bytes
    runtime.set_telemetry_gauge("transport_tracked_shm_bytes", topology.tracked_shm_bytes)
    runtime.set_telemetry_gauge("compiled_result_tracked_shm_bytes", memory.tracked_shm_bytes)
    runtime.set_telemetry_gauge("tracked_shm_bytes", tracked_shm_bytes)
    runtime.set_telemetry_gauge("transport_descriptor_bytes", TransportSlabDescriptor.binary_size())
    runtime.__dict__["_tracked_shm_bytes"] = tracked_shm_bytes

    def actor_produced_steps() -> int:
        return int(getattr(topology, "produced_steps", pipeline.sampled))

    def _publish_actor_process_telemetry() -> None:
        runtime.set_telemetry_gauge("active_actor_processes", len(active))
        runtime.set_telemetry_gauge("peak_active_actor_processes", peak_active_actors)
        runtime.set_telemetry_gauge("actor_processes_total_launched", len(topology.actor_processes))
        runtime.set_telemetry_gauge("actor_slots_filled", len(active))
        runtime.set_telemetry_gauge("actor_process_pids", ",".join(str(process.pid) for process in topology.actor_processes if process.pid is not None))

    def launch_one() -> bool:
        nonlocal peak_active_actors
        if not pending or not free_slots:
            return False
        actor_id, spec, steps, seed = pending.popleft()
        slot = free_slots.pop(0)
        launch_started = time.perf_counter()
        game_name = str(getattr(spec, "display_name", getattr(spec, "game_id", "")))
        stagnation = float((stagnation_by_game or {}).get(game_name, 0.0))
        topology.start_actor(
            index=slot,
            spec=spec,
            actor_id=actor_id,
            steps=steps,
            seed=seed,
            env_root=env_root,
            adapter_factory_path="v9.cli:make_adapter",
            alfred_backend_factory=alfred_backend_factory,
            run_nonce=run_nonce,
            initial_policy=(runtime.actor_policy_snapshot() if allow_policy_refresh else initial_policy),
            epsilon=float(epsilon),
            stagnation=stagnation,
            policy_refresh_steps=int(actor_view_refresh_steps),
            policy_refresh_ms=float(actor_view_refresh_ms),
            policy_refresh_enabled=bool(allow_policy_refresh),
            epoch_inference_view_id=bound_epoch_view_id,
            evidence_branch=str(evidence_branch),
            publish_transitions=not bool(evaluation_only),
        )
        process = topology.actor_processes[-1]
        if process.pid is None:
            raise RuntimeError(f"actor {actor_id} failed to start an OS process")
        active[actor_id] = (slot, process)
        peak_active_actors = max(peak_active_actors, len(active))
        runtime.set_telemetry_gauge("actor_launch_latency_ms", 1000.0 * (time.perf_counter() - launch_started))
        runtime.set_telemetry_gauge("actors_launched", len(topology.actor_processes))
        runtime.set_telemetry_gauge("live_environment_instances", len(active))
        runtime.set_telemetry_gauge("available_actor_slots", len(free_slots))
        runtime.set_telemetry_gauge("pending_environment_jobs", len(pending))
        _publish_actor_process_telemetry()
        return True

    def launch_available_slots() -> int:
        launched = 0
        while pending and free_slots:
            if not launch_one():
                break
            launched += 1
        return launched

    def dispatch_published_transitions(transitions: tuple[Any, ...], carried_bytes: int) -> int:
        nonlocal publication_dispatch_seconds, publication_batches, last_publication_batch_size
        if not transitions:
            return 0
        started = time.perf_counter()
        if hgt_writer is not None:
            hgt_writer.submit(transitions)
        accepted = int(
            pipeline.dispatch_transitions_batch(
                transitions,
                carried_bytes=int(carried_bytes),
            )
        )
        if accepted != len(transitions):
            raise RuntimeError(f"publication batch dispatch mismatch: expected={len(transitions)} accepted={accepted}")
        publication_dispatch_seconds += time.perf_counter() - started
        publication_batches += 1
        last_publication_batch_size = len(transitions)
        return accepted

    def drain_publication_queue() -> int:
        nonlocal publication_queue_drain_seconds, publication_queue_drain_rows
        available = int(pipeline.ingest_local_high_water) - max(0, int(pipeline.sampled) - int(pipeline.ingested))
        available_bytes = int(pipeline.ingest_local_byte_high_water) - int(pipeline.outstanding_ingest_bytes)
        if available < 64 or available_bytes < int(topology.transport_pool.slab_bytes):
            return 0
        backlog = max(0, actor_produced_steps() - int(pipeline.sampled))
        # A transport envelope contains at most 64 rows. Reserve its maximum
        # possible overshoot before removing it from the multiprocessing queue.
        budget = min(int(available) - 63, _adaptive_publication_batch_size(backlog))
        byte_budget = max(1, available_bytes - int(topology.transport_pool.slab_bytes) + 1)
        carried_bytes: list[int] = []
        started = time.perf_counter()
        transitions, _ = _drain_publication_batch(
            topology.publication_queue,
            budget,
            transport_pool=topology.transport_pool,
            byte_budget=byte_budget,
            carried_bytes=carried_bytes,
        )
        publication_queue_drain_seconds += time.perf_counter() - started
        publication_queue_drain_rows += len(transitions)
        if not transitions:
            return 0
        return dispatch_published_transitions(transitions, carried_bytes[0])

    def service_publication_priority() -> bool:
        progressed = False
        backlog = max(0, actor_produced_steps() - int(pipeline.sampled))
        drains = _PUBLICATION_PRIORITY_DRAINS if backlog >= 2 * _PUBLICATION_BATCH_MIN else 1
        for _ in range(drains):
            drained = drain_publication_queue()
            progressed = bool(drained) or progressed
            if not drained or int(pipeline.sampled) - int(pipeline.ingested) >= int(pipeline.ingest_local_high_water):
                break
        progressed = pipeline.service() or progressed
        return progressed

    def drain_actor_results() -> bool:
        nonlocal grounded_influence_total
        progressed = False
        for _ in range(256):
            try:
                done = topology.result_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(done, ActorError):
                raise RuntimeError(f"actor {done.actor_id} ({done.game_id}) failed: {done.message}\n{done.traceback_text}")
            if not isinstance(done, ActorDone):
                continue
            entry = active.pop(done.actor_id, None)
            if entry is None:
                continue
            slot, _ = entry
            clean_exit_without_done.pop(done.actor_id, None)
            free_slots.append(slot)
            free_slots.sort()
            grounded_influence_total += int(getattr(done, "grounded_action_influence", 0))
            runtime.set_telemetry_gauge("grounded_action_influence", grounded_influence_total)
            runtime.telemetry["grounded_action_influence"] = grounded_influence_total
            results.append(
                ProcessActorResult(
                    done.actor_id,
                    done.game_id,
                    done.steps,
                    done.positive_boundaries,
                    done.negative_boundaries,
                    done.episode_boundaries,
                    done.resets,
                    done.policy_refreshes,
                    done.task_successes,
                    done.task_failures,
                    done.task_truncations,
                        done.levels_completed,
                        done.epoch_inference_view_id,
                )
            )
            runtime.set_telemetry_gauge("live_environment_instances", len(active))
            runtime.set_telemetry_gauge("available_actor_slots", len(free_slots))
            runtime.set_telemetry_gauge("pending_environment_jobs", len(pending))
            _publish_actor_process_telemetry()
            progressed = True
        return progressed

    def telemetry() -> None:
        if hgt_dataset is not None:
            runtime.set_telemetry_gauge("hgt_dataset_written_transitions", int(hgt_dataset.count) - dataset_start_count)
            runtime.set_telemetry_gauge("hgt_dataset_bytes", int(hgt_dataset.bytes_written))
            if hgt_writer is not None:
                runtime.set_telemetry_gauge("hgt_append_ms", 1000.0 * hgt_writer.append_seconds)
                runtime.set_telemetry_gauge("hgt_append_ms_per_transition", 1000.0 * hgt_writer.append_seconds / max(1, hgt_writer.rows))
        elapsed = max(1e-9, time.monotonic() - started_at)
        produced = actor_produced_steps()
        published = int(pipeline.sampled)
        ingested = int(pipeline.ingested)
        for key, value in memory.queue_depths().items():
            runtime.set_telemetry_gauge(key, value)
        gauges = pipeline.diagnostics()
        gauges.update(
            {
                "sampled_steps": produced,
                "actor_produced_steps": produced,
                "causally_admitted_steps": published,
                "publication_drained_steps": published,
                "ingested_steps": ingested,
                "sampling_backlog": max(0, produced - ingested),
                "publication_backlog": max(0, produced - published),
                "canonical_ingest_backlog": max(0, published - ingested),
                "active_actor_processes": len(active),
                "peak_active_actor_processes": peak_active_actors,
                "actor_slots_target": target_actor_slots,
                "actor_slots_filled": len(active),
                "actor_processes_total_launched": len(topology.actor_processes),
                "live_environment_instances": len(active),
                "available_actor_slots": len(free_slots),
                "pending_environment_jobs": len(pending),
                "coordinator_action_requests": 0,
                "policy_snapshot_generation": int(published_policy_generation),
                "active_ingest_workers": int(ingest_workers),
                "active_derivation_workers": int(derivation_workers),
                "sampling_rate": produced / elapsed,
                "publication_rate": published / elapsed,
                "ingestion_rate": ingested / elapsed,
                "derivation_rate": pipeline.derived / elapsed,
                "actors_exited_without_done": len(clean_exit_without_done),
                "canonical_pipeline_version": 3,
                "publication_queue_drain_ms": 1000.0 * publication_queue_drain_seconds,
                "publication_queue_drain_rows": publication_queue_drain_rows,
                "publication_batch_size": last_publication_batch_size,
                "publication_dispatch_ms": 1000.0 * publication_dispatch_seconds,
                "publication_dispatch_ms_per_transition": 1000.0 * publication_dispatch_seconds / max(1, published),
                "publication_batches_per_second": publication_batches / elapsed,
            }
        )
        for key, value in gauges.items():
            runtime.set_telemetry_gauge(key, value)
        _publish_actor_process_telemetry()

    def progress() -> None:
        telemetry()
        diag = dict(runtime.unified_telemetry.diagnostic_metrics())
        produced = actor_produced_steps()
        published = int(pipeline.sampled)
        ingested = int(pipeline.ingested)
        pct = 100.0 * produced / requested_steps if requested_steps else 100.0
        print(
            f"{time.strftime('[%H:%M]')} {pct:5.1f}% sampled={produced}/{requested_steps} "
            f"published={published} ingested={ingested} "
            f"games_finished={len(results)}/{len(jobs)} actors_active={len(active)}/{target_actor_slots} "
            f"actors_peak={peak_active_actors} rate={float(diag.get('ingestion_rate', 0.0)):.0f}/s "
            f"pub_backlog={max(0, produced - published)} ingest_backlog={max(0, published - ingested)}",
            flush=True,
        )

    def wait_for_published_transitions(expected: int) -> None:
        nonlocal next_progress
        last_progress_at = time.monotonic()
        last_state = (pipeline.sampled, pipeline.ingested)
        while pipeline.sampled < int(expected):
            progressed = service_publication_priority()
            state = (pipeline.sampled, pipeline.ingested)
            if progressed or state != last_state:
                last_progress_at = time.monotonic()
                last_state = state
            now = time.monotonic()
            if now >= next_progress:
                progress()
                next_progress = now + max(1.0, float(progress_interval_seconds))
            if now - last_progress_at >= _PIPELINE_DRAIN_STALL_SECONDS:
                raise RuntimeError(
                    f"actor transition drain stalled: expected={expected} produced={actor_produced_steps()} "
                    f"published={pipeline.sampled} ingested={pipeline.ingested} pending_ingest={len(pipeline.pending_ingest)} "
                    f"pending_ingest_batches={len(pipeline.pending_ingest_batches)}"
                )
            if not progressed:
                time.sleep(0.001)

    try:
        initial_launched = launch_available_slots()
        if target_actor_slots > 0 and len(active) != target_actor_slots:
            raise RuntimeError(
                f"failed to prefill actor slots: target={target_actor_slots} active={len(active)} launched={initial_launched}"
            )
        if target_actor_slots > 1 and len({process.pid for _, process in active.values()}) != target_actor_slots:
            raise RuntimeError("actor slots did not produce distinct OS processes")
        runtime.set_telemetry_gauge("actor_prefill_complete", 1)
        runtime.set_telemetry_gauge("actor_prefill_processes", len(active))
        _publish_actor_process_telemetry()

        while active or pending:
            progressed = service_publication_priority()
            progressed = drain_actor_results() or progressed
            launched = launch_available_slots()
            progressed = bool(launched) or progressed
            missing = _reconcile_actor_liveness(active, clean_exit_without_done)
            runtime.set_telemetry_gauge("actors_exited_without_done", missing)
            now = time.monotonic()
            if allow_policy_refresh and now >= next_policy_publish:
                snapshot = runtime.actor_policy_snapshot()
                if int(snapshot.generation) > int(published_policy_generation):
                    topology.publish_policy_snapshot(tuple(slot for slot, _ in active.values()), snapshot)
                    published_policy_generation = int(snapshot.generation)
                next_policy_publish = now + max(0.01, float(actor_view_refresh_ms) / 1000.0)
            if now >= next_progress:
                progress()
                next_progress = now + max(1.0, float(progress_interval_seconds))
            if not progressed:
                time.sleep(0.001)

        expected_transitions = sum(int(row.steps) for row in results)
        if not evaluation_only:
            wait_for_published_transitions(expected_transitions)
        topology.join_actor_workers()
        topology.signal_stage_stop()
        topology.join_stage_workers()
        topology.stop_shard_workers()

        shard_done = 0
        shard_drain_started = time.monotonic()
        last_shard_progress = shard_drain_started
        completed_shards: set[int] = set()
        while shard_done < int(shards):
            available = int(pipeline.ingest_local_high_water) - max(0, int(pipeline.sampled) - int(pipeline.ingested))
            available_bytes = int(pipeline.ingest_local_byte_high_water) - int(pipeline.outstanding_ingest_bytes)
            if available < 64 or available_bytes < int(topology.transport_pool.slab_bytes):
                progressed = pipeline.service()
                if not progressed:
                    pipeline.block_for_result(timeout=0.05)
                if time.monotonic() - last_shard_progress >= _PIPELINE_DRAIN_STALL_SECONDS:
                    raise RuntimeError("final shard transition drain stalled under ingestion backpressure")
                continue
            backlog = max(0, actor_produced_steps() - int(pipeline.sampled))
            budget = min(int(available) - 63, _adaptive_publication_batch_size(backlog))
            byte_budget = max(1, available_bytes - int(topology.transport_pool.slab_bytes) + 1)
            carried_bytes: list[int] = []
            started = time.perf_counter()
            transitions, completed = _drain_publication_batch(
                topology.publication_queue,
                budget,
                transport_pool=topology.transport_pool,
                preserve_shard_done=False,
                first_timeout=0.05,
                byte_budget=byte_budget,
                carried_bytes=carried_bytes,
            )
            publication_queue_drain_seconds += time.perf_counter() - started
            publication_queue_drain_rows += len(transitions)
            if transitions:
                dispatch_published_transitions(transitions, carried_bytes[0])
                last_shard_progress = time.monotonic()
            for shard_id in completed:
                completed_shards.add(int(shard_id))
                last_shard_progress = time.monotonic()
            shard_done = len(completed_shards)
            if not transitions and not completed:
                pipeline.service()
                exited = {index: process.exitcode for index, process in enumerate(topology.shard_processes) if process.exitcode is not None}
                if exited and len(completed_shards) < int(shards):
                    missing = sorted(set(range(int(shards))) - completed_shards)
                    raise RuntimeError(
                        f"shard drain terminated before completion: completed={sorted(completed_shards)} missing={missing} exitcodes={exited}"
                    )
                if time.monotonic() - last_shard_progress >= _PIPELINE_DRAIN_STALL_SECONDS:
                    missing = sorted(set(range(int(shards))) - completed_shards)
                    raise RuntimeError(
                        f"shard drain stalled for {_PIPELINE_DRAIN_STALL_SECONDS:.0f}s: completed={sorted(completed_shards)} "
                        f"missing={missing} produced={actor_produced_steps()} published={pipeline.sampled} ingested={pipeline.ingested}"
                    )
                continue
            pipeline.service()
        runtime.set_telemetry_gauge("shard_drain_seconds", time.monotonic() - shard_drain_started)
        topology.join_shard_workers()

        last_progress_at = time.monotonic()
        while pipeline.ingested < pipeline.sampled or pipeline.pending_ingest_batches or pipeline.pending_ingest or pipeline.ingest_results:
            progressed = pipeline.service()
            if progressed:
                last_progress_at = time.monotonic()
            elif not pipeline.block_for_result():
                if time.monotonic() - last_progress_at >= _PIPELINE_DRAIN_STALL_SECONDS:
                    raise RuntimeError("ingestion drain stalled")
        memory.signal_ingest_stop()
        memory.join_ingest()

        last_progress_at = time.monotonic()
        last_progress_token = pipeline.derivation_drain_progress_token()
        while (
            pipeline.pending_derivation
            or pipeline.inflight
            or pipeline.derive_results
            or any(pipeline.derivation_leases.counts[:2])
        ):
            progressed = pipeline.service()
            if progressed:
                last_progress_at = time.monotonic()
                last_progress_token = pipeline.derivation_drain_progress_token()
                continue
            if pipeline.block_for_result():
                last_progress_at = time.monotonic()
                last_progress_token = pipeline.derivation_drain_progress_token()
                continue

            progress_token = pipeline.derivation_drain_progress_token()
            if progress_token != last_progress_token:
                last_progress_token = progress_token
                last_progress_at = time.monotonic()
                continue

            if time.monotonic() - last_progress_at >= _PIPELINE_DRAIN_STALL_SECONDS:
                retried = False
                for task_id in tuple(sorted(pipeline._derivation_lease_by_task)):
                    retried = pipeline._retry_derivation_task(int(task_id)) or retried
                if retried:
                    last_progress_at = time.monotonic()
                    last_progress_token = pipeline.derivation_drain_progress_token()
                    continue

                # Pending identities are valid work, not a stall. They have no
                # lease yet, so the old retry-only check falsely treated a
                # saturated derivation queue as fatal (for example pending=143,
                # inflight=0). Keep draining while workers are alive.
                lease_pending, lease_inflight, _ = pipeline.derivation_leases.counts
                if (
                    int(lease_pending) > 0
                    and int(lease_inflight) == 0
                    and pipeline.derivation_workers_alive()
                ):
                    last_progress_at = time.monotonic()
                    last_progress_token = pipeline.derivation_drain_progress_token()
                    continue

                diagnostics = pipeline.diagnostics()
                raise RuntimeError(
                    "derivation drain stalled with no live progress path: "
                    f"{diagnostics}"
                )
        memory.signal_derivation_stop()
        memory.join_derivation()

        produced_transitions = (
            int(expected_transitions) if evaluation_only else int(actor_produced_steps())
        )
        causally_admitted = int(pipeline.sampled)
        expected_ingested = 0 if evaluation_only else int(expected_transitions)
        if pipeline.ingested != expected_ingested:
            raise RuntimeError(
                f"ingestion count mismatch at epoch boundary: expected={expected_ingested} ingested={pipeline.ingested}"
            )
        if not evaluation_only and not produced_transitions == causally_admitted == int(pipeline.ingested):
            raise RuntimeError(
                "epoch causal completion mismatch: "
                f"produced={produced_transitions} causally_admitted={causally_admitted} "
                f"published={pipeline.sampled} ingested={pipeline.ingested}"
            )
        actor_view_ids = {row.epoch_inference_view_id for row in results}
        if bound_epoch_view is not None and actor_view_ids != {bound_epoch_view_id}:
            raise RuntimeError(
                f"matched actor EpochInferenceView mismatch: expected={bound_epoch_view_id} actual={sorted(actor_view_ids)}"
            )
        for key, value in {
            "actor_processes": target_actor_slots,
            "peak_active_actor_processes": peak_active_actors,
            "actor_processes_total_launched": len(topology.actor_processes),
            "stage_worker_processes": int(stage_workers),
            "shard_worker_processes": int(shards),
            "ingest_worker_processes": int(ingest_workers),
            "derivation_worker_processes": int(derivation_workers),
            "actor_produced_steps": produced_transitions,
            "causally_admitted_steps": causally_admitted,
            "publication_drained_steps": int(pipeline.sampled),
            "multiprocess_transitions_published": int(pipeline.sampled),
            "coordinator_action_requests": 0,
            "policy_snapshot_generation": int(published_policy_generation),
            "policy_snapshot_refreshes": sum(int(row.policy_refreshes) for row in results),
            "actor_epoch_inference_view_ids": ",".join(sorted(actor_view_ids)),
            "actors_exited_without_done": 0,
            "coordinator_pending_ingest": 0,
            "coordinator_pending_ingest_batches": 0,
            "coordinator_pending_derivation": 0,
            "canonical_pipeline_version": 3,
            "evaluation_only": int(bool(evaluation_only)),
            "evaluation_steps": int(expected_transitions) if evaluation_only else 0,
        }.items():
            runtime.set_telemetry_gauge(key, value)
        progress()
        clean_shutdown = True
        return sorted(results, key=lambda row: row.actor_id)
    except BaseException:
        memory.terminate()
        topology.terminate()
        raise
    finally:
        if hgt_writer is not None:
            hgt_writer.close()
        shutdown_parallel_pipeline = getattr(pipeline, "shutdown_parallel_pipeline", None)
        if callable(shutdown_parallel_pipeline):
            shutdown_parallel_pipeline()
        memory.close(drain=clean_shutdown)
        topology.close(drain=clean_shutdown)
        runtime.__dict__["_tracked_shm_bytes"] = 0
