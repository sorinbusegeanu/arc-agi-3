from __future__ import annotations

import inspect
import queue
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Thread
from typing import Any

from .canonical_commit import apply_canonical_commit_batch, canonical_commit_prefix_length
from .memory_pipeline import CanonicalMutationIntent, PreparedCommitBatch
from .parallel_memory_coordinator import _adaptive_canonical_batch_size
from .shared_batch_transport import (
    SharedBatchDescriptor,
    SlabOwnership,
    TransportSlabDescriptor,
    TransportSlabPool,
    consume_shared_batch,
    decode_transport_value,
)


_INGEST_TASK_BATCH_SIZE = 2048
_MIN_PUBLICATION_INTAKE_HIGH_WATER = 4096
_MAX_PUBLICATION_INTAKE_HIGH_WATER = 16384
_RESULT_QUEUE_DRAIN_BATCHES = 64
_REDUCER_QUEUE_BATCHES = 4
_MAX_DECODE_PENDING_BATCHES = 32
_MAX_PREPARED_INTENT_ROWS = 8192
_MAX_REDUCER_INFLIGHT_EVENTS = 8192
_REDUCER_SHUTDOWN_TIMEOUT_SECONDS = 5.0


def _batch_rows(value: Any) -> int:
    if isinstance(value, SharedBatchDescriptor):
        return int(value.rows)
    return len(value.rows)


def _batch_end_sequence(value: Any) -> int:
    if isinstance(value, SharedBatchDescriptor):
        return int(value.end_sequence)
    return int(value.end_sequence)


def _prepared_row_input_bytes(value: PreparedCommitBatch) -> tuple[int, ...]:
    measured = tuple(int(row) for row in (getattr(value, "row_input_bytes", ()) or ()))
    if measured:
        if len(measured) != len(value.rows) or sum(measured) != int(value.input_bytes):
            raise ValueError("prepared commit row-byte accounting mismatch")
        return measured
    rows = len(value.rows)
    if rows <= 0:
        return ()
    base, remainder = divmod(int(getattr(value, "input_bytes", 0)), rows)
    return tuple(base + int(index < remainder) for index in range(rows))


def _split_prepared_batch(
    value: PreparedCommitBatch, count: int
) -> tuple[PreparedCommitBatch, PreparedCommitBatch | None]:
    if not 0 < int(count) <= len(value.rows):
        raise ValueError("prepared commit split count is outside the batch")
    measured = _prepared_row_input_bytes(value)
    selected_rows = tuple(value.rows[:count])
    selected_bytes = measured[:count]
    selected = PreparedCommitBatch(
        int(value.start_sequence),
        int(value.start_sequence) + int(count) - 1,
        selected_rows,
        sum(selected_bytes),
        selected_bytes,
    )
    if count == len(value.rows):
        return selected, None
    remaining_bytes = measured[count:]
    remaining = PreparedCommitBatch(
        int(value.start_sequence) + int(count),
        int(value.end_sequence),
        tuple(value.rows[count:]),
        sum(remaining_bytes),
        remaining_bytes,
    )
    return selected, remaining


def _prepared_rows_waiting(service: Any) -> int:
    return sum(
        _batch_rows(value)
        for value in service.ingest_results.values()
        if not isinstance(value, BaseException)
    )


class _CanonicalReducer:
    """Single owner of authoritative canonical runtime mutation."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.input: queue.Queue[Any] = queue.Queue(maxsize=_REDUCER_QUEUE_BATCHES)
        self.output: queue.Queue[Any] = queue.Queue()
        self.thread = Thread(target=self._run, name="v9-canonical-reducer", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            item = self.input.get()
            if item is None:
                return
            batch_id, plans, input_bytes = item
            started = time.perf_counter()
            try:
                parameters = inspect.signature(
                    apply_canonical_commit_batch
                ).parameters.values()
                if any(
                    parameter.name == "input_bytes"
                    or parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters
                ):
                    result = apply_canonical_commit_batch(
                        self.runtime, plans, input_bytes=int(input_bytes)
                    )
                else:
                    result = apply_canonical_commit_batch(self.runtime, plans)
            except BaseException as exc:
                # Canonical mutation failure is terminal for this reducer.  Do not
                # process later batches after an authoritative commit has failed.
                self.output.put(("error", batch_id, exc))
                return
            self.output.put(("ok", batch_id, result, time.perf_counter() - started, len(plans), float(getattr(result, "lock_seconds", 0.0))))

    def submit(
        self,
        batch_id: int,
        plans: tuple[CanonicalMutationIntent, ...],
        *,
        input_bytes: int,
    ) -> bool:
        try:
            self.input.put_nowait((int(batch_id), plans, int(input_bytes)))
        except queue.Full:
            return False
        return True

    def close(self, *, timeout: float = _REDUCER_SHUTDOWN_TIMEOUT_SECONDS) -> None:
        if not self.thread.is_alive():
            return
        deadline = time.monotonic() + max(0.0, float(timeout))
        while self.thread.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError("canonical reducer shutdown timed out")
            try:
                self.input.put(None, timeout=min(0.05, remaining))
                break
            except queue.Full:
                continue
        remaining = max(0.0, deadline - time.monotonic())
        self.thread.join(timeout=remaining)
        if self.thread.is_alive():
            raise RuntimeError("canonical reducer shutdown timed out")


def _store_ingest_result(service: Any, start_sequence: int, value: Any) -> None:
    sequence = int(start_sequence)
    if sequence < int(service.ingest_apply):
        raise RuntimeError(f"stale ingest result sequence: {sequence} < {service.ingest_apply}")
    if sequence in service.ingest_results:
        raise RuntimeError(f"duplicate ingest result sequence: {sequence}")
    service.ingest_results[sequence] = value


def _drain_decode_completions(service: Any) -> bool:
    progressed = False
    while True:
        try:
            start_sequence, batch, decode_ms = service._decoded_ingest_queue.get_nowait()
        except queue.Empty:
            break
        sequence = int(start_sequence)
        reservation = service._decode_rows_by_sequence.pop(sequence, None)
        if reservation is None:
            raise RuntimeError(f"decode completion without reservation: {sequence}")
        service._decode_pending -= 1
        service._decode_pending_rows -= int(reservation)
        if service._decode_pending < 0 or service._decode_pending_rows < 0:
            raise RuntimeError("decode reservation accounting underflow")
        if not isinstance(batch, BaseException):
            expected_end = sequence + int(reservation) - 1
            if (
                len(batch.rows) != int(reservation)
                or int(batch.start_sequence) != sequence
                or int(batch.end_sequence) != expected_end
            ):
                batch = RuntimeError(
                    "decoded batch reservation mismatch: "
                    f"start={sequence} reserved={reservation} "
                    f"actual_start={getattr(batch, 'start_sequence', None)} "
                    f"actual_end={getattr(batch, 'end_sequence', None)} "
                    f"actual_rows={len(getattr(batch, 'rows', ())) }"
                )
            else:
                service.ingest_result_decode_ms += float(decode_ms)
        _store_ingest_result(service, sequence, batch)
        progressed = True
    return progressed


def _submit_decode(
    service: Any,
    descriptor: SharedBatchDescriptor | TransportSlabDescriptor,
    pool: TransportSlabPool | None = None,
) -> None:
    sequence = int(descriptor.start_sequence)
    rows = int(descriptor.rows)
    if rows <= 0:
        raise RuntimeError(f"invalid decode reservation rows: {rows}")
    if sequence < int(service.ingest_apply):
        raise RuntimeError(f"stale decode sequence: {sequence} < {service.ingest_apply}")
    if sequence in service._decode_rows_by_sequence or sequence in service.ingest_results:
        raise RuntimeError(f"duplicate decode sequence: {sequence}")
    service._decode_pending += 1
    service._decode_pending_rows += rows
    service._decode_rows_by_sequence[sequence] = rows

    if pool is not None:
        pool.transfer_to_coordinator(descriptor)

    def decode() -> tuple[Any, float]:
        if pool is None:
            return consume_shared_batch(descriptor)
        started = time.perf_counter()
        try:
            return decode_transport_value(pool.read(descriptor)), 1000.0 * (time.perf_counter() - started)
        finally:
            pool.release(descriptor, owner=SlabOwnership.COORDINATOR_OWNED)

    try:
        future: Future[Any] = service._decode_pool.submit(decode)
    except BaseException:
        if pool is not None:
            pool.release(descriptor, owner=SlabOwnership.COORDINATOR_OWNED)
        service._decode_pending -= 1
        service._decode_pending_rows -= rows
        service._decode_rows_by_sequence.pop(sequence, None)
        raise

    def done(completed: Future[Any]) -> None:
        try:
            batch, decode_ms = completed.result()
        except BaseException as exc:
            service._decoded_ingest_queue.put((int(descriptor.start_sequence), exc, -1.0))
            return
        service._decoded_ingest_queue.put((int(descriptor.start_sequence), batch, float(decode_ms)))

    future.add_done_callback(done)


def _finish_reducer_results(service: Any) -> bool:
    progressed = False
    while True:
        try:
            item = service._canonical_reducer.output.get_nowait()
        except queue.Empty:
            break
        kind = item[0]
        if kind == "error":
            failed_batch_id = int(item[1])
            service._reducer_inflight.pop(failed_batch_id, None)
            # The reducer terminates on a canonical failure, so queued batches will
            # never complete and must not keep shutdown waiting forever.
            service._reducer_inflight.clear()
            service.runtime._canonical_commit_inflight = False
            raise item[2]
        _, batch_id, result, elapsed, count, lock_seconds = item
        expected_batch_id = min(service._reducer_inflight) if service._reducer_inflight else None
        if expected_batch_id is None or int(batch_id) != int(expected_batch_id):
            raise RuntimeError(
                f"canonical reducer completion out of order: expected={expected_batch_id} actual={batch_id}"
            )
        reservation = service._reducer_inflight.pop(int(batch_id))
        service.canonical_apply_seconds += float(elapsed)
        service.canonical_apply_events += int(count)
        service.last_canonical_ingest_batch = int(count)
        service.ingested += int(count)
        service.release_ingest_input_bytes(int(reservation[3]) if len(reservation) > 3 else 0)
        service._canonical_commit_batches += 1
        service._reducer_apply_ms += 1000.0 * float(elapsed)
        service._reducer_lock_ms += 1000.0 * float(lock_seconds)
        for candidate in result.derivation_candidates:
            service._consider_candidate(candidate)
        progressed = True
    service.runtime._canonical_commit_inflight = bool(service._reducer_inflight)
    if progressed:
        backlog = max(
            len(service.pending_ingest),
            int(getattr(service, "pending_ingest_batch_rows", 0)),
            _prepared_rows_waiting(service),
            max(0, int(service.sampled) - int(service.ingested)),
        )
        service.canonical_batch_size = _adaptive_canonical_batch_size(service.canonical_batch_size, backlog)
    return progressed


def _submit_canonical_commit(service: Any, *, max_rows: int | None = None) -> bool:
    configured_max_rows = int(
        getattr(
            getattr(service.runtime, "config", None),
            "canonical_transaction_max_rows",
            1024,
        )
    )
    max_rows = configured_max_rows if max_rows is None else min(int(max_rows), configured_max_rows)
    transport_batches: list[PreparedCommitBatch] = []
    row_count = 0
    input_bytes = 0
    while service.ingest_apply in service.ingest_results:
        lookup_sequence = int(service.ingest_apply)
        value = service.ingest_results[lookup_sequence]
        if isinstance(value, BaseException):
            service.ingest_results.pop(lookup_sequence)
            raise value
        if int(value.start_sequence) != lookup_sequence:
            raise RuntimeError(
                f"compiled intent sequence mismatch: expected start={lookup_sequence} actual={value.start_sequence}"
            )
        rows = _batch_rows(value)
        split_remaining: PreparedCommitBatch | None = None
        remaining_rows = int(max_rows) - row_count
        if rows > remaining_rows:
            if remaining_rows <= 0:
                break
            selected, split_remaining = _split_prepared_batch(value, remaining_rows)
            value = selected
            rows = len(selected.rows)
        if transport_batches and row_count + rows > service.canonical_batch_size:
            break
        service.ingest_results.pop(lookup_sequence)
        if split_remaining is not None:
            service.ingest_results[int(split_remaining.start_sequence)] = split_remaining
        transport_batches.append(value)
        row_count += rows
        input_bytes += int(getattr(value, "input_bytes", 0))
        service.ingest_apply = _batch_end_sequence(value) + 1
        if row_count >= service.canonical_batch_size:
            break
    if not transport_batches:
        return False

    plans: list[CanonicalMutationIntent] = []
    plan_input_bytes: list[int] = []
    expected = None
    for batch in transport_batches:
        measured_input_bytes = _prepared_row_input_bytes(batch)
        if expected is not None and int(batch.start_sequence) != expected:
            raise RuntimeError(f"compiled intent sequence gap: expected={expected} actual={batch.start_sequence}")
        row_sequence = int(batch.start_sequence)
        for row, row_bytes in zip(batch.rows, measured_input_bytes, strict=True):
            if not isinstance(row, CanonicalMutationIntent):
                raise TypeError(f"ingest workers must emit CanonicalMutationIntent, got {type(row).__name__}")
            if int(row.sequence) != row_sequence:
                raise RuntimeError(
                    f"compiled intent row sequence mismatch: expected={row_sequence} actual={row.sequence}"
                )
            plans.append(row)
            plan_input_bytes.append(int(row_bytes))
            row_sequence += 1
        if row_sequence - 1 != int(batch.end_sequence):
            raise RuntimeError(
                f"compiled intent batch end mismatch: expected={row_sequence - 1} actual={batch.end_sequence}"
            )
        expected = int(batch.end_sequence) + 1

    admitted = canonical_commit_prefix_length(
        service.runtime,
        plans,
        row_input_bytes=plan_input_bytes,
    )
    if admitted < len(plans):
        remaining = PreparedCommitBatch(
            int(plans[admitted].sequence),
            int(plans[-1].sequence),
            tuple(plans[admitted:]),
            sum(plan_input_bytes[admitted:]),
            tuple(plan_input_bytes[admitted:]),
        )
        if int(remaining.start_sequence) in service.ingest_results:
            raise RuntimeError(
                f"canonical budget split collides at sequence {remaining.start_sequence}"
            )
        service.ingest_results[int(remaining.start_sequence)] = remaining
        plans = plans[:admitted]
        plan_input_bytes = plan_input_bytes[:admitted]
        input_bytes = sum(plan_input_bytes)
        service.ingest_apply = int(remaining.start_sequence)
        transport_batches = [
            PreparedCommitBatch(
                int(plans[0].sequence),
                int(plans[-1].sequence),
                tuple(plans),
                int(input_bytes),
                tuple(plan_input_bytes),
            )
        ]

    batch_id = service._reducer_batch_id
    if not service._canonical_reducer.submit(
        batch_id, tuple(plans), input_bytes=input_bytes
    ):
        # Restore batches so ordered submission can be retried without loss.
        for batch in reversed(transport_batches):
            service.ingest_results[int(batch.start_sequence)] = batch
        service.ingest_apply = int(transport_batches[0].start_sequence)
        return False
    service._reducer_inflight[int(batch_id)] = (
        int(row_count), int(plans[0].sequence), int(plans[-1].sequence), int(input_bytes)
    )
    service._reducer_batch_id += 1
    service.runtime._canonical_commit_inflight = True
    return True


def install_publication_throughput(pipeline_cls: type) -> None:
    """Parallel compile/decode with one thin, persistent ordered canonical reducer."""
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
            max(_MIN_PUBLICATION_INTAKE_HIGH_WATER, self.ipc_batch_size * ingest_workers),
        )
        self._decoded_ingest_queue: queue.Queue[Any] = queue.Queue()
        self._held_ingest_items: deque[Any] = deque()
        self._decode_pending = 0
        self._decode_pending_rows = 0
        self._decode_rows_by_sequence: dict[int, int] = {}
        self._decode_pool = ThreadPoolExecutor(max_workers=max(2, ingest_workers), thread_name_prefix="v9-intent-decode")
        self._canonical_reducer = _CanonicalReducer(runtime)
        self._reducer_inflight: dict[int, tuple[int, int, int]] = {}
        self._reducer_batch_id = 1
        self._canonical_commit_batches = 0
        self._intent_compile_ms = 0.0
        self._intent_compile_rows = 0
        self._reducer_apply_ms = 0.0
        self._reducer_lock_ms = 0.0
        self._backpressure_decode_hits = 0
        self._backpressure_prepared_hits = 0
        self._backpressure_reducer_hits = 0
        self._reducer_shutdown_timeout_seconds = _REDUCER_SHUTDOWN_TIMEOUT_SECONDS
        self.runtime._canonical_commit_inflight = False

    def drain_ingest_results(self: Any, *, block: bool = False, timeout: float = 0.0) -> bool:
        progressed = _drain_decode_completions(self)
        started = time.perf_counter()
        first = True
        for _ in range(_RESULT_QUEUE_DRAIN_BATCHES):
            if self._held_ingest_items:
                item = self._held_ingest_items.popleft()
            else:
                try:
                    item = self.memory.ingest_result_queue.get(timeout=timeout) if block and first else self.memory.ingest_result_queue.get_nowait()
                except queue.Empty:
                    break
            first = False
            if item[0] == "worker_error":
                raise RuntimeError(f"{item[1]} worker task {item[2]} failed: {item[3]}")
            if item[0] == "ingest_batch_shm":
                descriptor = item[3]
                if self._decode_pending >= _MAX_DECODE_PENDING_BATCHES:
                    self._held_ingest_items.appendleft(item)
                    self._backpressure_decode_hits += 1
                    break
                reserved_rows = _prepared_rows_waiting(self) + int(self._decode_pending_rows)
                if reserved_rows + int(descriptor.rows) > _MAX_PREPARED_INTENT_ROWS:
                    self._held_ingest_items.appendleft(item)
                    self._backpressure_prepared_hits += 1
                    break
                _submit_decode(self, descriptor)
                self.ingest_result_batches += 1
                self.ingest_result_rows += int(descriptor.rows)
                self.ingest_result_bytes += int(descriptor.size)
                self.ingest_result_encode_ms += float(descriptor.encode_ms)
                if len(item) > 4:
                    self._intent_compile_ms += float(item[4])
                    self._intent_compile_rows += int(descriptor.rows)
                progressed = True
                continue
            if item[0] == "ingest_batch_slab":
                descriptor = TransportSlabDescriptor.unpack(item[4])
                if self._decode_pending >= _MAX_DECODE_PENDING_BATCHES:
                    self._held_ingest_items.appendleft(item)
                    self._backpressure_decode_hits += 1
                    break
                reserved_rows = _prepared_rows_waiting(self) + int(self._decode_pending_rows)
                if reserved_rows + int(descriptor.rows) > _MAX_PREPARED_INTENT_ROWS:
                    self._held_ingest_items.appendleft(item)
                    self._backpressure_prepared_hits += 1
                    break
                _submit_decode(self, descriptor, self.memory.ingest_result_pool)
                self.ingest_result_batches += 1
                self.ingest_result_rows += int(descriptor.rows)
                self.ingest_result_bytes += int(descriptor.length)
                self.ingest_result_encode_ms += float(item[5])
                self._intent_compile_ms += float(item[3])
                self._intent_compile_rows += int(descriptor.rows)
                progressed = True
                continue
            if item[0] == "ingest_batch":
                batch = item[3]
                if _prepared_rows_waiting(self) + len(batch.rows) > _MAX_PREPARED_INTENT_ROWS:
                    self._held_ingest_items.appendleft(item)
                    self._backpressure_prepared_hits += 1
                    break
                _store_ingest_result(self, int(item[1]), batch)
                self.ingest_result_batches += 1
                self.ingest_result_rows += len(batch.rows)
                progressed = True
        self.ingest_result_drain_seconds += time.perf_counter() - started
        return _drain_decode_completions(self) or progressed

    def apply_ingest_ready(self: Any) -> bool:
        progressed = _finish_reducer_results(self)
        progressed = _drain_decode_completions(self) or progressed
        # Keep a bounded number of batches queued so the reducer never waits on the coordinator.
        while len(self._reducer_inflight) < _REDUCER_QUEUE_BATCHES:
            inflight_events = sum(row[0] for row in self._reducer_inflight.values())
            remaining_events = _MAX_REDUCER_INFLIGHT_EVENTS - inflight_events
            if remaining_events <= 0:
                self._backpressure_reducer_hits += 1
                break
            if not _submit_canonical_commit(self, max_rows=remaining_events):
                if self.ingest_apply in self.ingest_results:
                    self._backpressure_reducer_hits += 1
                break
            progressed = True
        return progressed

    def service(self: Any) -> bool:
        progressed = _finish_reducer_results(self)
        progressed = _drain_decode_completions(self) or progressed
        progressed = self.pump_ingest_tasks() or progressed
        progressed = self.pump_derivation_tasks() or progressed
        progressed = self.drain_ingest_results() or progressed
        progressed = self.drain_derivation_results() or progressed
        progressed = self.apply_derivation_ready() or progressed
        progressed = self.apply_ingest_ready() or progressed
        progressed = self.pump_ingest_tasks() or progressed
        progressed = self.pump_derivation_tasks() or progressed
        return progressed

    def block_for_result(self: Any, timeout: float = 0.05) -> bool:
        if _finish_reducer_results(self) or _drain_decode_completions(self):
            return True
        if self._reducer_inflight:
            try:
                item = self._canonical_reducer.output.get(timeout=max(0.0, float(timeout)))
            except queue.Empty:
                return False
            kind = item[0]
            if kind == "error":
                failed_batch_id = int(item[1])
                self._reducer_inflight.pop(failed_batch_id, None)
                self._reducer_inflight.clear()
                self.runtime._canonical_commit_inflight = False
                raise item[2]
            _, batch_id, result, elapsed, count, lock_seconds = item
            expected_batch_id = min(self._reducer_inflight) if self._reducer_inflight else None
            if expected_batch_id is None or int(batch_id) != int(expected_batch_id):
                raise RuntimeError(
                    f"canonical reducer completion out of order: expected={expected_batch_id} actual={batch_id}"
                )
            reservation = self._reducer_inflight.pop(int(batch_id))
            self.canonical_apply_seconds += float(elapsed)
            self.canonical_apply_events += int(count)
            self.last_canonical_ingest_batch = int(count)
            self.ingested += int(count)
            self.release_ingest_input_bytes(int(reservation[3]) if len(reservation) > 3 else 0)
            self._canonical_commit_batches += 1
            self._reducer_apply_ms += 1000.0 * float(elapsed)
            self._reducer_lock_ms += 1000.0 * float(lock_seconds)
            for candidate in result.derivation_candidates:
                self._consider_candidate(candidate)
            self.runtime._canonical_commit_inflight = bool(self._reducer_inflight)
            return True
        if self._decode_pending:
            time.sleep(min(max(0.0, float(timeout)), 0.005))
            return _drain_decode_completions(self)
        if self.drain_ingest_results(block=True, timeout=float(timeout)):
            self.apply_ingest_ready()
            return True
        return self.drain_derivation_results(block=True, timeout=float(timeout))

    def shutdown_parallel_pipeline(self: Any) -> None:
        self._decode_pool.shutdown(wait=True, cancel_futures=False)
        _drain_decode_completions(self)
        deadline = time.monotonic() + float(self._reducer_shutdown_timeout_seconds)
        while self._reducer_inflight:
            if _finish_reducer_results(self):
                continue
            if time.monotonic() >= deadline:
                raise RuntimeError("canonical reducer shutdown timed out")
            time.sleep(0.001)
        remaining = max(0.0, deadline - time.monotonic())
        self._canonical_reducer.close(timeout=remaining)
        self.runtime._canonical_commit_inflight = False

    def diagnostics(self: Any) -> dict[str, float | int]:
        result = dict(original_diagnostics(self))
        result.update({
            "canonical_commit_inflight": int(bool(self._reducer_inflight)),
            "canonical_commit_batches": int(self._canonical_commit_batches),
            "canonical_commit_pending_events": sum(row[0] for row in self._reducer_inflight.values()),
            "canonical_reducer_queue_depth": int(self._canonical_reducer.input.qsize()),
            "intent_decode_pending": int(self._decode_pending),
            "intent_decode_pending_rows": int(self._decode_pending_rows),
            "held_ingest_result_items": int(len(self._held_ingest_items)),
            "publication_intake_high_water": int(self.ingest_local_high_water),
            "ingest_task_batch_size": int(self.ipc_batch_size),
            "prepared_ingest_batches_waiting": int(len(self.ingest_results)),
            "prepared_ingest_rows_waiting": int(_prepared_rows_waiting(self)),
            "coordinator_ingest_decode_ms": 0.0,
            "intent_compile_ms": float(self._intent_compile_ms),
            "intent_compile_ms_per_transition": float(self._intent_compile_ms / max(1, self._intent_compile_rows)),
            "intent_bytes_per_transition": float(self.ingest_result_bytes / max(1, self.ingest_result_rows)),
            "reducer_apply_ms": float(self._reducer_apply_ms),
            "reducer_apply_ms_per_transition": float(self._reducer_apply_ms / max(1, self.canonical_apply_events)),
            "reducer_lock_ms": float(self._reducer_lock_ms),
            "reducer_lock_fraction": float(self._reducer_lock_ms / max(0.001, self._reducer_apply_ms)),
            "reducer_inflight_events": int(sum(row[0] for row in self._reducer_inflight.values())),
            "backpressure_decode_hits": int(self._backpressure_decode_hits),
            "backpressure_prepared_hits": int(self._backpressure_prepared_hits),
            "backpressure_reducer_hits": int(self._backpressure_reducer_hits),
            "max_decode_pending_batches": int(_MAX_DECODE_PENDING_BATCHES),
            "max_prepared_intent_rows": int(_MAX_PREPARED_INTENT_ROWS),
            "max_reducer_inflight_events": int(_MAX_REDUCER_INFLIGHT_EVENTS),
        })
        return result

    pipeline_cls.__init__ = init
    pipeline_cls.drain_ingest_results = drain_ingest_results
    pipeline_cls.apply_ingest_ready = apply_ingest_ready
    pipeline_cls.service = service
    pipeline_cls.block_for_result = block_for_result
    pipeline_cls.shutdown_parallel_pipeline = shutdown_parallel_pipeline
    pipeline_cls.diagnostics = diagnostics
    pipeline_cls._publication_throughput_installed = True
