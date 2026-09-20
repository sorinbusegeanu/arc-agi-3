from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect
import queue
from threading import RLock
from types import SimpleNamespace

import pytest

from v9.hgt.epoch_dataset import EpochTransitionDataset
from v9.runtime import ContinuousMemoryRuntime
from v9.runtime.memory_pipeline import IngestionBatchTask
from v9.runtime.parallel_memory_coordinator import (
    MemoryPipelineService,
    _adaptive_publication_batch_size,
    _drain_publication_batch,
    run_parallel_memory_jobs,
)


@dataclass(frozen=True)
class Transition:
    actor_id: int
    producer_sequence: int
    symbols: tuple[int, ...] = ()
    payload: str = "x"


class FakeRuntime:
    def __init__(self, *, watermark: int = 0, symbolic: bool = True) -> None:
        self.watermark = watermark
        self._producer_sequences: dict[int, int] = {}
        self._lock = RLock()
        self._canonical_commit_inflight = False
        self.config = SimpleNamespace(scientific=SimpleNamespace(
            symbolic_grounding_enabled=symbolic,
            symbol_budget_per_window=8,
            max_symbol_facts_per_window=8,
            symbol_payload_bytes=1024,
            max_cross_modal_facts_per_macro_event=8,
            symbol_deduplication_policy="stable_first",
            symbol_window_time_span=32,
            symbol_codec_name="test",
            symbol_codec_version=1,
        ))

    def reserve_producer_sequence(self, producer_id: int, proposed_sequence: int) -> int:
        raise AssertionError("batch publication must not call scalar sequence reservation")

    def reserve_producer_sequences_batch(self, requests):
        result = []
        with self._lock:
            for producer_id, proposed_sequence in requests:
                producer_id = int(producer_id)
                sequence = max(int(proposed_sequence), self._producer_sequences.get(producer_id, 0) + 1)
                self._producer_sequences[producer_id] = sequence
                result.append(sequence)
        return tuple(result)


class FakeMemory:
    def __init__(self, capacity: int = 8) -> None:
        self.ingest_queue = queue.Queue(maxsize=capacity)
        self.derivation_queue = queue.Queue()
        self.ingest_result_queue = queue.Queue()
        self.derivation_result_queue = queue.Queue()
        self.ingest_workers = 1


def _service(capacity: int = 8, *, symbolic: bool = True) -> MemoryPipelineService:
    service = MemoryPipelineService(FakeRuntime(symbolic=symbolic), FakeMemory(capacity), ingest_queue_capacity=capacity)
    service.ipc_batch_size = 2048
    return service


def test_runtime_batch_sequence_reservation_matches_scalar_semantics() -> None:
    scalar = ContinuousMemoryRuntime.__new__(ContinuousMemoryRuntime)
    scalar._lock = RLock()
    scalar._producer_sequences = {1: 4}
    batched = ContinuousMemoryRuntime.__new__(ContinuousMemoryRuntime)
    batched._lock = RLock()
    batched._producer_sequences = {1: 4}
    requests = ((1, 2), (2, 8), (1, 12), (2, 3), (1, 12))
    expected = tuple(
        super(ContinuousMemoryRuntime, scalar).reserve_producer_sequence(producer, proposed)
        for producer, proposed in requests
    )
    actual = batched.reserve_producer_sequences_batch(requests)
    assert actual == expected
    assert batched._producer_sequences == scalar._producer_sequences


def test_batch_dispatch_allocates_sequences_watermarks_and_symbols_in_order() -> None:
    service = _service()
    try:
        transitions = (Transition(1, 1, (1, 2)), Transition(1, 1, (3,)), Transition(2, 7, ()))
        assert service.dispatch_transitions_batch(transitions) == 3
        batch = service.memory.ingest_queue.get_nowait()
        assert isinstance(batch, IngestionBatchTask)
        assert [task.sequence for task in batch.tasks] == [1, 2, 3]
        assert [task.causal_watermark for task in batch.tasks] == [1, 4, 6]
        assert [task.transition.producer_sequence for task in batch.tasks] == [1, 2, 7]
        assert service.watermark_cursor == 6
        assert service.sampled == 3
    finally:
        service.shutdown_parallel_pipeline()


def test_batch_dispatch_strips_symbols_when_grounding_disabled() -> None:
    service = _service(symbolic=False)
    try:
        service.dispatch_transitions_batch((Transition(1, 1, (1, 2, 3)),))
        batch = service.memory.ingest_queue.get_nowait()
        assert batch.tasks[0].transition.symbols == ()
        assert batch.tasks[0].causal_watermark == 1
        assert service.watermark_cursor == 1
    finally:
        service.shutdown_parallel_pipeline()


def test_queue_full_retains_whole_batches_and_preserves_order() -> None:
    service = _service(capacity=1)
    try:
        service.memory.ingest_queue.put_nowait("occupied")
        rows = tuple(Transition(1, index + 1) for index in range(4))
        service.dispatch_transitions_batch(rows)
        assert len(service.pending_ingest_batches) == 1
        assert service.pending_ingest_batch_rows == 4
        assert service.memory.ingest_queue.get_nowait() == "occupied"
        assert service.pump_ingest_tasks()
        batch = service.memory.ingest_queue.get_nowait()
        assert [task.sequence for task in batch.tasks] == [1, 2, 3, 4]
        assert service.pending_ingest_batch_rows == 0
    finally:
        service.shutdown_parallel_pipeline()


def test_pending_batch_buffer_is_hard_bounded() -> None:
    service = _service(capacity=1)
    try:
        service.ingest_local_high_water = 4
        service.memory.ingest_queue.put_nowait("occupied")
        service.dispatch_transitions_batch(tuple(Transition(1, index + 1) for index in range(4)))
        assert service.pending_ingest_batch_rows == 4
        try:
            service.dispatch_transitions_batch((Transition(1, 5),))
        except RuntimeError as exc:
            assert "high-water" in str(exc)
        else:
            raise AssertionError("expected bounded high-water rejection")
    finally:
        service.shutdown_parallel_pipeline()


def test_pending_batch_buffer_enforces_exact_carried_byte_high_water() -> None:
    service = _service(capacity=1)
    try:
        service.ingest_local_byte_high_water = 10
        service.memory.ingest_queue.put_nowait("occupied")
        service.dispatch_transitions_batch(
            (Transition(1, 1), Transition(1, 2)),
            carried_bytes=10,
        )
        assert service.pending_ingest_batch_rows == 2
        assert service.pending_ingest_batch_bytes == 10
        assert service.outstanding_ingest_bytes == 10

        sampled = service.sampled
        ingest_sequence = service.ingest_sequence
        with pytest.raises(RuntimeError, match="byte high-water"):
            service.dispatch_transitions_batch((Transition(1, 3),), carried_bytes=1)
        assert service.sampled == sampled
        assert service.ingest_sequence == ingest_sequence
        assert service.outstanding_ingest_bytes == 10

        assert service.memory.ingest_queue.get_nowait() == "occupied"
        assert service.pump_ingest_tasks()
        queued = service.memory.ingest_queue.get_nowait()
        assert queued.input_bytes == 10
        assert service.pending_ingest_batch_bytes == 0
    finally:
        service.shutdown_parallel_pipeline()


def test_ingest_worker_accepts_batch_state_created_before_byte_field(monkeypatch) -> None:
    from v9.runtime import memory_pipeline

    task = SimpleNamespace(sequence=1)
    legacy = object.__new__(IngestionBatchTask)
    object.__setattr__(legacy, "start_sequence", 1)
    object.__setattr__(legacy, "end_sequence", 1)
    object.__setattr__(legacy, "tasks", (task,))
    assert not hasattr(legacy, "task_input_bytes")

    monkeypatch.setattr(memory_pipeline, "prepare_ingestion", lambda value: value)
    monkeypatch.setattr(memory_pipeline, "build_commit_plan", lambda value: value)
    prepared = memory_pipeline.prepare_commit_batch(legacy)

    assert prepared.input_bytes == 0
    assert prepared.rows == (task,)

    captured = []
    monkeypatch.setattr(
        memory_pipeline,
        "_publish_compiled_result",
        lambda _queue, _pool, value, **_kwargs: captured.append(value),
    )
    task_queue = queue.Queue()
    task_queue.put(legacy)
    from v9.runtime.multiprocess import WorkerStop

    task_queue.put(WorkerStop())
    memory_pipeline.ingest_batch_worker_main(
        task_queue,
        queue.Queue(),
        result_pool=object(),
    )
    assert len(captured) == 1
    assert captured[0].input_bytes == 0


def test_publication_queue_batch_drain_exceeds_old_512_limit() -> None:
    publication_queue = queue.Queue()
    for index in range(1500):
        publication_queue.put(("transition", 0, index, Transition(1, index + 1)))
    transitions, markers = _drain_publication_batch(publication_queue, 2048)
    assert len(transitions) == 1500
    assert not markers
    assert _adaptive_publication_batch_size(30_000) == 8192


def test_publication_queue_preserves_shard_done_marker() -> None:
    publication_queue = queue.Queue()
    publication_queue.put(("transition", 0, 0, Transition(1, 1)))
    publication_queue.put(("shard_done", 3, None, None))
    transitions, markers = _drain_publication_batch(publication_queue, 8)
    assert len(transitions) == 1
    assert markers == (3,)
    assert publication_queue.get_nowait()[0] == "shard_done"


def test_hgt_append_batch_is_byte_identical_to_repeated_append(tmp_path: Path) -> None:
    rows = tuple(Transition(1, index + 1, (index,)) for index in range(5))
    scalar_path = tmp_path / "scalar.jsonl"
    batch_path = tmp_path / "batch.jsonl"
    scalar = EpochTransitionDataset(scalar_path, epoch=1, branch="test", model_version="m")
    batched = EpochTransitionDataset(batch_path, epoch=1, branch="test", model_version="m")
    for row in rows:
        scalar.append(row)
    assert batched.append_batch(rows) == len(rows)
    scalar.close()
    batched.close()
    assert scalar_path.read_bytes() == batch_path.read_bytes()
    assert scalar.count == batched.count
    assert scalar.bytes_written == batched.bytes_written


def test_coordinator_source_uses_batch_publication_only() -> None:
    wrapped = run_parallel_memory_jobs
    original = inspect.getclosurevars(wrapped).nonlocals.get("original", wrapped)
    source = inspect.getsource(original)
    assert "dispatch_published_transitions" in source
    assert "pipeline.dispatch_transition(" not in source
    assert "hgt_dataset.append(" not in source
