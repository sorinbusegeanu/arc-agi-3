from __future__ import annotations

import importlib.util
import time
import queue

import pytest
from threading import Event
from types import SimpleNamespace
from pathlib import Path

from v9.mutation.versions import ObjectRef, VersionTable
from v9.runtime.actor_policy_cache import install_actor_policy_cache
from v9.runtime.canonical_commit import CanonicalCommitResult
from v9.runtime.memory_pipeline import CommitPlan, PreparedCommitBatch
from v9.runtime.parallel_memory_coordinator import MemoryPipelineService
import v9.runtime.publication_throughput as publication_throughput


def test_canonical_commit_submission_does_not_block_pipeline(monkeypatch) -> None:
    entered = Event()
    release = Event()

    def slow_commit(_runtime, _plans):
        entered.set()
        assert release.wait(timeout=2.0)
        return CanonicalCommitResult((), ())

    monkeypatch.setattr(publication_throughput, "apply_canonical_commit_batch", slow_commit)
    runtime = SimpleNamespace(watermark=0)
    service = MemoryPipelineService(runtime, SimpleNamespace(), ingest_queue_capacity=128)
    plan = CommitPlan(1, None, None, None, None, (), None, (), None, (), None, "", None)
    service.ingest_results[1] = PreparedCommitBatch(1, 1, (plan,))

    started = time.perf_counter()
    assert service.apply_ingest_ready()
    elapsed = time.perf_counter() - started

    assert elapsed < 0.5
    assert entered.wait(timeout=1.0)
    assert service.ingested == 0
    assert service.ingest_apply == 2
    assert runtime._canonical_commit_inflight is True
    assert service.ingest_local_high_water >= 4096

    release.set()
    deadline = time.monotonic() + 2.0
    while service.ingested < 1 and time.monotonic() < deadline:
        service.apply_ingest_ready()
        time.sleep(0.005)

    assert service.ingested == 1
    assert runtime._canonical_commit_inflight is False
    diagnostics = service.diagnostics()
    assert diagnostics["canonical_commit_batches"] == 1
    assert diagnostics["canonical_commit_inflight"] == 0


def test_primary_runtime_has_no_v2_compatibility_modules() -> None:
    for module_name in (
        "v9.runtime.memory_pipeline_v2",
        "v9.runtime.memory_worker_topology_v2",
        "v9.runtime.pipeline_service_v2",
        "v9.runtime.parallel_memory_coordinator_v2",
    ):
        assert importlib.util.find_spec(module_name) is None


def test_actor_policy_cache_serves_last_completed_snapshot_during_commit() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.graph = SimpleNamespace(generation=0)
            self._actor_policy_generation = 0
            self.unified_telemetry = SimpleNamespace(model_version="")
            self.calls = 0

        def set_telemetry_gauge(self, _key, _value) -> None:
            return None

        def actor_policy_snapshot(self):
            self.calls += 1
            return object()

    install_actor_policy_cache(Runtime)
    runtime = Runtime()
    first = runtime.actor_policy_snapshot()
    runtime._canonical_commit_inflight = True
    runtime.graph.generation = 1
    runtime.__dict__.pop("_actor_policy_snapshot_cache", None)

    second = runtime.actor_policy_snapshot()

    assert second is first
    assert runtime.calls == 1


def test_version_table_supports_batched_deltas() -> None:
    table = VersionTable()
    ref = ObjectRef("node", 11, 22)
    table.bump(ref)
    table.bump_many({ref: 999})
    assert table.get(ref) == 1000


def test_ingest_worker_source_compiles_canonical_intents() -> None:
    import inspect
    from v9.runtime import memory_pipeline
    source = inspect.getsource(memory_pipeline.ingest_batch_worker_main)
    assert "build_commit_plan(prepare_ingestion(task))" in source


def test_publication_throughput_has_persistent_reducer_and_no_plan_build() -> None:
    import inspect
    from v9.runtime import publication_throughput
    source = inspect.getsource(publication_throughput)
    assert "class _CanonicalReducer" in source
    assert "build_commit_plan" not in source
    assert "ThreadPoolExecutor" in source


def test_runtime_integrity_does_not_replace_ingest_worker() -> None:
    import inspect
    from v9.runtime import runtime_integrity
    source = inspect.getsource(runtime_integrity._install_shared_memory_cleanup)
    assert "ingest_batch_worker_main" not in source
    assert "derivation_batch_worker_main" not in source


def test_option_a_backpressure_limits_are_hard_bounded() -> None:
    assert publication_throughput._MAX_DECODE_PENDING_BATCHES > 0
    assert publication_throughput._MAX_PREPARED_INTENT_ROWS > 0
    assert publication_throughput._MAX_REDUCER_INFLIGHT_EVENTS > 0


def test_option_a_diagnostics_expose_compile_reducer_and_backpressure_metrics() -> None:
    runtime = SimpleNamespace(watermark=0)
    service = MemoryPipelineService(runtime, SimpleNamespace(), ingest_queue_capacity=128)
    diagnostics = service.diagnostics()
    required = {
        "intent_compile_ms_per_transition", "intent_bytes_per_transition",
        "canonical_reducer_queue_depth", "reducer_apply_ms",
        "reducer_apply_ms_per_transition", "reducer_lock_ms",
        "reducer_lock_fraction", "prepared_ingest_rows_waiting",
        "backpressure_decode_hits", "backpressure_prepared_hits",
        "backpressure_reducer_hits",
    }
    assert required <= set(diagnostics)


def test_coordinator_source_does_not_compile_commit_plans() -> None:
    import inspect
    source = inspect.getsource(publication_throughput)
    assert "build_commit_plan(" not in source
    assert "ThreadPoolExecutor" in source
    assert "_CanonicalReducer" in source


def test_reducer_is_single_canonical_writer() -> None:
    import inspect
    reducer_source = inspect.getsource(publication_throughput._CanonicalReducer)
    submit_source = inspect.getsource(publication_throughput._submit_canonical_commit)
    assert "apply_canonical_commit_batch" in reducer_source
    assert "apply_canonical_commit_batch" not in submit_source


def test_intent_worker_reports_compile_time() -> None:
    import inspect
    from v9.runtime import memory_pipeline
    source = inspect.getsource(memory_pipeline.ingest_batch_worker_main)
    assert "compile_ms" in source
    assert "build_commit_plan(prepare_ingestion(task))" in source



def test_reducer_failure_clears_inflight_and_shutdown_does_not_hang(monkeypatch) -> None:
    def failing_commit(_runtime, _plans):
        raise RuntimeError("canonical boom")

    monkeypatch.setattr(publication_throughput, "apply_canonical_commit_batch", failing_commit)
    runtime = SimpleNamespace(watermark=0)
    service = MemoryPipelineService(runtime, SimpleNamespace(), ingest_queue_capacity=128)
    plan = CommitPlan(1, None, None, None, None, (), None, (), None, (), None, "", None)
    service.ingest_results[1] = PreparedCommitBatch(1, 1, (plan,))
    assert service.apply_ingest_ready()

    deadline = time.monotonic() + 2.0
    while service._canonical_reducer.output.empty() and time.monotonic() < deadline:
        time.sleep(0.005)
    with pytest.raises(RuntimeError, match="canonical boom"):
        service.apply_ingest_ready()
    assert service._reducer_inflight == {}
    assert runtime._canonical_commit_inflight is False

    started = time.perf_counter()
    service.shutdown_parallel_pipeline()
    assert time.perf_counter() - started < 1.0


def test_reducer_event_limit_cannot_be_overshot(monkeypatch) -> None:
    entered = Event()
    release = Event()

    def slow_commit(_runtime, plans):
        entered.set()
        assert release.wait(timeout=2.0)
        return CanonicalCommitResult(tuple(() for _ in plans), ())

    monkeypatch.setattr(publication_throughput, "apply_canonical_commit_batch", slow_commit)
    monkeypatch.setattr(publication_throughput, "_MAX_REDUCER_INFLIGHT_EVENTS", 1)
    runtime = SimpleNamespace(watermark=0)
    service = MemoryPipelineService(runtime, SimpleNamespace(), ingest_queue_capacity=128)
    plan1 = CommitPlan(1, None, None, None, None, (), None, (), None, (), None, "", None)
    plan2 = CommitPlan(2, None, None, None, None, (), None, (), None, (), None, "", None)
    service.ingest_results[1] = PreparedCommitBatch(1, 1, (plan1,))
    service.ingest_results[2] = PreparedCommitBatch(2, 2, (plan2,))

    assert service.apply_ingest_ready()
    assert entered.wait(timeout=1.0)
    assert sum(row[0] for row in service._reducer_inflight.values()) <= 1
    assert service.ingest_apply == 2
    assert 2 in service.ingest_results
    release.set()
    service.shutdown_parallel_pipeline()


def test_prepared_intent_limit_holds_result_instead_of_overshooting(monkeypatch) -> None:
    monkeypatch.setattr(publication_throughput, "_MAX_PREPARED_INTENT_ROWS", 1)
    memory = SimpleNamespace(ingest_result_queue=queue.Queue())
    runtime = SimpleNamespace(watermark=0)
    service = MemoryPipelineService(runtime, memory, ingest_queue_capacity=128)
    plan1 = CommitPlan(1, None, None, None, None, (), None, (), None, (), None, "", None)
    plan2 = CommitPlan(2, None, None, None, None, (), None, (), None, (), None, "", None)
    memory.ingest_result_queue.put(("ingest_batch", 1, 1, PreparedCommitBatch(1, 1, (plan1,))))
    memory.ingest_result_queue.put(("ingest_batch", 2, 2, PreparedCommitBatch(2, 2, (plan2,))))

    assert service.drain_ingest_results()
    assert publication_throughput._prepared_rows_waiting(service) <= 1
    assert len(service._held_ingest_items) == 1
    service.shutdown_parallel_pipeline()


def test_compiled_batch_start_and_row_sequences_are_validated() -> None:
    runtime = SimpleNamespace(watermark=0)
    service = MemoryPipelineService(runtime, SimpleNamespace(), ingest_queue_capacity=128)
    wrong_start = CommitPlan(2, None, None, None, None, (), None, (), None, (), None, "", None)
    service.ingest_results[1] = PreparedCommitBatch(2, 2, (wrong_start,))
    with pytest.raises(RuntimeError, match="sequence mismatch"):
        service.apply_ingest_ready()
    service.shutdown_parallel_pipeline()


def test_final_shard_drain_uses_normal_transition_handler_and_backpressure() -> None:
    source = (Path(__file__).parents[1] / "runtime" / "parallel_memory_coordinator.py").read_text()
    assert source.count("dispatch_published_transition(item[3])") >= 2
    assert "final shard transition drain stalled under ingestion backpressure" in source
    assert source.count("hgt_dataset.append") == 1


def test_lock_telemetry_measures_hold_time_after_acquisition() -> None:
    source = (Path(__file__).parents[1] / "runtime" / "canonical_commit.py").read_text()
    assert "with runtime._lock:\n        lock_acquired = time.perf_counter()" in source
    assert "time.perf_counter() - lock_acquired" in source
