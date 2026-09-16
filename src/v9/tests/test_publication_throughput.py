from __future__ import annotations

import time
from threading import Event
from types import SimpleNamespace

from v9.mutation.versions import ObjectRef, VersionTable
from v9.runtime.actor_policy_cache import install_actor_policy_cache
from v9.runtime.canonical_commit import CanonicalCommitResult
from v9.runtime.memory_pipeline_v2 import PreparedCommitBatch
from v9.runtime.pipeline_service_v2 import MemoryPipelineServiceV2
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
    service = MemoryPipelineServiceV2(runtime, SimpleNamespace(), ingest_queue_capacity=128)
    service.ingest_results[1] = PreparedCommitBatch(1, 1, (object(),))

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
