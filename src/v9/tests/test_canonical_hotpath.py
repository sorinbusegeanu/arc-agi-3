from __future__ import annotations

from types import SimpleNamespace

import v9  # noqa: F401 - installs the authoritative runtime wrappers
from v9.memory.identity import MemoryUid
from v9.runtime import canonical_commit
from v9.runtime import canonical_commit_derivation
from v9.runtime import canonical_hotpath
from v9.runtime import publication_throughput
from v9.runtime import residency


def _relation(*, uid: MemoryUid, signature: int, evidence: tuple[MemoryUid, ...]):
    return SimpleNamespace(
        uid=uid,
        structural_signature=int(signature),
        observable_relation="TEST",
        channel=SimpleNamespace(value="WORLD"),
        provenance=SimpleNamespace(parents=evidence, evidence=evidence),
    )


def test_dirty_normalized_support_updates_coalesce_inside_one_canonical_batch() -> None:
    evidence_a = MemoryUid(1, 1)
    evidence_b = MemoryUid(1, 2)
    relation_a = _relation(uid=MemoryUid(9, 9), signature=17, evidence=(evidence_a,))
    relation_b = _relation(uid=MemoryUid(9, 9), signature=17, evidence=(evidence_b,))

    class Runtime:
        canonical_store = object()

        def __init__(self) -> None:
            self._m1n_dirty = {17}
            self._m1n_occurrences = {17: [relation_a]}
            self._watermark = 10
            self._canonical_dirty_coalesce = {}
            self.telemetry: dict[str, int] = {}
            self.support = 1

        def signature_support(self, signature: int, fallback: int = 0) -> int:
            assert signature == 17
            return int(self.support or fallback)

    runtime = Runtime()
    first_rows: list[object] = []
    assert canonical_commit._append_dirty_normalized(runtime, relation_a, first_rows) == 17
    assert len(first_rows) == 1

    runtime.support = 9
    runtime._watermark = 11
    runtime._m1n_occurrences[17].append(relation_b)
    second_rows: list[object] = []
    assert canonical_commit._append_dirty_normalized(runtime, relation_b, second_rows) == 17
    assert second_rows == []

    row = first_rows[0]
    assert row[1]["support"] == 9
    assert row[1]["parents"] == [[1, 1], [1, 2]]
    assert row[2] == [evidence_a, evidence_b]
    assert row[0].created_watermark == 11
    assert runtime.telemetry["dirty_m1n_support_updates_published"] == 1
    assert runtime.telemetry["dirty_m1n_support_updates_coalesced"] == 1


def test_family_support_uses_aggregate_signature_support_not_reservoir_length() -> None:
    rows = (
        SimpleNamespace(structural_signature=11),
        SimpleNamespace(structural_signature=11),
        SimpleNamespace(structural_signature=22),
    )

    class Runtime:
        def signature_support(self, signature: int) -> int:
            return {11: 100, 22: 7}[int(signature)]

    assert canonical_hotpath._aggregate_family_support(Runtime(), rows) == 107


def test_idle_compaction_yields_to_canonical_backlog() -> None:
    class Manager:
        def __init__(self) -> None:
            self.requests = 0
            self.services = 0

        def request_compaction(self, *, force: bool = False) -> bool:
            self.requests += 1
            return True

        def service_prepared_compaction(self, *, max_batches: int = 1) -> int:
            self.services += 1
            return 23

    class Runtime:
        def __init__(self) -> None:
            self._resident_memory = Manager()
            self.gauges: dict[str, int] = {}

        def set_telemetry_gauge(self, key: str, value: int) -> None:
            self.gauges[key] = value

    runtime = Runtime()
    service = SimpleNamespace(
        runtime=runtime,
        sampled=10_000,
        ingested=1_000,
        canonical_batch_size=1024,
        _reducer_inflight={},
    )
    assert canonical_hotpath._service_prepared_compaction_if_idle(service) == 0
    assert runtime._resident_memory.services == 0

    service.ingested = 9_500
    assert canonical_hotpath._service_prepared_compaction_if_idle(service) == 23
    assert runtime._resident_memory.services == 1
    assert runtime.gauges["idle_compaction_last_deleted"] == 23


def test_epoch_boundary_compaction_is_bounded_and_reports_remaining_backlog() -> None:
    class Manager:
        def __init__(self) -> None:
            self.remaining = 30

        def request_compaction(self, *, force: bool = False) -> bool:
            return True

        def backlog(self) -> int:
            return self.remaining

        def service_prepared_compaction(self, *, max_batches: int = 1) -> int:
            return 0

        def compact_once(self, *, force: bool = False) -> int:
            if self.remaining <= 0:
                return 0
            deleted = min(10, self.remaining)
            self.remaining -= deleted
            return deleted

    class Runtime:
        def __init__(self) -> None:
            self._resident_memory = Manager()
            self.gauges: dict[str, float | int] = {}

        def set_telemetry_gauge(self, key: str, value: float | int) -> None:
            self.gauges[key] = value

    runtime = Runtime()
    service = SimpleNamespace(runtime=runtime)
    assert canonical_hotpath._catch_up_residency_at_epoch_boundary(service) == 30
    assert runtime.gauges["epoch_compaction_start_backlog"] == 30
    assert runtime.gauges["epoch_compaction_deleted"] == 30
    assert runtime.gauges["epoch_compaction_remaining_backlog"] == 0


def test_authoritative_install_wires_hotpath_into_reducer_and_derivation() -> None:
    assert canonical_commit.apply_canonical_commit_batch is canonical_hotpath._apply_canonical_commit_coalesced
    assert publication_throughput.apply_canonical_commit_batch is canonical_hotpath._apply_canonical_commit_coalesced
    assert canonical_commit_derivation.derivation_candidates is canonical_hotpath._derivation_candidates_with_aggregate_support
    assert canonical_commit.derivation_candidates is canonical_hotpath._derivation_candidates_with_aggregate_support
    assert residency._PREPARED_COMPACTION_BATCH_LIMIT >= 16
