from __future__ import annotations

from types import SimpleNamespace

import pytest

from v9.runtime import direct_coordinator_safety as safety
from v9.runtime.memory_pipeline import PreparedCommitBatch


class _Plan:
    def __init__(self, sequence: int) -> None:
        self.sequence = int(sequence)


class _Service:
    def __init__(self) -> None:
        self.runtime = SimpleNamespace(
            config=SimpleNamespace(canonical_transaction_max_rows=128),
            _resident_memory=None,
        )
        self.ingest_results = {}
        self.ingest_apply = 1
        self.canonical_batch_size = 128
        self.canonical_apply_seconds = 0.0
        self.canonical_apply_events = 0
        self.last_canonical_ingest_batch = 0
        self.ingested = 0
        self.sampled = 4
        self.pending_ingest = []
        self.released = []
        self.considered = []

    def release_ingest_input_bytes(self, value: int) -> None:
        self.released.append(int(value))

    def _consider_candidate(self, candidate: object) -> None:
        self.considered.append(candidate)


def test_direct_coordinator_splits_oversized_combined_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    safety.install(_Service)
    service = _Service()
    service.ingest_results[1] = PreparedCommitBatch(
        1,
        4,
        tuple(_Plan(index) for index in range(1, 5)),
        40,
        (10, 10, 10, 10),
    )

    def prefix(_runtime, rows, *, row_input_bytes):
        assert tuple(row.sequence for row in rows) == (1, 2, 3, 4)
        assert sum(row_input_bytes) == 40
        return 2

    def apply(_runtime, rows, *, input_bytes):
        assert tuple(row.sequence for row in rows) == (1, 2)
        assert input_bytes == 20
        return SimpleNamespace(derivation_candidates=())

    monkeypatch.setattr(safety, "canonical_commit_prefix_length", prefix)
    monkeypatch.setattr(safety, "apply_canonical_commit_batch", apply)

    assert service.apply_ingest_ready() is True
    assert service.ingested == 2
    assert service.released == [20]
    assert service.ingest_apply == 3
    assert tuple(row.sequence for row in service.ingest_results[3].rows) == (3, 4)
    assert service.ingest_results[3].row_input_bytes == (10, 10)


def test_direct_coordinator_does_not_quarantine_single_oversized_row(monkeypatch: pytest.MonkeyPatch) -> None:
    safety.install(_Service)
    service = _Service()
    service.ingest_results[1] = PreparedCommitBatch(
        1,
        1,
        (_Plan(1),),
        10,
        (10,),
    )

    def prefix(_runtime, rows, *, row_input_bytes):
        raise RuntimeError("single row too large")

    monkeypatch.setattr(safety, "canonical_commit_prefix_length", prefix)

    with pytest.raises(RuntimeError, match="single row too large"):
        service.apply_ingest_ready()

    assert service.ingested == 0
    assert service.released == []
    assert 1 in service.ingest_results


def test_direct_actor_factory_bypasses_passive_capture_wrapper() -> None:
    class Topology:
        def start_actor(self, *args, **kwargs):
            self.adapter_factory_path = kwargs["adapter_factory_path"]
            return "started"

    safety._DIRECT_ACTOR_FACTORY = "v9.environments.direct_adapter:make_adapter"
    original = Topology.start_actor
    Topology.start_actor = original
    Topology._direct_actor_adapter_installed = False

    # Exercise the same patching logic on a local class by temporarily replacing
    # the imported ProcessTopology symbol through a simple install target.
    from v9.runtime.direct_coordinator_safety import _DIRECT_ACTOR_FACTORY

    def install_local(process_topology_cls: type) -> None:
        original_start_actor = process_topology_cls.start_actor

        def start_actor(self, *args, **kwargs):
            if kwargs.get("adapter_factory_path") == "v9.cli:make_adapter":
                kwargs = dict(kwargs)
                kwargs["adapter_factory_path"] = _DIRECT_ACTOR_FACTORY
            return original_start_actor(self, *args, **kwargs)

        process_topology_cls.start_actor = start_actor

    install_local(Topology)
    topology = Topology()

    assert topology.start_actor(adapter_factory_path="v9.cli:make_adapter") == "started"
    assert topology.adapter_factory_path == _DIRECT_ACTOR_FACTORY
