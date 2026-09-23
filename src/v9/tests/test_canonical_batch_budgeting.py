from __future__ import annotations

from types import SimpleNamespace

import pytest

from v9.runtime import canonical_batch_budgeting as budgeting
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


def test_apply_ingest_ready_splits_oversized_combined_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    budgeting.install(_Service)
    service = _Service()
    service.ingest_results[1] = PreparedCommitBatch(
        1,
        4,
        tuple(_Plan(index) for index in range(1, 5)),
        40,
        (10, 10, 10, 10),
    )

    def prefix(_runtime, rows, *, row_input_bytes):
        assert tuple(row.sequence for row in rows) in {
            (1, 2, 3, 4),
            (1, 2),
        }
        assert sum(row_input_bytes) in {40, 20}
        return min(2, len(tuple(rows)))

    def apply(_runtime, rows, *, input_bytes):
        assert tuple(row.sequence for row in rows) == (1, 2)
        assert input_bytes == 20
        return SimpleNamespace(derivation_candidates=())

    monkeypatch.setattr(budgeting, "canonical_commit_prefix_length", prefix)
    monkeypatch.setattr(budgeting, "apply_canonical_commit_batch", apply)

    assert service.apply_ingest_ready() is True
    assert service.ingested == 2
    assert service.released == [20]
    assert service.ingest_apply == 3
    assert tuple(row.sequence for row in service.ingest_results[3].rows) == (3, 4)
    assert service.ingest_results[3].row_input_bytes == (10, 10)


def test_apply_ingest_ready_keeps_crash_for_single_oversized_row(monkeypatch: pytest.MonkeyPatch) -> None:
    budgeting.install(_Service)
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

    monkeypatch.setattr(budgeting, "canonical_commit_prefix_length", prefix)

    with pytest.raises(RuntimeError, match="single row too large"):
        service.apply_ingest_ready()

    assert service.ingested == 0
    assert service.released == []
    assert 1 in service.ingest_results
