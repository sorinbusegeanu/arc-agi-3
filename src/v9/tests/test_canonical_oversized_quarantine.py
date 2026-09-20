from __future__ import annotations

from types import SimpleNamespace

import pytest

from v9.benchmarks.ingestion_drain import _plans
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig
from v9.runtime.canonical_commit import (
    apply_canonical_commit_batch,
    estimate_canonical_commit_batch,
)
from v9.runtime.canonical_transaction import (
    CanonicalTransactionQuarantined,
    CanonicalTransactionStatus,
)
from v9.runtime.memory_pipeline import PreparedCommitBatch
from v9.runtime.publication_throughput import (
    _split_prepared_batch,
    _submit_canonical_commit,
)


def test_oversized_transaction_is_quarantined_before_mutation(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(
            tmp_path,
            restore=False,
            enable_snapshots=False,
            canonical_transaction_max_rows=1,
        )
    )
    generation = runtime.graph.generation
    with pytest.raises(CanonicalTransactionQuarantined) as raised:
        apply_canonical_commit_batch(runtime, _plans(2), input_bytes=2)
    assert raised.value.status is CanonicalTransactionStatus.OVERSIZED_CANONICAL_TRANSACTION
    assert raised.value.sequences == (1, 2)
    assert runtime.graph.generation == generation
    assert runtime.graph.nodes == {}


def test_oversized_primitive_is_quarantined_before_mutation(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(
            tmp_path,
            restore=False,
            enable_snapshots=False,
            canonical_continuation_max_bytes=1,
        )
    )
    with pytest.raises(CanonicalTransactionQuarantined) as raised:
        apply_canonical_commit_batch(runtime, _plans(1), input_bytes=1)
    assert raised.value.status is CanonicalTransactionStatus.OVERSIZED_CANONICAL_PRIMITIVE
    assert runtime.graph.nodes == {}


def test_transaction_split_preserves_exact_row_and_byte_measurements() -> None:
    plans = _plans(3)
    selected, remaining = _split_prepared_batch(
        PreparedCommitBatch(1, 3, plans, 60, (10, 20, 30)), 2
    )
    assert selected.start_sequence == 1
    assert selected.end_sequence == 2
    assert selected.input_bytes == 30
    assert selected.row_input_bytes == (10, 20)
    assert remaining is not None
    assert remaining.start_sequence == 3
    assert remaining.end_sequence == 3
    assert remaining.input_bytes == 30
    assert remaining.row_input_bytes == (30,)


def test_publication_admits_safe_prefix_before_expanded_work_exceeds_budget() -> None:
    plans = _plans(3)
    one = estimate_canonical_commit_batch(plans[:1], input_bytes=10)
    two = estimate_canonical_commit_batch(plans[:2], input_bytes=30)
    assert two.materialized_mutation_bytes > one.materialized_mutation_bytes

    config = SimpleNamespace(
        canonical_transaction_max_rows=1024,
        canonical_transaction_max_input_bytes=64 * 1024 * 1024,
        canonical_transaction_max_mutation_bytes=two.materialized_mutation_bytes - 1,
        canonical_continuation_max_bytes=16 * 1024 * 1024,
        canonical_transaction_max_writes=65_536,
        canonical_transaction_max_work_units=1_000_000,
    )
    runtime = SimpleNamespace(config=config, _canonical_commit_inflight=False)
    submitted = []

    class Reducer:
        def submit(self, batch_id, rows, *, input_bytes):
            submitted.append((batch_id, rows, input_bytes))
            return True

    service = SimpleNamespace(
        runtime=runtime,
        ingest_apply=1,
        ingest_results={1: PreparedCommitBatch(1, 3, plans, 60, (10, 20, 30))},
        canonical_batch_size=1024,
        _canonical_reducer=Reducer(),
        _reducer_batch_id=1,
        _reducer_inflight={},
    )

    assert _submit_canonical_commit(service)
    assert len(submitted) == 1
    assert tuple(row.sequence for row in submitted[0][1]) == (1,)
    assert submitted[0][2] == 10
    assert service._reducer_inflight[1][0] == 1
    assert service.ingest_apply == 2
    remainder = service.ingest_results[2]
    assert tuple(row.sequence for row in remainder.rows) == (2, 3)
    assert remainder.input_bytes == 50
    assert remainder.row_input_bytes == (20, 30)
