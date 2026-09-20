from __future__ import annotations

import pytest

from v9.benchmarks.ingestion_drain import _plans
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig
from v9.runtime.canonical_commit import apply_canonical_commit_batch
from v9.runtime.canonical_transaction import (
    CanonicalTransactionQuarantined,
    CanonicalTransactionStatus,
)
from v9.runtime.memory_pipeline import PreparedCommitBatch
from v9.runtime.publication_throughput import _split_prepared_batch


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
