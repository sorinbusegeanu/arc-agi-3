from __future__ import annotations

from v9.runtime.canonical_store import CanonicalStore
from v9.runtime.canonical_transaction import (
    CanonicalTransactionStatus,
    CanonicalWorkBudget,
    estimate_overlay_work,
)


def test_one_oversized_primitive_has_distinct_status() -> None:
    store = CanonicalStore()
    overlay = store.begin_overlay(1)
    overlay.put("payload", "one", "x" * 200)
    estimate = estimate_overlay_work(overlay, rows=1, input_bytes=200)
    budget = CanonicalWorkBudget(max_continuation_bytes=32)
    assert budget.status(estimate) is CanonicalTransactionStatus.OVERSIZED_CANONICAL_PRIMITIVE
