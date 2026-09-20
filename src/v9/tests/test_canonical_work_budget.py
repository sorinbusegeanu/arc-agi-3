from __future__ import annotations

from v9.runtime.canonical_store import CanonicalStore
from v9.runtime.canonical_transaction import (
    CanonicalTransactionStatus,
    CanonicalWorkBudget,
    estimate_overlay_work,
)


def test_exact_overlay_measurements_drive_admission() -> None:
    store = CanonicalStore()
    overlay = store.begin_overlay(1)
    overlay.put("graph", "a", {"value": 1})
    estimate = estimate_overlay_work(overlay, rows=1, input_bytes=12, work_units=3)
    assert estimate.write_count == 1
    assert estimate.materialized_mutation_bytes == overlay.estimated_bytes
    assert CanonicalWorkBudget().status(estimate) is CanonicalTransactionStatus.READY


def test_total_transaction_ceiling_is_explicit() -> None:
    store = CanonicalStore()
    overlay = store.begin_overlay(1)
    overlay.put("graph", "a", 1)
    estimate = estimate_overlay_work(overlay, rows=2, input_bytes=1)
    budget = CanonicalWorkBudget(max_rows=1)
    assert budget.status(estimate) is CanonicalTransactionStatus.OVERSIZED_CANONICAL_TRANSACTION
