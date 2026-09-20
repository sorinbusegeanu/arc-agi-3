from __future__ import annotations

import pytest

from v9.runtime.derivation_merge import DerivationAggregate, merge_derivation_aggregates


def _aggregate(evidence: str, support: int, maturity: int, representative: str) -> DerivationAggregate:
    return DerivationAggregate(1, frozenset((evidence,)), (("support", support),), maturity, representative)


def test_derivation_merge_is_commutative_associative_and_idempotent() -> None:
    a = _aggregate("a", 1, 1, "z")
    b = _aggregate("b", 2, 3, "y")
    c = _aggregate("c", 4, 2, "x")
    assert merge_derivation_aggregates(a, b) == merge_derivation_aggregates(b, a)
    assert merge_derivation_aggregates(merge_derivation_aggregates(a, b), c) == merge_derivation_aggregates(a, merge_derivation_aggregates(b, c))
    assert merge_derivation_aggregates(a, a) == a
    merged = merge_derivation_aggregates(a, b)
    assert merged.evidence_ids == frozenset(("a", "b"))
    assert merged.sufficient_statistics == (("support", 3),)
    assert merged.maturity == 3
    assert merged.representative == "y"


def test_conflicting_overlap_fails_explicitly() -> None:
    left = _aggregate("a", 1, 1, "a")
    right = _aggregate("a", 2, 1, "a")
    with pytest.raises(ValueError, match="overlap"):
        merge_derivation_aggregates(left, right)
