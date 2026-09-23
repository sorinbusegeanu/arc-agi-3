from __future__ import annotations

from itertools import permutations

from v9.runtime.derivation_merge import DerivationAggregate, merge_derivation_aggregates


def test_completion_permutations_produce_identical_aggregate() -> None:
    rows = tuple(
        DerivationAggregate(1, frozenset((key,)), (("support", value),), value, key)
        for key, value in (("c", 3), ("a", 1), ("b", 2))
    )
    results = set()
    for order in permutations(rows):
        current = order[0]
        for row in order[1:]:
            current = merge_derivation_aggregates(current, row)
        results.add(current)
    assert len(results) == 1
