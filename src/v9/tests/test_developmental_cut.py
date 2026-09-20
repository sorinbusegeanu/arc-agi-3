from __future__ import annotations

from v9.runtime.canonical_store import CanonicalStore
from v9.runtime.developmental_cut import (
    DevelopmentalCandidate,
    DevelopmentalCut,
    DevelopmentalWorkStatus,
)


def test_developmental_cut_selects_stably_with_fixed_budget() -> None:
    store = CanonicalStore()
    cut = DevelopmentalCut()
    candidates = {
        "m1_maturation": (
            DevelopmentalCandidate("b", (1,), 1),
            DevelopmentalCandidate("a", (1,), 1),
        )
    }
    result = cut.run(
        evidence_cut_id="evidence-1",
        base_handle=store.current_handle,
        candidates=candidates,
        apply=lambda _operator, _candidate: DevelopmentalWorkStatus.APPLIED,
    )
    assert tuple(row.stable_key for row in result.results) == ("a", "b")
    assert len(result.manifest_id) == 64
