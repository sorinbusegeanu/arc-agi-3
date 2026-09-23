from __future__ import annotations

from v9.runtime.canonical_store import CanonicalStore
from v9.runtime.developmental_cut import DevelopmentalCandidate, DevelopmentalCut, DevelopmentalWorkStatus


def test_worker_completion_order_does_not_change_cut_manifest() -> None:
    store = CanonicalStore()
    rows = tuple(DevelopmentalCandidate(key, (priority,), 1) for key, priority in (("z", 2), ("a", 1), ("m", 2)))
    cut = DevelopmentalCut()
    first = cut.run(evidence_cut_id="e", base_handle=store.current_handle, candidates={"m1_maturation": rows}, apply=lambda *_: DevelopmentalWorkStatus.NO_CHANGE)
    second = cut.run(evidence_cut_id="e", base_handle=store.current_handle, candidates={"m1_maturation": tuple(reversed(rows))}, apply=lambda *_: DevelopmentalWorkStatus.NO_CHANGE)
    assert first == second
