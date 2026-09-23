from __future__ import annotations

from v9.runtime.canonical_store import CanonicalStore
from v9.runtime.developmental_cut import (
    DevelopmentalCandidate,
    DevelopmentalCut,
    DevelopmentalCutSession,
    DevelopmentalMutationGate,
    DevelopmentalWorkStatus,
)
from v9.runtime.scientific_modes import ScientificVisibilityMode


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


def test_matched_cut_session_pins_base_and_canonicalizes_operation_order() -> None:
    store = CanonicalStore()
    cut = DevelopmentalCut()
    gate = DevelopmentalMutationGate(ScientificVisibilityMode.MATCHED_REASONING)

    def execute(order: tuple[str, ...]):
        session = DevelopmentalCutSession(
            cut=cut,
            gate=gate,
            evidence_cut_id="evidence-1",
            pinned_handle=store.pin(),
        )
        called = []
        for operator in order:
            session.run(
                operator,
                lambda operator=operator: (
                    gate.assert_publication_allowed(),
                    called.append(operator),
                ),
                stable_key=operator,
            )
        result = session.finish()
        assert called == list(order)
        assert not store._pin_counts
        return result

    first = execute(("lifecycle", "m1_maturation"))
    second = execute(("m1_maturation", "lifecycle"))
    assert first == second
    assert tuple(row.operator for row in first.results) == tuple(
        operator.name for operator in cut.pipeline.operators
    )


def test_failed_matched_cut_operation_releases_pinned_handle() -> None:
    store = CanonicalStore()
    session = DevelopmentalCutSession(
        cut=DevelopmentalCut(),
        gate=DevelopmentalMutationGate(ScientificVisibilityMode.MATCHED_REASONING),
        evidence_cut_id="evidence-1",
        pinned_handle=store.pin(),
    )

    try:
        session.run("m1_maturation", lambda: (_ for _ in ()).throw(ValueError("broken")))
    except ValueError:
        pass
    else:
        raise AssertionError("expected deterministic developmental operation failure")
    assert not store._pin_counts
