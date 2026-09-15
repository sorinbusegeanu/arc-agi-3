from __future__ import annotations

from types import SimpleNamespace

from v9.hgt.grounding_objectives import GroundingObjectiveEvidence, validate_objective_evidence
from v9.memory.identity import MemoryUid
from v9.memory.symbolic_grounding import SymbolicRelation
from v9.memory.symbolic_relations import derive_symbolic_relations, shuffled_alignment_control
from v9.research.grounding_h16 import H16RunState, evaluate_h16, load_h16_trials, run_synthetic_h16_controls, save_h16_evidence


def _grounded(low: int):
    uid = MemoryUid(1, low)
    return SimpleNamespace(uid=uid, provenance=SimpleNamespace(evidence=(uid,)))


def _symbol(low: int, identity: tuple[int, int, int, int]):
    return SimpleNamespace(m0=SimpleNamespace(symbol_identity=identity), m1g=_grounded(low), event=SimpleNamespace(identity=SimpleNamespace(causal_watermark=10 + low)))


def test_symbolic_relation_derivation_covers_phase4_runtime_producers() -> None:
    rows = (_symbol(10, (1, 2, 3, 0)), _symbol(11, (1, 2, 3, 1)))
    transition = SimpleNamespace(
        before_signature=1,
        after_signature=2,
        boundary_scope="TASK",
        task_success=True,
        task_failure=False,
        task_truncated=False,
        levels_completed=1,
        primary_valence=1,
    )
    derived = derive_symbolic_relations(
        rows,
        interaction_grounding=_grounded(20),
        previous_interaction_grounding=_grounded(19),
        transition=transition,
        causal_watermark=50,
    )
    kinds = {row.relation_kind for row in derived}
    required = {
        SymbolicRelation.SYMBOL_PRECEDES_SYMBOL,
        SymbolicRelation.SYMBOL_FOLLOWS_SYMBOL,
        SymbolicRelation.SYMBOL_RECURS_WITHIN_WINDOW,
        SymbolicRelation.SYMBOL_PRECEDES_ACTION,
        SymbolicRelation.SYMBOL_FOLLOWS_ACTION,
        SymbolicRelation.SYMBOL_PRECEDES_NORMALIZED_CHANGE,
        SymbolicRelation.SYMBOL_FOLLOWS_NORMALIZED_CHANGE,
        SymbolicRelation.SYMBOL_NEAR_BOUNDARY,
        SymbolicRelation.SYMBOL_COINCIDENT_WITH_PROGRESS,
        SymbolicRelation.SYMBOL_COINCIDENT_WITH_OUTCOME,
        SymbolicRelation.SYMBOL_INTERACTION_ALIGNMENT,
    }
    assert required <= kinds
    assert all(row.relation.causal_watermark == 50 for row in derived)
    assert all(row.relation.support >= 0.0 and row.relation.contradiction >= 0.0 for row in derived)


def test_shuffled_alignment_is_negative_control_only() -> None:
    row = _symbol(10, (1, 2, 3, 0))
    control = shuffled_alignment_control(row, _grounded(20), causal_watermark=30)
    assert control.control == "shuffled"
    assert control.relation.support == 0.0
    assert control.relation.contradiction == 1.0


def test_grounding_objective_evidence_tracks_aligned_and_shuffled_controls() -> None:
    rows = (
        GroundingObjectiveEvidence("shuffled_alignment_discrimination", 1.0, 0.9, 1, 2, "a", "aligned"),
        GroundingObjectiveEvidence("shuffled_alignment_discrimination", 0.0, 0.1, 1, 2, "s", "shuffled"),
    )
    result = validate_objective_evidence(rows)
    assert result == {"examples": 2, "positive_examples": 1, "negative_examples": 1, "control_groups": 2}


def test_h16_evidence_round_trip_preserves_matched_run_state(tmp_path) -> None:
    state = H16RunState("config", "model-v1", 4, 2, "snapshot:4")
    trials = run_synthetic_h16_controls(seeds=(1, 2), environment_config_id=9, interaction_budget=24, evaluation_id=3, run_state=state)
    report = evaluate_h16(trials)
    assert report.matched
    path = tmp_path / "h16.json"
    save_h16_evidence(path, trials, report)
    restored = load_h16_trials(path)
    assert restored == trials
    restored_report = evaluate_h16(restored)
    assert restored_report.matched
    assert restored_report.means == report.means
    assert restored_report.causal_effect == report.causal_effect


def test_public_runtime_is_completed_v978_runtime() -> None:
    import v9
    import v9.runtime
    from v9.runtime.completed_runtime import CompletedContinuousMemoryRuntime
    assert v9.ContinuousMemoryRuntime is CompletedContinuousMemoryRuntime
    assert v9.runtime.ContinuousMemoryRuntime is CompletedContinuousMemoryRuntime
