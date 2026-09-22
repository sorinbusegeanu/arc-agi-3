from __future__ import annotations

from v9.memory.symbolic_grounding import GroundingEvidence, GroundingMaturity, SymbolicRelation, modality_support
from v9.research.symbolic_grounding_experiment import H16Condition, condition_payload
from v9.modalities.symbols import DeterministicSymbolCodec, SYMBOL_SCHEMA_VERSION, SymbolOccurrence
from v9.memory.identity import EpisodeId, EventUid


def test_symbol_occurrence_contract_preserves_identity_order_time_and_provenance() -> None:
    codec = DeterministicSymbolCodec("opaque")
    row = codec.encode_stream(("x",), stream_name="instruction")[0]
    event = EventUid.from_producer(3, 4)
    occurrence = SymbolOccurrence(event, row.symbol_id, row.position.value, row.stream_id, row.vocabulary_id, "identity:1", 7, 2, 1, 9, EpisodeId(5), event)
    assert occurrence.symbol_id == row.symbol_id
    assert occurrence.position == 0
    assert occurrence.causal_watermark == 7
    assert occurrence.provenance_id == event
    assert SYMBOL_SCHEMA_VERSION >= 3


def test_symbolic_relation_contract_covers_design_m1n_relations() -> None:
    expected = {
        "SYMBOL_PRECEDES_SYMBOL", "SYMBOL_FOLLOWS_SYMBOL", "SYMBOL_RECURS_WITHIN_WINDOW",
        "SYMBOL_PRECEDES_ACTION", "SYMBOL_FOLLOWS_ACTION",
        "SYMBOL_PRECEDES_NORMALIZED_CHANGE", "SYMBOL_FOLLOWS_NORMALIZED_CHANGE",
        "SYMBOL_NEAR_BOUNDARY", "SYMBOL_COINCIDENT_WITH_PROGRESS", "SYMBOL_COINCIDENT_WITH_OUTCOME",
        "CROSS_MODAL_CORRESPONDENCE", "SYMBOL_INTERACTION_ALIGNMENT", "SYMBOL_TO_INTERACTION_PREDICTION",
        "INTERACTION_TO_SYMBOL_GENERALIZATION", "CROSS_MODAL_HELDOUT_TRANSFER", "CROSS_MODAL_COMPOSITION",
    }
    assert expected == {getattr(SymbolicRelation, name) for name in expected}


def test_grounding_requires_predictive_heldout_evidence_for_behavior_authority() -> None:
    cooccurrence = GroundingEvidence((), (), (), 3.0, 0.0)
    assert cooccurrence.maturity == GroundingMaturity.G0_INTERACTION
    assert not cooccurrence.behavior_eligible


def test_modality_support_keeps_shared_family_evidence_decomposable() -> None:
    assert modality_support(("WORLD", "SYMBOL", "CROSS_MODAL", "WORLD")) == {"interaction": 2, "symbol": 1, "cross_modal": 1}


def test_h16_conditions_preserve_statistics_and_shuffle_alignment() -> None:
    world = (1, 2, 3, 4)
    symbols = ("a", "b", "c", "d")
    assert condition_payload(H16Condition.C0_INTERACTION_ONLY, world, symbols, seed=7) == (world, ())
    assert condition_payload(H16Condition.C1_SYMBOLS_ONLY, world, symbols, seed=7) == ((), symbols)
    assert condition_payload(H16Condition.C2_ALIGNED, world, symbols, seed=7) == (world, symbols)
    c3_world, shuffled = condition_payload(H16Condition.C3_SHUFFLED, world, symbols, seed=7)
    assert c3_world == world
    assert sorted(shuffled) == sorted(symbols)
    assert shuffled != symbols


def test_hgt_uses_canonical_symbol_not_legacy_tuple_authority() -> None:
    from v9.hgt.canonical_symbol_graph import HGT_V978_MODEL_SCHEMA_VERSION, _without_legacy_symbol_tuples
    from v9.hgt.training import SEMANTIC_NODE_TYPES
    assert "SYMBOL" in SEMANTIC_NODE_TYPES
    assert "TEXT" not in SEMANTIC_NODE_TYPES
    assert HGT_V978_MODEL_SCHEMA_VERSION == 8
    payload = {"symbol_identity": [1, 2, 3, 0], "semantic_before": [[7, 99, 1, 2, 1.0], [5, 1, 2, 3, 1.0]]}
    filtered = _without_legacy_symbol_tuples(payload)
    assert "symbol_identity" not in filtered
    assert filtered["semantic_before"] == [[5, 1, 2, 3, 1.0]]


def test_m2_retains_modality_support_decomposition() -> None:
    import inspect
    from v9.memory.m2_family import M2TransformationFamily
    assert "modality_support" in inspect.signature(M2TransformationFamily).parameters
    assert "support_decomposition" in inspect.signature(M2TransformationFamily).parameters


def test_symbolic_commit_plan_carries_occurrence_provenance() -> None:
    import inspect
    from v9.runtime.memory_pipeline import PreparedIngestion
    from v9.runtime.memory_pipeline import CommitPlan
    assert "symbol_occurrences" in inspect.signature(PreparedIngestion).parameters
    assert "symbol_occurrences" in inspect.signature(CommitPlan).parameters


def test_grounding_graph_relations_are_explicit() -> None:
    from v9.memory.relations import RelationType
    required = {"OBSERVED_IN", "PRECEDES", "FOLLOWS", "TEMPORALLY_ALIGNED_WITH", "STRUCTURALLY_CORRESPONDS_TO", "PARTICIPATES_IN", "SUPPORTS", "CONTRADICTS", "GROUNDS", "TRANSFER_VALIDATES"}
    assert required <= {row.name for row in RelationType}


def test_cognition_grounding_g3_becomes_behavior_eligible_and_round_trips() -> None:
    from v9.cognition.grounding import GroundingEvidence as RuntimeGroundingEvidence
    from v9.cognition.grounding import GroundingMaturity as RuntimeGroundingMaturity
    from v9.cognition.grounding import GroundingRegistry
    registry = GroundingRegistry()
    state = registry.observe(RuntimeGroundingEvidence(11, 22, 3, 4, 5, 100, recurrent_symbol=True, cross_modal_association=True, prospective_prediction=True, validation_trial_id="trial-1", support=2.0))
    assert state.maturity == RuntimeGroundingMaturity.G3
    assert state.behavior_eligible
    assert registry.authority(11, 22, 3, 4, 5) > 0.0
    restored = GroundingRegistry.from_state_dict(registry.state_dict())
    row = restored.states[(11, 22, 3, 4, 5)]
    assert row.behavior_eligible
    assert row.validation_trial_ids == ("trial-1",)
    assert row.last_causal_watermark == 100


def test_grounding_g4_and_g5_have_distinct_evidence_requirements() -> None:
    from v9.cognition.grounding import GroundingEvidence as RuntimeGroundingEvidence
    from v9.cognition.grounding import GroundingMaturity as RuntimeGroundingMaturity
    from v9.cognition.grounding import GroundingRegistry
    registry = GroundingRegistry()
    g3 = registry.observe(RuntimeGroundingEvidence(1, 2, 3, 0, 0, 1, heldout_transfer=True, support=1.0))
    assert g3.maturity == RuntimeGroundingMaturity.G3
    g4 = registry.observe(RuntimeGroundingEvidence(1, 2, 3, 0, 0, 2, novel_composition=True, support=1.0))
    assert g4.maturity == RuntimeGroundingMaturity.G4
    g5 = registry.observe(RuntimeGroundingEvidence(1, 2, 3, 0, 0, 3, symbol_mediated_learning=True, support=1.0))
    assert g5.maturity == RuntimeGroundingMaturity.G5


def test_cognition_grounding_negative_evidence_suspends_behavior_authority() -> None:
    from v9.cognition.grounding import GroundingEvidence as RuntimeGroundingEvidence
    from v9.cognition.grounding import GroundingRegistry
    registry = GroundingRegistry()
    registry.observe(RuntimeGroundingEvidence(1, 2, 3, 0, 0, 1, recurrent_symbol=True, cross_modal_association=True, prospective_prediction=True, support=1.0))
    state = registry.observe(RuntimeGroundingEvidence(1, 2, 3, 0, 0, 2, causal_intervention=True, positive=False, support=2.0, contradiction=2.0, validation_trial_id="negative"))
    assert state.suspended
    assert not state.behavior_eligible
    assert registry.authority(1, 2, 3) == 0.0


def test_h16_expanded_metrics_are_matched_and_reported() -> None:
    from v9.research.grounding_h16 import GroundingCondition, H16Metrics, H16Trial, evaluate_h16
    trials = []
    for condition in GroundingCondition:
        aligned = condition is GroundingCondition.C2_ALIGNED
        trials.append(H16Trial(condition, 7, 1, 20, 1, H16Metrics(interaction_prediction=0.9 if aligned else 0.2, action_success=0.9 if aligned else 0.2, symbol_conditioned_transfer=0.9 if aligned else 0.1, world_to_symbol_generalization=0.8 if aligned else 0.1, composition_success=0.8 if aligned else 0.0, persistence_without_symbols=0.4, grounding_calibration=0.9), 0.3 if aligned else 0.0, f"trial:{condition.value}"))
    report = evaluate_h16(tuple(trials))
    assert report.matched
    assert report.causal_effect == 0.3
    assert report.metric_means["C2"]["composition_success"] == 0.8
