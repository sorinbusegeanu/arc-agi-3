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
    assert SYMBOL_SCHEMA_VERSION >= 2


def test_symbolic_relation_contract_covers_design_m1n_relations() -> None:
    expected = {
        "SYMBOL_PRECEDES_SYMBOL", "SYMBOL_FOLLOWS_SYMBOL", "SYMBOL_RECURS_WITHIN_WINDOW",
        "SYMBOL_PRECEDES_ACTION", "SYMBOL_FOLLOWS_ACTION",
        "SYMBOL_PRECEDES_NORMALIZED_CHANGE", "SYMBOL_FOLLOWS_NORMALIZED_CHANGE",
        "SYMBOL_NEAR_BOUNDARY", "SYMBOL_COINCIDENT_WITH_PROGRESS", "SYMBOL_COINCIDENT_WITH_OUTCOME",
        "CROSS_MODAL_CORRESPONDENCE",
    }
    assert expected == {getattr(SymbolicRelation, name) for name in expected}


def test_grounding_requires_predictive_heldout_evidence_for_behavior_authority() -> None:
    cooccurrence = GroundingEvidence((), (), (), 3.0, 0.0)
    assert cooccurrence.maturity == GroundingMaturity.G0_INTERACTION
    assert not cooccurrence.behavior_eligible


def test_modality_support_keeps_shared_family_evidence_decomposable() -> None:
    assert modality_support(("WORLD", "SYMBOL", "CROSS_MODAL", "WORLD")) == {
        "interaction": 2, "symbol": 1, "cross_modal": 1,
    }


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


def test_hgt_uses_symbol_not_text_node_type() -> None:
    from v9.hgt.training import SEMANTIC_NODE_TYPES, _semantic_node_type
    assert "SYMBOL" in SEMANTIC_NODE_TYPES
    assert "TEXT" not in SEMANTIC_NODE_TYPES
    assert _semantic_node_type(7) == "SYMBOL"

def test_m2_retains_modality_support_decomposition() -> None:
    import inspect
    from v9.memory.m2_family import M2TransformationFamily
    assert "modality_support" in inspect.signature(M2TransformationFamily).parameters


def test_symbolic_commit_plan_carries_occurrence_provenance() -> None:
    import inspect
    from v9.runtime.memory_pipeline import PreparedIngestion
    from v9.runtime.memory_pipeline_v2 import CommitPlan
    assert "symbol_occurrences" in inspect.signature(PreparedIngestion).parameters
    assert "symbol_occurrences" in inspect.signature(CommitPlan).parameters


def test_grounding_graph_relations_are_explicit() -> None:
    from v9.memory.relations import RelationType
    required = {"OBSERVED_IN", "PRECEDES", "FOLLOWS", "TEMPORALLY_ALIGNED_WITH", "STRUCTURALLY_CORRESPONDS_TO", "PARTICIPATES_IN", "SUPPORTS", "CONTRADICTS", "GROUNDS", "TRANSFER_VALIDATES"}
    assert required <= {row.name for row in RelationType}
