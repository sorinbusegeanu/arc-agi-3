from __future__ import annotations

from v9.hgt.grounding_objectives import GROUNDING_OBJECTIVES, GroundingObjectiveEvidence, validate_objective_evidence
from v9.memory.m1_normalized import M1NormalizedRelation, NormalizedChannel
from v9.memory.symbolic_grounding import SymbolicRelation
from v9.memory.relations import RelationType


def test_complete_symbolic_relation_contract() -> None:
    required = {
        "SYMBOL_PRECEDES_SYMBOL", "SYMBOL_FOLLOWS_SYMBOL", "SYMBOL_RECURS_WITHIN_WINDOW",
        "SYMBOL_PRECEDES_ACTION", "SYMBOL_FOLLOWS_ACTION",
        "SYMBOL_PRECEDES_NORMALIZED_CHANGE", "SYMBOL_FOLLOWS_NORMALIZED_CHANGE",
        "SYMBOL_NEAR_BOUNDARY", "SYMBOL_COINCIDENT_WITH_PROGRESS", "SYMBOL_COINCIDENT_WITH_OUTCOME",
        "SYMBOL_INTERACTION_ALIGNMENT", "SYMBOL_TO_INTERACTION_PREDICTION",
        "INTERACTION_TO_SYMBOL_GENERALIZATION", "CROSS_MODAL_HELDOUT_TRANSFER", "CROSS_MODAL_COMPOSITION",
    }
    assert all(hasattr(SymbolicRelation, name) for name in required)
    assert all(hasattr(RelationType, name) for name in required)


def test_grounding_objectives_require_positive_negative_evidence() -> None:
    objective = GROUNDING_OBJECTIVES[0]
    rows = (
        GroundingObjectiveEvidence(objective, 1.0, 0.8, 1, 2, "p", "aligned"),
        GroundingObjectiveEvidence(objective, 0.0, 0.2, 1, 2, "n", "shuffled"),
    )
    result = validate_objective_evidence(rows)
    assert result["positive_examples"] == 1
    assert result["negative_examples"] == 1
    assert result["control_groups"] == 2
