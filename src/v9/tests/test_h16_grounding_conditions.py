from __future__ import annotations

from v9.research.experiment_manifest import GroundingExposure, apply_grounding_condition
from v9.research.grounding_h16 import GroundingCondition


def _rows():
    return tuple(GroundingExposure(index, {"world": index}, {"symbol": index}, {"true_alignment": index}) for index in range(4))


def test_c0_c1_c2_are_separate_immutable_exposures() -> None:
    rows = _rows()
    c0 = apply_grounding_condition(rows, GroundingCondition.C0_INTERACTION_ONLY)
    c1 = apply_grounding_condition(rows, GroundingCondition.C1_SYMBOLS_ONLY)
    c2 = apply_grounding_condition(rows, GroundingCondition.C2_ALIGNED)
    assert all(row.learner_symbol is None for row in c0)
    assert all(row.learner_interaction is None for row in c1)
    assert all(row.learner_interaction is not None and row.learner_symbol is not None for row in c2)
    assert rows == _rows()
