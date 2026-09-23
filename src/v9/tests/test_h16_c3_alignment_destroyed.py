from __future__ import annotations

from v9.research.experiment_manifest import GroundingExposure, apply_grounding_condition
from v9.research.grounding_h16 import GroundingCondition


def test_c3_preserves_marginals_but_destroys_pairing() -> None:
    rows = tuple(GroundingExposure(index, {"world": index}, {"symbol": index}, {"pair": index}) for index in range(8))
    shuffled = apply_grounding_condition(rows, GroundingCondition.C3_SHUFFLED, shuffle_seed=3)
    assert sorted(row.learner_interaction["world"] for row in shuffled) == list(range(8))
    assert sorted(row.learner_symbol["symbol"] for row in shuffled) == list(range(8))
    assert any(row.learner_interaction["world"] != row.learner_symbol["symbol"] for row in shuffled)
