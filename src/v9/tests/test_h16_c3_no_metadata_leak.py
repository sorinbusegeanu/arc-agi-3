from __future__ import annotations

from v9.research.experiment_manifest import GroundingExposure, apply_grounding_condition
from v9.research.grounding_h16 import GroundingCondition


def test_c3_removes_every_learner_visible_alignment_identifier() -> None:
    metadata = {"timestamp": 1, "episode_id": 2, "producer_id": 3, "sequence_position": 4, "provenance_id": 5, "alignment_id": 6, "content": "kept"}
    rows = (GroundingExposure(0, metadata, metadata, {"timestamp": 1, "episode_id": 2}),)
    result = apply_grounding_condition(rows, GroundingCondition.C3_SHUFFLED)[0]
    forbidden = set(metadata) - {"content"}
    assert forbidden.isdisjoint(result.learner_interaction)
    assert forbidden.isdisjoint(result.learner_symbol)
    assert result.system_bookkeeping == rows[0].system_bookkeeping
