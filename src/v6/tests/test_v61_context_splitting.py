from __future__ import annotations

from v6.context.contradiction_tracker import (
    ContextContradictionTracker,
)


def test_repeated_conflicting_outcomes_create_split_proposal() -> None:
    tracker = ContextContradictionTracker(
        min_confidence=0.5,
        min_repeats_for_expansion=2,
    )
    tracker.record_prediction_result(
        interaction_id="1",
        context_signature='[1, 2]',
        action_signature="3",
        predicted_family_id="A",
        actual_family_id="B",
        prediction_correct=False,
        prediction_confidence=0.9,
        context_depth=1,
        max_context_depth=3,
    )
    event = tracker.record_prediction_result(
        interaction_id="2",
        context_signature='[1, 2]',
        action_signature="3",
        predicted_family_id="A",
        actual_family_id="C",
        prediction_correct=False,
        prediction_confidence=0.9,
        context_depth=1,
        max_context_depth=3,
    )

    assert event is not None
    assert event.split_proposal_id is not None
    proposals = tracker.split_proposals()
    assert len(proposals) == 1
    assert proposals[0].conflicting_actual_families == ("B", "C")
