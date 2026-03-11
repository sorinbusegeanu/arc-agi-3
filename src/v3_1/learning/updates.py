from __future__ import annotations

from v3_1.learning.ranker_state import RankerState


def update_ranker_state(state: RankerState, outcome: dict | None) -> RankerState:
    return state if outcome is None else RankerState(ranker_version=state.ranker_version, weights=dict(state.weights))

