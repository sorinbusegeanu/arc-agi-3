from __future__ import annotations

import ray

from v3_1.learning.ranker_state import RankerState
from v3_1.learning.score_service import score_candidates
from v3_1.learning.updates import update_ranker_state


@ray.remote
class RankerAgent:
    def __init__(self, state: RankerState | None = None) -> None:
        self.state = state or RankerState()

    def score(self, candidates: list[dict]) -> list[dict]:
        return score_candidates(self.state, candidates)

    def update(self, outcome: dict | None):
        self.state = update_ranker_state(self.state, outcome)
        return self.state

    def get_state(self) -> RankerState:
        return self.state
