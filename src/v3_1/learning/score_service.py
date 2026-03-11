from __future__ import annotations

from v3_1.learning.ranker_state import RankerState


def score_candidates(state: RankerState, candidates: list[dict]) -> list[dict]:
    if not state.weights:
        return candidates
    scored = []
    for row in candidates:
        bonus = 0.0
        for key, weight in state.weights.items():
            bonus += float(row.get(key, 0.0)) * float(weight)
        scored.append(dict(row, final_score=float(row.get("final_score", row.get("score", 0.0))) + bonus))
    return scored

