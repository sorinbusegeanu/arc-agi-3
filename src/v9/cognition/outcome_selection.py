from __future__ import annotations

from v9.memory.m6_outcome import M6Outcome
from v9.memory.identity import MemoryUid


def select_target_outcome(outcomes: tuple[M6Outcome, ...], *, reachable: set[MemoryUid] | None = None) -> M6Outcome | None:
    eligible = tuple(row for row in outcomes if reachable is None or row.uid in reachable)
    if not eligible:
        return None
    return min(eligible, key=lambda row: (-row.mean_primary_valence, -row.equivalence_confidence, row.uid))


def target_outcome_stability(selections: tuple[MemoryUid, ...]) -> float:
    if not selections:
        return 0.0
    counts: dict[MemoryUid, int] = {}
    for uid in selections:
        counts[uid] = counts.get(uid, 0) + 1
    return max(counts.values()) / len(selections)
