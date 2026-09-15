from __future__ import annotations

from typing import Protocol, TypeVar

from v9.memory.identity import MemoryUid


class OutcomeLike(Protocol):
    equivalence_confidence: float
    mean_primary_valence: float


T = TypeVar("T", bound=OutcomeLike)


def _uid(row: OutcomeLike) -> MemoryUid:
    value = getattr(row, "uid", None)
    if value is None:
        value = getattr(row, "outcome_uid")
    return value


def select_target_outcome(outcomes: tuple[T, ...], *, reachable: set[MemoryUid] | None = None) -> T | None:
    eligible = tuple(row for row in outcomes if reachable is None or _uid(row) in reachable)
    if not eligible:
        return None
    return min(eligible, key=lambda row: (-float(row.mean_primary_valence), -float(row.equivalence_confidence), _uid(row)))


def target_outcome_stability(selections: tuple[MemoryUid, ...]) -> float:
    if not selections:
        return 0.0
    counts: dict[MemoryUid, int] = {}
    for uid in selections:
        counts[uid] = counts.get(uid, 0) + 1
    return max(counts.values()) / len(selections)
