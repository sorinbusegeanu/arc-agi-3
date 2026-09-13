from __future__ import annotations

from dataclasses import dataclass

from .codec import SymbolObservation


@dataclass(frozen=True, slots=True)
class SymbolRelation:
    relation: str
    left: SymbolObservation
    right: SymbolObservation | None = None


def observable_relations(stream: tuple[SymbolObservation, ...], *, limit: int) -> tuple[SymbolRelation, ...]:
    if limit <= 0:
        raise ValueError("relation limit must be positive")
    rows: list[SymbolRelation] = []
    for index, current in enumerate(stream):
        rows.append(SymbolRelation("OCCURRED", current))
        if index:
            rows.append(SymbolRelation("PRECEDES", stream[index - 1], current))
        if len(rows) >= limit:
            break
    return tuple(rows[:limit])

