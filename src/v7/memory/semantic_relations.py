from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class SemanticRelation(IntEnum):
    PRECEDES = 90_001
    ENABLES = 90_002
    CONSTRAINS = 90_003
    PRESERVES = 90_004
    OPENS_OPTIONS = 90_005
    CLOSES_OPTIONS = 90_006
    CAUSES_PROGRESS = 90_007
    CAUSES_REGRESSION = 90_008
    SHARED_OUTCOME = 90_009


REL_PRECEDES = int(SemanticRelation.PRECEDES)
REL_ENABLES = int(SemanticRelation.ENABLES)
REL_CONSTRAINS = int(SemanticRelation.CONSTRAINS)
REL_PRESERVES = int(SemanticRelation.PRESERVES)
REL_OPENS_OPTIONS = int(SemanticRelation.OPENS_OPTIONS)
REL_CLOSES_OPTIONS = int(SemanticRelation.CLOSES_OPTIONS)
REL_CAUSES_PROGRESS = int(SemanticRelation.CAUSES_PROGRESS)
REL_CAUSES_REGRESSION = int(SemanticRelation.CAUSES_REGRESSION)
REL_SHARED_OUTCOME = int(SemanticRelation.SHARED_OUTCOME)

TYPE_RELATIONAL_WORLD_MODEL = 510


@dataclass(frozen=True, slots=True)
class RelationObservation:
    source_concept: int
    target_concept: int
    relation_type: int
    context_signature: int
    action_id: int
    support: int = 1


def classify_transition_relations(
    *,
    prior_concepts: tuple[int, ...],
    current_concepts: tuple[int, ...],
    future_option_delta: float,
    terminal_polarity: int,
) -> tuple[tuple[int, int, int], ...]:
    """Derive deterministic typed relations from one observed abstract transition."""
    prior = tuple(sorted(set(int(v) for v in prior_concepts)))
    current = tuple(sorted(set(int(v) for v in current_concepts)))
    if not prior or not current:
        return ()
    rows: set[tuple[int, int, int]] = set()
    for source in prior:
        for target in current:
            rows.add((source, REL_PRECEDES, target))
            if source == target:
                rows.add((source, REL_PRESERVES, target))
            if float(future_option_delta) > 0.0:
                rows.add((source, REL_OPENS_OPTIONS, target))
                rows.add((source, REL_ENABLES, target))
            elif float(future_option_delta) < 0.0:
                rows.add((source, REL_CLOSES_OPTIONS, target))
                rows.add((source, REL_CONSTRAINS, target))
            if int(terminal_polarity) > 0:
                rows.add((source, REL_CAUSES_PROGRESS, target))
                rows.add((source, REL_ENABLES, target))
            elif int(terminal_polarity) < 0:
                rows.add((source, REL_CAUSES_REGRESSION, target))
                rows.add((source, REL_CONSTRAINS, target))
    if len(current) >= 2:
        for source in current:
            for target in current:
                if source < target:
                    rows.add((source, REL_SHARED_OUTCOME, target))
                    rows.add((target, REL_SHARED_OUTCOME, source))
    return tuple(sorted(rows))
