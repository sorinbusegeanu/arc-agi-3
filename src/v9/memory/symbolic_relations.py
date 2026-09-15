from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .m1_grounded import M1GroundedContingency
from .m1_normalized import M1NormalizedRelation, NormalizedChannel
from .symbolic_grounding import SymbolicRelation


@dataclass(frozen=True, slots=True)
class DerivedSymbolicRelation:
    relation: M1NormalizedRelation
    symbol_identity: tuple[int, int, int, int] | None
    relation_kind: str
    control: str = "aligned"

    def payload(self) -> dict[str, Any]:
        return {
            "symbol_identity": self.symbol_identity,
            "symbol_relation": self.relation_kind,
            "support": float(self.relation.support),
            "contradiction": float(self.relation.contradiction),
            "temporal_offsets": list(self.relation.temporal_offsets),
            "causal_watermark": int(self.relation.causal_watermark),
            "cross_modal_control": self.control,
        }


def _identity(row: Any) -> tuple[int, int, int, int] | None:
    value = getattr(row.m0, "symbol_identity", None)
    if value is None:
        return None
    return tuple(int(item) for item in value)


def _make(
    kind: str,
    channel: NormalizedChannel,
    parents: tuple[M1GroundedContingency, ...],
    *,
    symbol_identity: tuple[int, int, int, int] | None,
    causal_watermark: int,
    temporal_offsets: tuple[int, ...] = (),
    support: float = 1.0,
    contradiction: float = 0.0,
    control: str = "aligned",
) -> DerivedSymbolicRelation:
    return DerivedSymbolicRelation(
        M1NormalizedRelation.build(
            kind,
            channel,
            parents,
            support=support,
            contradiction=contradiction,
            temporal_offsets=temporal_offsets,
            causal_watermark=causal_watermark,
        ),
        symbol_identity,
        kind,
        control,
    )


def derive_symbolic_relations(
    symbol_rows: Iterable[Any],
    *,
    interaction_grounding: M1GroundedContingency | None,
    previous_interaction_grounding: M1GroundedContingency | None,
    transition: Any,
    causal_watermark: int,
) -> tuple[DerivedSymbolicRelation, ...]:
    """Derive structural symbolic/cross-modal M1N evidence without semantics.

    Symbols are treated as opaque identities. Relations use only temporal order,
    recurrence, interaction boundaries, observable transition change and outcomes.
    """
    rows = tuple(symbol_rows)
    derived: list[DerivedSymbolicRelation] = []

    for left, right in zip(rows, rows[1:]):
        left_id, right_id = _identity(left), _identity(right)
        left_pos = 0 if left_id is None else int(left_id[3])
        right_pos = 0 if right_id is None else int(right_id[3])
        offset = right_pos - left_pos
        derived.append(_make(SymbolicRelation.SYMBOL_PRECEDES_SYMBOL, NormalizedChannel.SYMBOL, (left.m1g, right.m1g), symbol_identity=left_id, causal_watermark=causal_watermark, temporal_offsets=(offset,)))
        derived.append(_make(SymbolicRelation.SYMBOL_FOLLOWS_SYMBOL, NormalizedChannel.SYMBOL, (right.m1g, left.m1g), symbol_identity=right_id, causal_watermark=causal_watermark, temporal_offsets=(-offset,)))

    by_symbol: dict[tuple[int, int, int], list[Any]] = {}
    for row in rows:
        identity = _identity(row)
        if identity is None:
            continue
        key = identity[:3]
        prior = by_symbol.setdefault(key, [])
        if prior:
            first = prior[-1]
            first_id = _identity(first)
            offset = int(identity[3]) - (0 if first_id is None else int(first_id[3]))
            derived.append(_make(SymbolicRelation.SYMBOL_RECURS_WITHIN_WINDOW, NormalizedChannel.SYMBOL, (first.m1g, row.m1g), symbol_identity=identity, causal_watermark=causal_watermark, temporal_offsets=(offset,)))
        prior.append(row)

    changed = int(getattr(transition, "before_signature", 0)) != int(getattr(transition, "after_signature", 0))
    boundary = str(getattr(transition, "boundary_scope", "NONE")) != "NONE" or bool(getattr(transition, "task_success", False)) or bool(getattr(transition, "task_failure", False)) or bool(getattr(transition, "task_truncated", False))
    progress = bool(getattr(transition, "levels_completed", 0)) or bool(getattr(transition, "task_success", False))
    outcome = int(getattr(transition, "primary_valence", 0)) != 0 or bool(getattr(transition, "task_success", False)) or bool(getattr(transition, "task_failure", False))

    for row in rows:
        identity = _identity(row)
        if interaction_grounding is not None:
            derived.append(_make(SymbolicRelation.SYMBOL_PRECEDES_ACTION, NormalizedChannel.CROSS_MODAL, (row.m1g, interaction_grounding), symbol_identity=identity, causal_watermark=causal_watermark, temporal_offsets=(0,)))
            derived.append(_make(SymbolicRelation.SYMBOL_INTERACTION_ALIGNMENT, NormalizedChannel.CROSS_MODAL, (row.m1g, interaction_grounding), symbol_identity=identity, causal_watermark=causal_watermark, temporal_offsets=(0,)))
            if changed:
                derived.append(_make(SymbolicRelation.SYMBOL_PRECEDES_NORMALIZED_CHANGE, NormalizedChannel.CROSS_MODAL, (row.m1g, interaction_grounding), symbol_identity=identity, causal_watermark=causal_watermark, temporal_offsets=(0,)))
        if previous_interaction_grounding is not None:
            derived.append(_make(SymbolicRelation.SYMBOL_FOLLOWS_ACTION, NormalizedChannel.CROSS_MODAL, (row.m1g, previous_interaction_grounding), symbol_identity=identity, causal_watermark=causal_watermark, temporal_offsets=(1,)))
            derived.append(_make(SymbolicRelation.SYMBOL_FOLLOWS_NORMALIZED_CHANGE, NormalizedChannel.CROSS_MODAL, (row.m1g, previous_interaction_grounding), symbol_identity=identity, causal_watermark=causal_watermark, temporal_offsets=(1,)))
        if boundary:
            derived.append(_make(SymbolicRelation.SYMBOL_NEAR_BOUNDARY, NormalizedChannel.CROSS_MODAL, (row.m1g,), symbol_identity=identity, causal_watermark=causal_watermark, temporal_offsets=(0,)))
        if progress:
            derived.append(_make(SymbolicRelation.SYMBOL_COINCIDENT_WITH_PROGRESS, NormalizedChannel.CROSS_MODAL, (row.m1g,), symbol_identity=identity, causal_watermark=causal_watermark, temporal_offsets=(0,)))
        if outcome:
            derived.append(_make(SymbolicRelation.SYMBOL_COINCIDENT_WITH_OUTCOME, NormalizedChannel.CROSS_MODAL, (row.m1g,), symbol_identity=identity, causal_watermark=causal_watermark, temporal_offsets=(0,)))

    return tuple(derived)


def shuffled_alignment_control(
    symbol_row: Any,
    interaction_grounding: M1GroundedContingency,
    *,
    causal_watermark: int,
) -> DerivedSymbolicRelation:
    """Construct a negative control without changing the symbol identity itself."""
    return _make(
        SymbolicRelation.CROSS_MODAL_CORRESPONDENCE,
        NormalizedChannel.CROSS_MODAL,
        (symbol_row.m1g, interaction_grounding),
        symbol_identity=_identity(symbol_row),
        causal_watermark=causal_watermark,
        support=0.0,
        contradiction=1.0,
        control="shuffled",
    )
