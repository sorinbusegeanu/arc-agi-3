from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .m1_grounded import M1GroundedContingency
from .m1_normalized import M1NormalizedRelation, NormalizedChannel
from .symbolic_grounding import SymbolicRelation
from v9.modalities.symbols import SymbolTemporalPhase


@dataclass(frozen=True, slots=True)
class DerivedSymbolicRelation:
    relation: M1NormalizedRelation
    symbol_identity: tuple[int, int, int, int] | None
    relation_kind: str
    control: str = "aligned"
    temporal_phase: str = SymbolTemporalPhase.COINCIDENT.value

    def payload(self) -> dict[str, Any]:
        return {
            "symbol_identity": self.symbol_identity,
            "symbol_relation": self.relation_kind,
            "family_signature": int(self.relation.family_signature or self.relation.structural_signature),
            "support": float(self.relation.support),
            "contradiction": float(self.relation.contradiction),
            "temporal_offsets": list(self.relation.temporal_offsets),
            "causal_watermark": int(self.relation.causal_watermark),
            "cross_modal_control": self.control,
            "symbol_temporal_phase": self.temporal_phase,
            "heldout_transfer": bool(self.relation.heldout_transfer),
        }


def _identity(row: Any) -> tuple[int, int, int, int] | None:
    m0 = getattr(row, "m0", None)
    if m0 is None:
        writes = getattr(row, "base_writes", ())
        if writes:
            value = writes[0].payload.get("symbol_identity")
            if value is not None:
                return tuple(int(item) for item in value)
        return None
    value = getattr(m0, "symbol_identity", None)
    if value is None:
        return None
    return tuple(int(item) for item in value)


def _grounding(row: Any) -> M1GroundedContingency:
    value = getattr(row, "m1g", None)
    if value is None:
        value = getattr(row, "relation", None)
        parents = getattr(value, "provenance", None)
        if parents is None:
            raise ValueError("symbol relation lacks grounded provenance")
        # Commit plans do not retain the grounded object; caller should only use
        # this helper with prepared rows. Canonical commit derives before flattening.
        raise ValueError("grounded symbolic row is required")
    return value


def _interaction_family_key(row: M1GroundedContingency) -> tuple[object, ...]:
    return (
        "INTERACTION_TRANSITION",
        int(row.environment_instance_id),
        int(row.grounded_context_signature),
        -1 if row.executable_action_token is None else int(row.executable_action_token),
        int(row.realized_transition_signature),
        int(row.grounded_next_context_signature),
    )


def _occurrence_map(occurrences: Iterable[Any]) -> dict[tuple[int, int], Any]:
    result: dict[tuple[int, int], Any] = {}
    for occurrence in occurrences:
        result[(int(occurrence.occurrence_id.hi), int(occurrence.occurrence_id.lo))] = occurrence
    return result


def _occurrence(row: Any, by_event: dict[tuple[int, int], Any]) -> Any | None:
    direct = getattr(row, "occurrence", None)
    if direct is not None:
        return direct
    event = getattr(row, "event", None)
    if event is None:
        return None
    identity = event.identity.event_id
    return by_event.get((int(identity.hi), int(identity.lo)))


def _phase(row: Any, by_event: dict[tuple[int, int], Any]) -> str:
    occurrence = _occurrence(row, by_event)
    if occurrence is not None:
        return str(occurrence.temporal_phase)
    return SymbolTemporalPhase.COINCIDENT.value


def _symbol_key(identity: tuple[int, int, int, int] | None) -> tuple[object, ...]:
    if identity is None:
        return ("UNKNOWN_SYMBOL",)
    return ("SYMBOL", int(identity[0]), int(identity[2]))


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
    temporal_phase: str = SymbolTemporalPhase.COINCIDENT.value,
    family_key: tuple[object, ...] | None = None,
    structural_extra: tuple[object, ...] = (),
    heldout_transfer: bool = False,
) -> DerivedSymbolicRelation:
    structural_key = (kind, *_symbol_key(symbol_identity), *structural_extra)
    return DerivedSymbolicRelation(
        M1NormalizedRelation.build(
            kind,
            channel,
            parents,
            support=support,
            contradiction=contradiction,
            temporal_offsets=temporal_offsets,
            causal_watermark=causal_watermark,
            structural_key=structural_key,
            family_key=family_key or structural_key,
            heldout_transfer=heldout_transfer,
        ),
        symbol_identity,
        kind,
        control,
        temporal_phase,
    )


def derive_symbolic_relations(
    symbol_rows: Iterable[Any],
    *,
    interaction_grounding: M1GroundedContingency | None,
    previous_interaction_grounding: M1GroundedContingency | None,
    transition: Any,
    causal_watermark: int,
    occurrences: Iterable[Any] = (),
) -> tuple[DerivedSymbolicRelation, ...]:
    """Derive structural symbolic/cross-modal M1N evidence without semantics."""
    rows = tuple(symbol_rows)
    by_event = _occurrence_map(occurrences)
    derived: list[DerivedSymbolicRelation] = []

    for left, right in zip(rows, rows[1:]):
        left_id, right_id = _identity(left), _identity(right)
        left_occ, right_occ = _occurrence(left, by_event), _occurrence(right, by_event)
        if left_occ is not None and right_occ is not None:
            offset = int(right_occ.micro_step) - int(left_occ.micro_step)
        else:
            left_pos = 0 if left_id is None else int(left_id[3])
            right_pos = 0 if right_id is None else int(right_id[3])
            offset = right_pos - left_pos
        pair_key = ("SYMBOL_PAIR", *_symbol_key(left_id), *_symbol_key(right_id))
        derived.append(_make(SymbolicRelation.SYMBOL_PRECEDES_SYMBOL, NormalizedChannel.SYMBOL, (_grounding(left), _grounding(right)), symbol_identity=left_id, causal_watermark=causal_watermark, temporal_offsets=(offset,), temporal_phase=_phase(left, by_event), family_key=pair_key, structural_extra=_symbol_key(right_id)))
        derived.append(_make(SymbolicRelation.SYMBOL_FOLLOWS_SYMBOL, NormalizedChannel.SYMBOL, (_grounding(right), _grounding(left)), symbol_identity=right_id, causal_watermark=causal_watermark, temporal_offsets=(-offset,), temporal_phase=_phase(right, by_event), family_key=pair_key, structural_extra=_symbol_key(left_id)))

    by_symbol: dict[tuple[int, int, int], list[Any]] = {}
    for row in rows:
        identity = _identity(row)
        if identity is None:
            continue
        key = identity[:3]
        prior = by_symbol.setdefault(key, [])
        if prior:
            first = prior[-1]
            first_occ, current_occ = _occurrence(first, by_event), _occurrence(row, by_event)
            if first_occ is not None and current_occ is not None:
                offset = int(current_occ.micro_step) - int(first_occ.micro_step)
            else:
                first_id = _identity(first)
                offset = int(identity[3]) - (0 if first_id is None else int(first_id[3]))
            derived.append(_make(SymbolicRelation.SYMBOL_RECURS_WITHIN_WINDOW, NormalizedChannel.SYMBOL, (_grounding(first), _grounding(row)), symbol_identity=identity, causal_watermark=causal_watermark, temporal_offsets=(offset,), temporal_phase=_phase(row, by_event), family_key=("SYMBOL_RECURRENCE", *_symbol_key(identity))))
        prior.append(row)

    changed = int(getattr(transition, "before_signature", 0)) != int(getattr(transition, "after_signature", 0))
    boundary = str(getattr(transition, "boundary_scope", "NONE")) != "NONE" or bool(getattr(transition, "task_success", False)) or bool(getattr(transition, "task_failure", False)) or bool(getattr(transition, "task_truncated", False))
    progress = bool(getattr(transition, "levels_completed", 0)) or bool(getattr(transition, "task_success", False))
    outcome = int(getattr(transition, "primary_valence", 0)) != 0 or bool(getattr(transition, "task_success", False)) or bool(getattr(transition, "task_failure", False))

    for row in rows:
        identity = _identity(row)
        phase = _phase(row, by_event)
        before_action = phase == SymbolTemporalPhase.BEFORE_ACTION.value
        after_action = phase in {SymbolTemporalPhase.AFTER_ACTION.value, SymbolTemporalPhase.AFTER_OUTCOME.value, SymbolTemporalPhase.BETWEEN_ACTIONS.value}
        coincident = phase == SymbolTemporalPhase.COINCIDENT.value
        current_family = None if interaction_grounding is None else _interaction_family_key(interaction_grounding)
        previous_family = None if previous_interaction_grounding is None else _interaction_family_key(previous_interaction_grounding)

        if interaction_grounding is not None:
            derived.append(_make(SymbolicRelation.SYMBOL_INTERACTION_ALIGNMENT, NormalizedChannel.CROSS_MODAL, (_grounding(row), interaction_grounding), symbol_identity=identity, causal_watermark=causal_watermark, temporal_offsets=(0,), temporal_phase=phase, family_key=current_family))
            if before_action:
                derived.append(_make(SymbolicRelation.SYMBOL_PRECEDES_ACTION, NormalizedChannel.CROSS_MODAL, (_grounding(row), interaction_grounding), symbol_identity=identity, causal_watermark=causal_watermark, temporal_offsets=(-1,), temporal_phase=phase, family_key=current_family))
                if changed:
                    derived.append(_make(SymbolicRelation.SYMBOL_PRECEDES_NORMALIZED_CHANGE, NormalizedChannel.CROSS_MODAL, (_grounding(row), interaction_grounding), symbol_identity=identity, causal_watermark=causal_watermark, temporal_offsets=(-1,), temporal_phase=phase, family_key=current_family))
            elif after_action or coincident:
                derived.append(_make(SymbolicRelation.SYMBOL_FOLLOWS_ACTION, NormalizedChannel.CROSS_MODAL, (_grounding(row), interaction_grounding), symbol_identity=identity, causal_watermark=causal_watermark, temporal_offsets=(0 if coincident else 1,), temporal_phase=phase, family_key=current_family))
                if changed:
                    derived.append(_make(SymbolicRelation.SYMBOL_FOLLOWS_NORMALIZED_CHANGE, NormalizedChannel.CROSS_MODAL, (_grounding(row), interaction_grounding), symbol_identity=identity, causal_watermark=causal_watermark, temporal_offsets=(0 if coincident else 1,), temporal_phase=phase, family_key=current_family))
        elif previous_interaction_grounding is not None and after_action:
            derived.append(_make(SymbolicRelation.SYMBOL_FOLLOWS_ACTION, NormalizedChannel.CROSS_MODAL, (_grounding(row), previous_interaction_grounding), symbol_identity=identity, causal_watermark=causal_watermark, temporal_offsets=(1,), temporal_phase=phase, family_key=previous_family))

        local_family = current_family or previous_family or ("SYMBOL_CONTEXT", *_symbol_key(identity))
        if boundary:
            derived.append(_make(SymbolicRelation.SYMBOL_NEAR_BOUNDARY, NormalizedChannel.CROSS_MODAL, (_grounding(row),), symbol_identity=identity, causal_watermark=causal_watermark, temporal_offsets=(0,), temporal_phase=phase, family_key=local_family))
        if progress:
            derived.append(_make(SymbolicRelation.SYMBOL_COINCIDENT_WITH_PROGRESS, NormalizedChannel.CROSS_MODAL, (_grounding(row),), symbol_identity=identity, causal_watermark=causal_watermark, temporal_offsets=(0,), temporal_phase=phase, family_key=local_family))
        if outcome and phase in {SymbolTemporalPhase.AFTER_OUTCOME.value, SymbolTemporalPhase.AFTER_ACTION.value, SymbolTemporalPhase.COINCIDENT.value}:
            derived.append(_make(SymbolicRelation.SYMBOL_COINCIDENT_WITH_OUTCOME, NormalizedChannel.CROSS_MODAL, (_grounding(row),), symbol_identity=identity, causal_watermark=causal_watermark, temporal_offsets=(0,), temporal_phase=phase, family_key=local_family))

    return tuple(derived)


def shuffled_alignment_control(
    symbol_row: Any,
    interaction_grounding: M1GroundedContingency,
    *,
    causal_watermark: int,
    occurrence: Any | None = None,
) -> DerivedSymbolicRelation:
    """Construct deterministic negative/control evidence against a mismatched interaction."""
    identity = _identity(symbol_row)
    family_key = _interaction_family_key(interaction_grounding)
    phase = str(getattr(occurrence, "temporal_phase", SymbolTemporalPhase.COINCIDENT.value))
    return _make(
        SymbolicRelation.CROSS_MODAL_CORRESPONDENCE,
        NormalizedChannel.CROSS_MODAL,
        (_grounding(symbol_row), interaction_grounding),
        symbol_identity=identity,
        causal_watermark=causal_watermark,
        support=0.0,
        contradiction=1.0,
        control="shuffled",
        temporal_phase=phase,
        family_key=family_key,
        structural_extra=("SHUFFLED",),
    )
