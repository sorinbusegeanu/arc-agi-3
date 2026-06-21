from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class CarrierCandidate:
    carrier_id: str
    carrier_signature: str
    context_signature: str | None
    action_signature: str | None
    family_id: str | None
    support_count: int
    distinct_family_count: int
    distinct_context_count: int
    prediction_lift: float
    compression_gain: float
    evidence_interaction_ids: list[str]
    status: str = "candidate"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CarrierEvidenceEvent:
    interaction_id: str
    carrier_signature: str
    context_signature: str | None
    action_signature: str | None
    family_id: str | None
    delta_signature: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CarrierEmergenceTracker:
    def __init__(
        self,
        *,
        min_support: int = 3,
        min_distinct_contexts: int = 2,
        min_prediction_lift: float = 0.05,
        min_compression_gain: float = 0.01,
    ) -> None:
        self.min_support = int(min_support)
        self.min_distinct_contexts = int(min_distinct_contexts)
        self.min_prediction_lift = float(min_prediction_lift)
        self.min_compression_gain = float(min_compression_gain)
        self.events: list[CarrierEvidenceEvent] = []
        self.by_carrier: dict[str, list[CarrierEvidenceEvent]] = defaultdict(list)
        self.family_counts_by_carrier: dict[str, Counter[str]] = defaultdict(Counter)
        self.context_counts_by_carrier: dict[str, Counter[str]] = defaultdict(Counter)
        self.prediction_with_carrier: Counter[str] = Counter()
        self.prediction_without_carrier: Counter[str] = Counter()
        self.correct_with_carrier: Counter[str] = Counter()
        self.correct_without_carrier: Counter[str] = Counter()

    def record_interaction(
        self,
        *,
        interaction_id: str,
        carrier_signature: str | None,
        context_signature: str | None,
        action_signature: str | None,
        family_id: str | None,
        delta_signature: str | None,
        prediction_correct: bool | None,
    ) -> CarrierEvidenceEvent | None:
        if carrier_signature is None:
            return None
        event = CarrierEvidenceEvent(
            interaction_id=str(interaction_id),
            carrier_signature=str(carrier_signature),
            context_signature=None if context_signature is None else str(context_signature),
            action_signature=None if action_signature is None else str(action_signature),
            family_id=None if family_id is None else str(family_id),
            delta_signature=None if delta_signature is None else str(delta_signature),
        )
        self.events.append(event)
        self.by_carrier[event.carrier_signature].append(event)
        if event.family_id is not None:
            self.family_counts_by_carrier[event.carrier_signature][event.family_id] += 1
            self.prediction_without_carrier[event.family_id] += 1
            if prediction_correct is True:
                self.correct_without_carrier[event.family_id] += 1
        if event.context_signature is not None:
            self.context_counts_by_carrier[event.carrier_signature][event.context_signature] += 1
        self.prediction_with_carrier[event.carrier_signature] += 1
        if prediction_correct is True:
            self.correct_with_carrier[event.carrier_signature] += 1
        return event

    def build_candidates(self) -> list[CarrierCandidate]:
        return [self._build_candidate(carrier_signature) for carrier_signature in sorted(self.by_carrier)]

    def stats_for_carrier(self, carrier_signature: str) -> dict[str, Any]:
        candidate = self._build_candidate(str(carrier_signature))
        return {
            "carrier_signature": candidate.carrier_signature,
            "carrier_support_count": candidate.support_count,
            "carrier_distinct_family_count": candidate.distinct_family_count,
            "carrier_distinct_context_count": candidate.distinct_context_count,
            "carrier_prediction_lift": candidate.prediction_lift,
            "carrier_compression_gain": candidate.compression_gain,
            "carrier_status": candidate.status,
        }

    def _build_candidate(self, carrier_signature: str) -> CarrierCandidate:
        events = self.by_carrier.get(str(carrier_signature), [])
        support_count = len(events)
        family_counter = self.family_counts_by_carrier.get(str(carrier_signature), Counter())
        context_counter = self.context_counts_by_carrier.get(str(carrier_signature), Counter())
        family_id = family_counter.most_common(1)[0][0] if family_counter else None
        distinct_family_count = len(family_counter)
        distinct_context_count = len(context_counter)
        carrier_predictions = max(1, int(self.prediction_with_carrier.get(str(carrier_signature), 0)))
        carrier_accuracy = float(self.correct_with_carrier.get(str(carrier_signature), 0)) / carrier_predictions
        family_predictions = max(1, int(self.prediction_without_carrier.get(str(family_id), 0))) if family_id is not None else 1
        family_accuracy = float(self.correct_without_carrier.get(str(family_id), 0)) / family_predictions if family_id is not None else 0.0
        prediction_lift = carrier_accuracy - family_accuracy
        compression_gain = max(0.0, 1.0 - (float(distinct_family_count) / max(1, support_count)))
        status = "candidate"
        if (
            support_count >= self.min_support
            and distinct_context_count >= self.min_distinct_contexts
            and prediction_lift >= self.min_prediction_lift
            and compression_gain >= self.min_compression_gain
        ):
            status = "emergent_carrier"
        first = events[0] if events else None
        return CarrierCandidate(
            carrier_id=f"carrier:{carrier_signature}",
            carrier_signature=str(carrier_signature),
            context_signature=None if first is None else first.context_signature,
            action_signature=None if first is None else first.action_signature,
            family_id=family_id,
            support_count=support_count,
            distinct_family_count=distinct_family_count,
            distinct_context_count=distinct_context_count,
            prediction_lift=prediction_lift,
            compression_gain=compression_gain,
            evidence_interaction_ids=[event.interaction_id for event in events[:20]],
            status=status,
        )


def extract_carrier_signature(
    *,
    before_observation: Any,
    after_observation: Any,
    delta: Any,
    context_signature: str | None,
    action_signature: str | None,
) -> str | None:
    del before_observation, after_observation
    for key in ("object_id", "entity_id", "cell_id"):
        value = _lookup(delta, key)
        if value is not None:
            return f"{key}:{value}"
    for key in ("position", "source_position", "target_position", "changed_position"):
        value = _lookup(delta, key)
        position = _position_signature(value)
        if position is not None:
            return f"{key}:{position}"
    positions = _positions_from_delta(delta)
    if positions:
        limited = positions[:8]
        return "cells:" + ";".join(f"({y},{x})" for y, x in limited)
    if context_signature is not None or action_signature is not None:
        return f"context_action:{context_signature}|{action_signature}"
    return None


def _lookup(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    if is_dataclass(value):
        return getattr(value, key, None)
    return getattr(value, key, None)


def _position_signature(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if "y" in value and "x" in value:
            return f"{value['y']},{value['x']}"
        if "row" in value and "col" in value:
            return f"{value['row']},{value['col']}"
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return f"{value[0]},{value[1]}"
    return None


def _positions_from_delta(delta: Any) -> list[tuple[int, int]]:
    for key in ("changed_positions", "changed_cells"):
        value = _lookup(delta, key)
        positions = _coerce_positions(value)
        if positions:
            return positions
    return []


def _coerce_positions(value: Any) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    if value is None:
        return output
    items: Iterable[Any]
    if isinstance(value, Mapping):
        items = value.values()
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return output
    for item in items:
        position = _normalize_position(item)
        if position is not None:
            output.append(position)
    return sorted(output)[:8]


def _normalize_position(item: Any) -> tuple[int, int] | None:
    if isinstance(item, Mapping):
        if "y" in item and "x" in item:
            return (int(item["y"]), int(item["x"]))
        if "row" in item and "col" in item:
            return (int(item["row"]), int(item["col"]))
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return (int(item[0]), int(item[1]))
    return None
