from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ContextContradictionEvent:
    interaction_id: str
    context_signature: str
    action_signature: str | None
    predicted_family_id: str | None
    actual_family_id: str | None
    prediction_confidence: float | None
    context_depth: int
    contradiction_key: str
    suggested_context_depth: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContextContradictionTracker:
    def __init__(self, *, min_confidence: float = 0.5, min_repeats_for_expansion: int = 2) -> None:
        self.min_confidence = float(min_confidence)
        self.min_repeats_for_expansion = int(min_repeats_for_expansion)
        self.events: list[ContextContradictionEvent] = []
        self.by_context: Counter[str] = Counter()
        self.by_context_action: Counter[str] = Counter()
        self.by_contradiction_key: Counter[str] = Counter()

    def record_prediction_result(
        self,
        *,
        interaction_id: str,
        context_signature: str | None,
        action_signature: str | None,
        predicted_family_id: str | None,
        actual_family_id: str | None,
        prediction_correct: bool | None,
        prediction_confidence: float | None,
        context_depth: int,
        max_context_depth: int | None = None,
    ) -> ContextContradictionEvent | None:
        if prediction_correct is not False:
            return None
        if not context_signature or predicted_family_id is None or actual_family_id is None:
            return None
        if str(predicted_family_id) == str(actual_family_id):
            return None
        if prediction_confidence is not None and float(prediction_confidence) < self.min_confidence:
            return None
        contradiction_key = f"{context_signature}|{action_signature}|{predicted_family_id}->{actual_family_id}"
        suggested_context_depth = min(int(context_depth) + 1, int(max_context_depth or (int(context_depth) + 1)))
        event = ContextContradictionEvent(
            interaction_id=str(interaction_id),
            context_signature=str(context_signature),
            action_signature=None if action_signature is None else str(action_signature),
            predicted_family_id=str(predicted_family_id),
            actual_family_id=str(actual_family_id),
            prediction_confidence=None if prediction_confidence is None else float(prediction_confidence),
            context_depth=int(context_depth),
            contradiction_key=contradiction_key,
            suggested_context_depth=suggested_context_depth,
            reason="confident_wrong_prediction_same_context",
        )
        self.events.append(event)
        self.by_context[str(context_signature)] += 1
        self.by_context_action[f"{context_signature}|{action_signature}"] += 1
        self.by_contradiction_key[contradiction_key] += 1
        return event

    def should_expand_context(self, context_signature: str, action_signature: str | None = None) -> bool:
        if action_signature is not None:
            return int(self.by_context_action.get(f"{context_signature}|{action_signature}", 0)) >= self.min_repeats_for_expansion
        return int(self.by_context.get(str(context_signature), 0)) >= self.min_repeats_for_expansion

    def summary(self) -> dict[str, Any]:
        top_contradictions = [
            {"contradiction_key": key, "count": int(count)}
            for key, count in self.by_contradiction_key.most_common(20)
        ]
        return {
            "context_contradiction_count": len(self.events),
            "contradicted_context_count": len(self.by_context),
            "contradicted_context_action_count": len(self.by_context_action),
            "repeated_contradiction_count": sum(1 for count in self.by_contradiction_key.values() if int(count) >= self.min_repeats_for_expansion),
            "top_contradictions": top_contradictions,
        }
