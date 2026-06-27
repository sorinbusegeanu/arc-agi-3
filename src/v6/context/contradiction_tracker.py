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
        self.prediction_result_count = 0
        self.prediction_error_positive_count = 0
        self.predicted_family_available_count = 0
        self.actual_family_available_count = 0
        self.wrong_prediction_count = 0
        self.confident_wrong_prediction_count = 0
        self.contradiction_suppressed_missing_prediction_count = 0
        self.contradiction_suppressed_missing_actual_count = 0
        self.contradiction_suppressed_low_confidence_count = 0
        self.contradiction_suppressed_correct_or_unknown_count = 0

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
        self.prediction_result_count += 1
        if prediction_correct is False:
            self.prediction_error_positive_count += 1
        if predicted_family_id is not None:
            self.predicted_family_available_count += 1
        if actual_family_id is not None:
            self.actual_family_available_count += 1
        if prediction_correct is not False:
            self.contradiction_suppressed_correct_or_unknown_count += 1
            return None
        if predicted_family_id is None:
            self.contradiction_suppressed_missing_prediction_count += 1
            return None
        if actual_family_id is None:
            self.contradiction_suppressed_missing_actual_count += 1
            return None
        if not context_signature:
            self.contradiction_suppressed_correct_or_unknown_count += 1
            return None
        if str(predicted_family_id) == str(actual_family_id):
            self.contradiction_suppressed_correct_or_unknown_count += 1
            return None
        self.wrong_prediction_count += 1
        if prediction_confidence is not None and float(prediction_confidence) < self.min_confidence:
            self.contradiction_suppressed_low_confidence_count += 1
            return None
        self.confident_wrong_prediction_count += 1
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
            "prediction_result_count": int(self.prediction_result_count),
            "prediction_error_positive_count": int(self.prediction_error_positive_count),
            "predicted_family_available_count": int(self.predicted_family_available_count),
            "actual_family_available_count": int(self.actual_family_available_count),
            "wrong_prediction_count": int(self.wrong_prediction_count),
            "confident_wrong_prediction_count": int(self.confident_wrong_prediction_count),
            "contradiction_event_count": len(self.events),
            "contradiction_suppressed_missing_prediction_count": int(self.contradiction_suppressed_missing_prediction_count),
            "contradiction_suppressed_missing_actual_count": int(self.contradiction_suppressed_missing_actual_count),
            "contradiction_suppressed_low_confidence_count": int(self.contradiction_suppressed_low_confidence_count),
            "contradiction_suppressed_correct_or_unknown_count": int(self.contradiction_suppressed_correct_or_unknown_count),
            "contradicted_context_count": len(self.by_context),
            "contradicted_context_action_count": len(self.by_context_action),
            "repeated_contradiction_count": sum(1 for count in self.by_contradiction_key.values() if int(count) >= self.min_repeats_for_expansion),
            "top_contradictions": top_contradictions,
        }
