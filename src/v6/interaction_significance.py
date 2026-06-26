from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any, Mapping


def clamp01(value: float) -> float:
    if value != value:
        return 0.0
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class InteractionSignificanceScore:
    survival_impact: float
    prediction_error: float
    learning_value: float
    transfer_potential: float
    explanatory_potential: float
    total: float
    weights: dict[str, float]
    outcome_state: str | None = None
    outcome_polarity: str | None = None
    version: str = "isf_v02"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_ISF_WEIGHTS = {
    "survival_impact": 0.20,
    "prediction_error": 0.25,
    "learning_value": 0.25,
    "transfer_potential": 0.15,
    "explanatory_potential": 0.15,
}


def scalarize_reward(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        numbers = [abs(float(item)) for item in value.values() if isinstance(item, (int, float)) and not isinstance(item, bool)]
        return max(numbers, default=0.0)
    if isinstance(value, (list, tuple)):
        numbers = [abs(float(item)) for item in value if isinstance(item, (int, float)) and not isinstance(item, bool)]
        return max(numbers, default=0.0)
    return 0.0


def bool_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return any(bool_scalar(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(bool_scalar(item) for item in value)
    return False


def compute_interaction_significance(
    *,
    reward: Any,
    terminated: Any,
    truncated: Any,
    prediction_correct: bool | None,
    prediction_confidence: float | None,
    actual_family_id: str | None,
    delta_id: str | None,
    context_signature: str | None,
    memory_counts: Mapping[str, int] | None = None,
    graph_counts: Mapping[str, int] | None = None,
    future_option_delta: float | None = None,
    explanatory_delta: float | None = None,
    outcome_state: str | None = None,
    outcome_polarity: str | None = None,
    weights: Mapping[str, float] | None = None,
) -> InteractionSignificanceScore:
    normalized_weights = _normalize_weights(weights)
    memory_counts = memory_counts or {}
    graph_counts = graph_counts or {}

    reward_value = scalarize_reward(reward)
    survival_impact = clamp01(abs(reward_value))
    if bool_scalar(terminated) or bool_scalar(truncated):
        survival_impact = max(survival_impact, 0.75)
    normalized_outcome_state = None if outcome_state in (None, "") else str(outcome_state)
    normalized_outcome_polarity = None if outcome_polarity in (None, "") else str(outcome_polarity)
    if normalized_outcome_state == "game_won":
        survival_impact = 1.0
        normalized_outcome_polarity = "positive"
    elif normalized_outcome_state == "dead":
        survival_impact = 1.0
        normalized_outcome_polarity = "negative"
    elif normalized_outcome_state == "end_game":
        survival_impact = max(survival_impact, 0.85)
        normalized_outcome_polarity = normalized_outcome_polarity or "unknown"
    elif normalized_outcome_state == "level_advanced":
        survival_impact = max(survival_impact, 0.75)
        normalized_outcome_polarity = "positive"
    elif normalized_outcome_state == "alive":
        survival_impact = max(survival_impact, 0.0)
        normalized_outcome_polarity = normalized_outcome_polarity or "neutral"

    if prediction_correct is True:
        prediction_error = 1.0 - clamp01(prediction_confidence or 0.0)
    elif prediction_correct is False:
        prediction_error = max(0.5, clamp01(prediction_confidence or 0.5))
    else:
        prediction_error = 0.5

    novelty_scores: list[float] = []
    if delta_id:
        novelty_scores.append(_novelty(memory_counts.get(f"delta_id:{delta_id}", 0)))
    if actual_family_id:
        novelty_scores.append(_novelty(memory_counts.get(f"actual_family_id:{actual_family_id}", 0)))
    if context_signature:
        novelty_scores.append(_novelty(memory_counts.get(f"context_signature:{context_signature}", 0)))
    if context_signature and actual_family_id:
        novelty_scores.append(_novelty(memory_counts.get(f"context_family:{context_signature}|{actual_family_id}", 0)))
    if normalized_outcome_state:
        novelty_scores.append(_novelty(memory_counts.get(f"outcome_state:{normalized_outcome_state}", 0)))
    if context_signature and normalized_outcome_state:
        novelty_scores.append(_novelty(memory_counts.get(f"context_outcome:{context_signature}|{normalized_outcome_state}", 0)))
    learning_value = max(novelty_scores) if novelty_scores else 0.5
    if normalized_outcome_state in {"game_won", "dead", "end_game", "level_advanced"}:
        progress_outcome_learning_value = _novelty(memory_counts.get(f"outcome_state:{normalized_outcome_state}", 0))
        learning_value = max(learning_value, progress_outcome_learning_value)

    if future_option_delta is not None:
        transfer_potential = clamp01(abs(future_option_delta))
    else:
        transfer_potential = 0.0
    if actual_family_id and int(memory_counts.get(f"actual_family_id:{actual_family_id}", 0)) <= 2:
        transfer_potential = max(transfer_potential, 0.4)

    if explanatory_delta is not None:
        explanatory_potential = clamp01(abs(explanatory_delta))
    else:
        explanatory_potential = 0.0
    if int(graph_counts.get("new_contingency", 0)) > 0:
        explanatory_potential = max(explanatory_potential, 0.5)
    if int(graph_counts.get("new_graph_edge", 0)) > 0:
        explanatory_potential = max(explanatory_potential, 0.35)

    total = clamp01(
        survival_impact * normalized_weights["survival_impact"]
        + prediction_error * normalized_weights["prediction_error"]
        + learning_value * normalized_weights["learning_value"]
        + transfer_potential * normalized_weights["transfer_potential"]
        + explanatory_potential * normalized_weights["explanatory_potential"]
    )
    return InteractionSignificanceScore(
        survival_impact=survival_impact,
        prediction_error=prediction_error,
        learning_value=learning_value,
        transfer_potential=transfer_potential,
        explanatory_potential=explanatory_potential,
        total=total,
        weights=normalized_weights,
        outcome_state=normalized_outcome_state,
        outcome_polarity=normalized_outcome_polarity,
    )


def _novelty(prior_count: int) -> float:
    return clamp01(1.0 / sqrt(1.0 + max(0, int(prior_count))))


def _normalize_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    merged = {key: float(value) for key, value in DEFAULT_ISF_WEIGHTS.items()}
    if weights:
        for key, value in weights.items():
            if key in merged:
                merged[key] = max(0.0, float(value))
    total = sum(merged.values())
    if total <= 0:
        return dict(DEFAULT_ISF_WEIGHTS)
    return {key: float(value) / total for key, value in merged.items()}
