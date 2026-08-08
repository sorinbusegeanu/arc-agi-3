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
    transfer_prior: float = 0.0
    transfer_empirical_rate: float | None = None
    component_active: dict[str, bool] | None = None
    version: str = "isf_v63"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_ISF_WEIGHTS = {
    "survival_impact": 0.20,
    "prediction_error": 0.25,
    "learning_value": 0.25,
    "transfer_potential": 0.15,
    "explanatory_potential": 0.15,
}


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
    level_completed_event: bool = False,
    weights: Mapping[str, float] | None = None,
) -> InteractionSignificanceScore:
    normalized_weights = _normalize_weights(weights)
    memory_counts = memory_counts or {}
    graph_counts = graph_counts or {}

    # ISF is not reward, value, or utility.
    # survival_impact means effect on continued agency/future-interaction possibility.
    # High survival_impact means memory significance, not positive outcome.
    survival_impact = 0.0
    normalized_outcome_state = None if outcome_state in (None, "") else str(outcome_state)
    normalized_outcome_polarity = None if outcome_polarity in (None, "") else str(outcome_polarity)
    if normalized_outcome_state == "GAME_OVER":
        survival_impact = 1.0
        normalized_outcome_polarity = "negative"
    elif normalized_outcome_state == "WIN":
        survival_impact = 0.75
        normalized_outcome_polarity = "positive"
    elif bool(level_completed_event):
        survival_impact = max(survival_impact, 0.50)
        normalized_outcome_polarity = "positive"
    elif normalized_outcome_state == "NOT_FINISHED":
        survival_impact = max(survival_impact, 0.0)
        normalized_outcome_polarity = normalized_outcome_polarity or "neutral"
    if bool_scalar(terminated) or bool_scalar(truncated):
        survival_impact = max(survival_impact, 0.50)
    if future_option_delta is not None and float(future_option_delta) < 0.0:
        survival_impact = max(survival_impact, clamp01(abs(float(future_option_delta))))

    # v6.3: no supported expectation means PE is inactive. It must not be
    # represented by a synthetic 0.5 surprise value during contingency seeding.
    prediction_active = (
        prediction_correct is not None
        and prediction_confidence is not None
        and float(prediction_confidence) > 0.0
    )
    if not prediction_active:
        prediction_error = 0.0
    elif prediction_correct is True:
        prediction_error = 1.0 - clamp01(prediction_confidence or 0.0)
    else:
        prediction_error = max(0.5, clamp01(prediction_confidence or 0.5))

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
    if bool(level_completed_event):
        novelty_scores.append(_novelty(memory_counts.get("level_completed_event:true", 0)))
    if context_signature and bool(level_completed_event):
        novelty_scores.append(_novelty(memory_counts.get(f"context_level_completed:{context_signature}|true", 0)))
    learning_value = max(novelty_scores) if novelty_scores else 0.5
    if normalized_outcome_state in {"WIN", "GAME_OVER"}:
        terminal_outcome_learning_value = _novelty(memory_counts.get(f"outcome_state:{normalized_outcome_state}", 0))
        learning_value = max(learning_value, terminal_outcome_learning_value)
    if bool(level_completed_event):
        progress_learning_value = _novelty(memory_counts.get("level_completed_event:true", 0))
        learning_value = max(learning_value, progress_learning_value)

    # v6.3: transfer prior is structural/contextual reuse evidence available now.
    # Future-option magnitude and novelty are not transfer evidence.
    transfer_prior = 0.0
    transfer_active = False
    if actual_family_id:
        family_count = max(
            0,
            int(memory_counts.get(f"actual_family_id:{actual_family_id}", 0)),
        )
        local_count = 0
        if context_signature:
            local_count = max(
                0,
                int(
                    memory_counts.get(
                        f"context_family:{context_signature}|{actual_family_id}",
                        0,
                    )
                ),
            )
        if family_count > 0:
            cross_context_count = max(0, family_count - local_count)
            transfer_prior = clamp01(cross_context_count / family_count)
            transfer_active = True
    transfer_potential = transfer_prior

    if explanatory_delta is not None:
        explanatory_potential = clamp01(abs(explanatory_delta))
    else:
        explanatory_potential = 0.0
    if int(graph_counts.get("new_contingency", 0)) > 0:
        explanatory_potential = max(explanatory_potential, 0.5)
    if int(graph_counts.get("new_graph_edge", 0)) > 0:
        explanatory_potential = max(explanatory_potential, 0.35)

    component_active = {
        "survival_impact": True,
        "prediction_error": prediction_active,
        "learning_value": True,
        "transfer_potential": transfer_active,
        "explanatory_potential": bool(graph_counts) or explanatory_delta is not None,
    }
    total = _active_weighted_total(
        {
            "survival_impact": survival_impact,
            "prediction_error": prediction_error,
            "learning_value": learning_value,
            "transfer_potential": transfer_potential,
            "explanatory_potential": explanatory_potential,
        },
        normalized_weights,
        component_active,
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
        transfer_prior=transfer_prior,
        transfer_empirical_rate=None,
        component_active=component_active,
    )


def _active_weighted_total(
    values: Mapping[str, float],
    weights: Mapping[str, float],
    active: Mapping[str, bool],
) -> float:
    active_weight = sum(
        float(weights.get(key, 0.0))
        for key in values
        if bool(active.get(key, False)) and float(weights.get(key, 0.0)) > 0.0
    )
    if active_weight <= 0.0:
        return 0.0
    weighted = sum(
        clamp01(value) * float(weights.get(key, 0.0))
        for key, value in values.items()
        if bool(active.get(key, False)) and float(weights.get(key, 0.0)) > 0.0
    )
    return clamp01(weighted / active_weight)


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
