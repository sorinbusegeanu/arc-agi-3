from __future__ import annotations

from dataclasses import dataclass

from .contracts import PredictionDirection, PredictionOutcome
from .models import ExperimentPrediction, MetricResult


@dataclass(frozen=True)
class PredictionEvaluation:
    prediction: ExperimentPrediction
    result: MetricResult
    outcome: PredictionOutcome
    reason: str


def evaluate_prediction(
    prediction: ExperimentPrediction,
    result: MetricResult,
    *,
    min_samples: int = 2,
) -> PredictionEvaluation:
    if prediction.metric != result.metric:
        raise ValueError("prediction metric does not match result metric")
    if result.sample_count < min_samples:
        return PredictionEvaluation(
            prediction,
            result,
            PredictionOutcome.INCONCLUSIVE,
            f"sample_count={result.sample_count} < {min_samples}",
        )

    effect = result.effect
    threshold = abs(float(prediction.expected_min_effect))

    if prediction.direction == PredictionDirection.UP:
        confirmed = effect >= threshold
        contradicted = effect <= -threshold if threshold > 0 else effect < 0
    elif prediction.direction == PredictionDirection.DOWN:
        confirmed = effect <= -threshold
        contradicted = effect >= threshold if threshold > 0 else effect > 0
    else:
        tolerance = max(abs(float(prediction.flat_tolerance)), threshold)
        confirmed = abs(effect) <= tolerance
        contradicted = abs(effect) > tolerance

    if confirmed:
        outcome = PredictionOutcome.CONFIRMED
    elif contradicted:
        outcome = PredictionOutcome.CONTRADICTED
    else:
        outcome = PredictionOutcome.INCONCLUSIVE

    return PredictionEvaluation(prediction, result, outcome, f"effect={effect:.6g}")
