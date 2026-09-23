from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from random import Random

from v9.environments.synthetic_symbolic import SyntheticSymbolicConfig, SyntheticSymbolicEnvironment

from .hypotheses import HypothesisResult, HypothesisStatus


class GroundingCondition(str, Enum):
    C0_INTERACTION_ONLY = "C0"
    C1_SYMBOLS_ONLY = "C1"
    C2_ALIGNED = "C2"
    C3_SHUFFLED = "C3"


@dataclass(frozen=True, slots=True)
class H16Trial:
    condition: GroundingCondition
    seed: int
    environment_config_id: int
    interaction_budget: int
    evaluation_id: int
    prediction_score: float
    transfer_score: float
    held_out_causal_effect: float


@dataclass(frozen=True, slots=True)
class H16Report:
    result: HypothesisResult
    means: dict[str, float]
    matched: bool


def evaluate_h16(trials: tuple[H16Trial, ...], *, advantage_threshold: float = 0.0, causal_threshold: float = 0.0) -> H16Report:
    by_condition = {condition: tuple(row for row in trials if row.condition is condition) for condition in GroundingCondition}
    if any(not rows for rows in by_condition.values()):
        return H16Report(HypothesisResult("H16", HypothesisStatus.UNTESTED, None, len(trials), "all C0-C3 conditions are required"), {}, False)
    keys = lambda rows: {(row.seed, row.environment_config_id, row.interaction_budget, row.evaluation_id) for row in rows}
    matched = len({frozenset(keys(rows)) for rows in by_condition.values()}) == 1
    means = {condition.value: sum(row.prediction_score + row.transfer_score for row in rows) / (2 * len(rows)) for condition, rows in by_condition.items()}
    aligned = means[GroundingCondition.C2_ALIGNED.value]
    control = max(means[GroundingCondition.C0_INTERACTION_ONLY.value], means[GroundingCondition.C1_SYMBOLS_ONLY.value], means[GroundingCondition.C3_SHUFFLED.value])
    causal = sum(row.held_out_causal_effect for row in by_condition[GroundingCondition.C2_ALIGNED]) / len(by_condition[GroundingCondition.C2_ALIGNED])
    effect = aligned - control
    status = HypothesisStatus.SUPPORTED if matched and effect > advantage_threshold and causal > causal_threshold else HypothesisStatus.FALSIFIED
    return H16Report(HypothesisResult("H16", status, effect, len(trials), "matched aligned advantage and held-out causal effect" if status is HypothesisStatus.SUPPORTED else "required matched advantage was not demonstrated"), means, matched)


def run_matched_controls(runner: Callable[[GroundingCondition, int, int], tuple[float, float, float]], *, seeds: tuple[int, ...], environment_config_id: int, interaction_budget: int, evaluation_id: int = 1) -> tuple[H16Trial, ...]:
    return tuple(H16Trial(condition, seed, environment_config_id, interaction_budget, evaluation_id, *runner(condition, seed, interaction_budget)) for seed in seeds for condition in GroundingCondition)


def _majority_model(rows: list[tuple[tuple[int, ...], int]]) -> tuple[dict[tuple[int, ...], int], int]:
    counts: dict[tuple[int, ...], dict[int, int]] = {}
    global_counts: dict[int, int] = {}
    for features, target in rows:
        bucket = counts.setdefault(features, {})
        bucket[target] = bucket.get(target, 0) + 1
        global_counts[target] = global_counts.get(target, 0) + 1
    choose = lambda values: min(values, key=lambda value: (-values[value], value))
    return {key: choose(values) for key, values in counts.items()}, choose(global_counts)


def run_synthetic_h16_controls(*, seeds: tuple[int, ...], environment_config_id: int, interaction_budget: int, evaluation_id: int = 1) -> tuple[H16Trial, ...]:
    """Execute deterministic arbitrary-symbol C0-C3 controls.

    The held-out comparison uses the same recorded transitions with the aligned
    symbol channel enabled versus ablated. It records evidence; callers still
    decide the preregistered quality/dependency gates.
    """
    if interaction_budget < 6:
        raise ValueError("synthetic H16 requires at least six interactions")

    def runner(condition: GroundingCondition, seed: int, budget: int) -> tuple[float, float, float]:
        environment = SyntheticSymbolicEnvironment(SyntheticSymbolicConfig(seed=seed, horizon=budget + 1, aligned=True, shuffled=condition is GroundingCondition.C3_SHUFFLED))
        rng = Random(seed)
        transitions: list[tuple[int, int, int]] = []
        for _ in range(budget):
            symbol = int(environment.optional_symbol_stream()[0])
            action = rng.randrange(2)
            target = int(environment.step(action))
            transitions.append((symbol, action, target))
        split = max(2, (2 * len(transitions)) // 3)

        def features(row: tuple[int, int, int], selected: GroundingCondition) -> tuple[int, ...]:
            symbol, action, _target = row
            if selected is GroundingCondition.C0_INTERACTION_ONLY:
                return (action,)
            if selected is GroundingCondition.C1_SYMBOLS_ONLY:
                return (symbol,)
            return symbol, action

        training = [(features(row, condition), row[2]) for row in transitions[:split]]
        model, fallback = _majority_model(training)
        evaluation = transitions[split:]
        correct = sum(model.get(features(row, condition), fallback) == row[2] for row in evaluation)
        prediction = correct / len(evaluation)
        transfer = prediction
        causal = 0.0
        if condition is GroundingCondition.C2_ALIGNED:
            ablated_model, ablated_fallback = _majority_model([((row[1],), row[2]) for row in transitions[:split]])
            ablated_correct = sum(ablated_model.get((row[1],), ablated_fallback) == row[2] for row in evaluation)
            causal = prediction - ablated_correct / len(evaluation)
        return prediction, transfer, causal

    return run_matched_controls(runner, seeds=seeds, environment_config_id=environment_config_id, interaction_budget=interaction_budget, evaluation_id=evaluation_id)
