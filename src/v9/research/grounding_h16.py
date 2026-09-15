from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from random import Random\nimport json\nfrom pathlib import Path\nfrom dataclasses import asdict
from typing import Callable

from v9.environments.synthetic_symbolic import SyntheticSymbolicConfig, SyntheticSymbolicEnvironment

from .hypotheses import HypothesisResult, HypothesisStatus


class GroundingCondition(str, Enum):
    C0_INTERACTION_ONLY = "C0"
    C1_SYMBOLS_ONLY = "C1"
    C2_ALIGNED = "C2"
    C3_SHUFFLED = "C3"


@dataclass(frozen=True, slots=True)
class H16Metrics:
    interaction_prediction: float = 0.0
    action_success: float = 0.0
    symbol_conditioned_transfer: float = 0.0
    world_to_symbol_generalization: float = 0.0
    composition_success: float = 0.0
    persistence_without_symbols: float = 0.0
    grounding_calibration: float = 0.0

    @property
    def aggregate(self) -> float:
        values = (
            self.interaction_prediction,
            self.action_success,
            self.symbol_conditioned_transfer,
            self.world_to_symbol_generalization,
            self.composition_success,
            self.persistence_without_symbols,
        )
        return sum(values) / len(values)


@dataclass(frozen=True, slots=True)
class H16Trial:
    condition: GroundingCondition
    seed: int
    environment_config_id: int
    interaction_budget: int
    evaluation_id: int
    metrics: H16Metrics
    held_out_causal_effect: float
    trial_id: str = ""

    @property
    def prediction_score(self) -> float:
        return self.metrics.interaction_prediction

    @property
    def transfer_score(self) -> float:
        return self.metrics.symbol_conditioned_transfer


@dataclass(frozen=True, slots=True)
class H16Report:
    result: HypothesisResult
    means: dict[str, float]
    matched: bool
    metric_means: dict[str, dict[str, float]]
    causal_effect: float


def _mean(rows: tuple[H16Trial, ...], metric: str) -> float:
    if not rows:
        return 0.0
    return sum(float(getattr(row.metrics, metric)) for row in rows) / len(rows)


def evaluate_h16(
    trials: tuple[H16Trial, ...],
    *,
    advantage_threshold: float = 0.0,
    causal_threshold: float = 0.0,
) -> H16Report:
    by_condition = {
        condition: tuple(row for row in trials if row.condition is condition)
        for condition in GroundingCondition
    }
    if any(not rows for rows in by_condition.values()):
        return H16Report(
            HypothesisResult("H16", HypothesisStatus.UNTESTED, None, len(trials), "all C0-C3 conditions are required"),
            {},
            False,
            {},
            0.0,
        )

    keys = lambda rows: {
        (row.seed, row.environment_config_id, row.interaction_budget, row.evaluation_id)
        for row in rows
    }
    matched = len({frozenset(keys(rows)) for rows in by_condition.values()}) == 1

    metric_names = tuple(H16Metrics.__dataclass_fields__)
    metric_means = {
        condition.value: {metric: _mean(rows, metric) for metric in metric_names}
        for condition, rows in by_condition.items()
    }
    means = {
        condition.value: sum(row.metrics.aggregate for row in rows) / len(rows)
        for condition, rows in by_condition.items()
    }
    aligned = means[GroundingCondition.C2_ALIGNED.value]
    control = max(
        means[GroundingCondition.C0_INTERACTION_ONLY.value],
        means[GroundingCondition.C1_SYMBOLS_ONLY.value],
        means[GroundingCondition.C3_SHUFFLED.value],
    )
    causal = sum(
        row.held_out_causal_effect
        for row in by_condition[GroundingCondition.C2_ALIGNED]
    ) / len(by_condition[GroundingCondition.C2_ALIGNED])
    effect = aligned - control
    status = (
        HypothesisStatus.SUPPORTED
        if matched and effect > advantage_threshold and causal > causal_threshold
        else HypothesisStatus.FALSIFIED
    )
    return H16Report(
        HypothesisResult(
            "H16",
            status,
            effect,
            len(trials),
            "matched aligned advantage and held-out causal effect"
            if status is HypothesisStatus.SUPPORTED
            else "required matched advantage was not demonstrated",
        ),
        means,
        matched,
        metric_means,
        causal,
    )


def run_matched_controls(
    runner: Callable[[GroundingCondition, int, int], tuple[H16Metrics, float]],
    *,
    seeds: tuple[int, ...],
    environment_config_id: int,
    interaction_budget: int,
    evaluation_id: int = 1,
) -> tuple[H16Trial, ...]:
    rows: list[H16Trial] = []
    for seed in seeds:
        for condition in GroundingCondition:
            metrics, causal = runner(condition, seed, interaction_budget)
            rows.append(
                H16Trial(
                    condition=condition,
                    seed=seed,
                    environment_config_id=environment_config_id,
                    interaction_budget=interaction_budget,
                    evaluation_id=evaluation_id,
                    metrics=metrics,
                    held_out_causal_effect=causal,
                    trial_id=f"h16:{environment_config_id}:{evaluation_id}:{seed}:{condition.value}",
                )
            )
    return tuple(rows)


def _majority_model(rows: list[tuple[tuple[int, ...], int]]) -> tuple[dict[tuple[int, ...], int], int]:
    counts: dict[tuple[int, ...], dict[int, int]] = {}
    global_counts: dict[int, int] = {}
    for features, target in rows:
        bucket = counts.setdefault(features, {})
        bucket[target] = bucket.get(target, 0) + 1
        global_counts[target] = global_counts.get(target, 0) + 1
    choose = lambda values: min(values, key=lambda value: (-values[value], value))
    return {key: choose(values) for key, values in counts.items()}, choose(global_counts)


def run_synthetic_h16_controls(
    *,
    seeds: tuple[int, ...],
    environment_config_id: int,
    interaction_budget: int,
    evaluation_id: int = 1,
) -> tuple[H16Trial, ...]:
    """Execute deterministic arbitrary-symbol C0-C3 controls with matched budgets."""
    if interaction_budget < 6:
        raise ValueError("synthetic H16 requires at least six interactions")

    def runner(condition: GroundingCondition, seed: int, budget: int) -> tuple[H16Metrics, float]:
        environment = SyntheticSymbolicEnvironment(
            SyntheticSymbolicConfig(
                seed=seed,
                horizon=budget + 1,
                aligned=True,
                shuffled=condition is GroundingCondition.C3_SHUFFLED,
            )
        )
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

        ablated_model, ablated_fallback = _majority_model(
            [((row[1],), row[2]) for row in transitions[:split]]
        )
        ablated_correct = sum(
            ablated_model.get((row[1],), ablated_fallback) == row[2]
            for row in evaluation
        )
        persistence = ablated_correct / len(evaluation)
        causal = prediction - persistence if condition is GroundingCondition.C2_ALIGNED else 0.0

        symbol_only_model, symbol_only_fallback = _majority_model(
            [((row[0],), row[2]) for row in transitions[:split]]
        )
        symbol_only_correct = sum(
            symbol_only_model.get((row[0],), symbol_only_fallback) == row[2]
            for row in evaluation
        )
        symbol_generalization = symbol_only_correct / len(evaluation)

        metrics = H16Metrics(
            interaction_prediction=prediction,
            action_success=prediction,
            symbol_conditioned_transfer=prediction if condition in {GroundingCondition.C2_ALIGNED, GroundingCondition.C3_SHUFFLED} else 0.0,
            world_to_symbol_generalization=symbol_generalization if condition is not GroundingCondition.C0_INTERACTION_ONLY else 0.0,
            composition_success=prediction if condition is GroundingCondition.C2_ALIGNED else 0.0,
            persistence_without_symbols=persistence,
            grounding_calibration=max(0.0, 1.0 - abs(causal)),
        )
        return metrics, causal

    return run_matched_controls(
        runner,
        seeds=seeds,
        environment_config_id=environment_config_id,
        interaction_budget=interaction_budget,
        evaluation_id=evaluation_id,
    )


def publish_h16_report(runtime: object, report: H16Report) -> None:
    """Publish matched H16 evidence into the unified telemetry/dashboard path."""
    setter = getattr(runtime, "set_telemetry_gauge")
    for condition in GroundingCondition:
        setter(f"h16_{condition.value}_score", float(report.means.get(condition.value, 0.0)))
    setter("h16_aligned_advantage", float(report.result.effect or 0.0))
    setter("h16_heldout_causal_effect", float(report.causal_effect))
    setter("h16_matched", int(report.matched))
    aligned = report.metric_means.get(GroundingCondition.C2_ALIGNED.value, {})
    for metric, value in aligned.items():
        setter(f"h16_aligned_{metric}", float(value))


def save_h16_trials(path: str | Path, trials: tuple[H16Trial, ...]) -> None:
    payload = {
        "schema_version": 1,
        "trials": [
            {
                **asdict(row),
                "condition": row.condition.value,
            }
            for row in trials
        ],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def load_h16_trials(path: str | Path) -> tuple[H16Trial, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported H16 trial schema")
    rows = []
    for raw in payload.get("trials", []):
        metrics = H16Metrics(**dict(raw["metrics"]))
        rows.append(H16Trial(
            condition=GroundingCondition(str(raw["condition"])),
            seed=int(raw["seed"]),
            environment_config_id=int(raw["environment_config_id"]),
            interaction_budget=int(raw["interaction_budget"]),
            evaluation_id=int(raw["evaluation_id"]),
            metrics=metrics,
            held_out_causal_effect=float(raw["held_out_causal_effect"]),
            trial_id=str(raw.get("trial_id", "")),
        ))
    return tuple(rows)
