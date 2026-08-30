from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from statistics import mean
from typing import Iterable

from v8.environments.synthetic_symbolic import SyntheticSymbolicConfig, SyntheticSymbolicEnvironment


class GroundingCondition(str, Enum):
    C0_INTERACTION_ONLY = "C0"
    C1_SYMBOLS_ONLY = "C1"
    C2_ALIGNED = "C2"
    C3_SHUFFLED = "C3"


@dataclass(frozen=True, slots=True)
class GroundingTrial:
    condition: GroundingCondition
    seed: int
    mechanic: str
    steps: int
    positive_boundary: bool
    symbol_observations: int
    world_observations: int


@dataclass(frozen=True, slots=True)
class H16Metrics:
    condition: GroundingCondition
    trials: int
    prediction_improvement: float = 0.0
    action_selection_improvement: float = 0.0
    held_out_transfer: float = 0.0
    false_transfer: float = 0.0
    time_to_g4: float = 0.0
    time_to_g5: float = 0.0
    cross_modal_memory_count: int = 0
    memory_growth: int = 0
    candidate_search_compute: int = 0
    negative_grounding_evidence: int = 0


@dataclass(frozen=True, slots=True)
class H16Decision:
    interpretable: bool
    c2_separates_controls: bool
    reason: str


class GroundingControlRunner:
    @staticmethod
    def config_for(condition: GroundingCondition, *, seed: int, mechanic: str = "advance") -> SyntheticSymbolicConfig:
        symbol_condition = {GroundingCondition.C0_INTERACTION_ONLY: "none", GroundingCondition.C1_SYMBOLS_ONLY: "symbol_only", GroundingCondition.C2_ALIGNED: "aligned", GroundingCondition.C3_SHUFFLED: "shuffled"}[condition]
        return SyntheticSymbolicConfig(mechanic=mechanic, symbol_condition=symbol_condition, seed=int(seed))

    def run_trial(self, condition: GroundingCondition, *, seed: int, mechanic: str = "advance", actions: tuple[int, ...] = (0, 0, 0, 0, 0, 0)) -> GroundingTrial:
        env = SyntheticSymbolicEnvironment(self.config_for(condition, seed=seed, mechanic=mechanic)); world_count = 0 if condition is GroundingCondition.C1_SYMBOLS_ONLY else 1; symbol_count = len(env.passive_symbol_tokens()); steps = 0
        for action in actions:
            env.step(action); steps += 1
            if condition is not GroundingCondition.C1_SYMBOLS_ONLY: world_count += 1
            symbol_count += len(env.passive_symbol_tokens())
            if env.cognitive_boundary_event().crossed: break
        return GroundingTrial(condition, int(seed), mechanic, steps, env.cognitive_boundary_event().positive, symbol_count, world_count)

    def matched_trials(self, seeds: Iterable[int], *, mechanic: str = "advance") -> dict[GroundingCondition, tuple[GroundingTrial, ...]]:
        seed_rows = tuple(int(seed) for seed in seeds)
        return {condition: tuple(self.run_trial(condition, seed=seed, mechanic=mechanic) for seed in seed_rows) for condition in GroundingCondition}


class H16Evaluator:
    REQUIRED_CONDITIONS = frozenset(GroundingCondition)

    def evaluate(self, metrics: Iterable[H16Metrics]) -> H16Decision:
        rows = {row.condition: row for row in metrics}
        if set(rows) != set(self.REQUIRED_CONDITIONS): return H16Decision(False, False, "all C0-C3 conditions are required")
        if any(int(row.trials) <= 0 for row in rows.values()): return H16Decision(False, False, "each condition requires held-out trials")
        c2 = rows[GroundingCondition.C2_ALIGNED]; controls = [rows[GroundingCondition.C0_INTERACTION_ONLY], rows[GroundingCondition.C1_SYMBOLS_ONLY], rows[GroundingCondition.C3_SHUFFLED]]
        c2_effect = mean((c2.prediction_improvement, c2.action_selection_improvement, c2.held_out_transfer)); control_effect = max(mean((row.prediction_improvement, row.action_selection_improvement, row.held_out_transfer)) for row in controls)
        false_transfer_ok = c2.false_transfer <= max(row.false_transfer for row in controls); separates = bool(c2_effect > control_effect and false_transfer_ok)
        return H16Decision(True, separates, "C2 separates controls" if separates else "C2 does not separate C0/C1/C3 on interaction-relevant held-out measures")

    @staticmethod
    def report(metrics: Iterable[H16Metrics]) -> list[dict[str, object]]:
        return [{**asdict(row), "condition": row.condition.value} for row in metrics]
