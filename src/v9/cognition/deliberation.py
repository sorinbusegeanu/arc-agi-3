from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, TypeVar

from v9.cognition.reasoning_workspace import ReasoningOperator, ReasoningWorkspace


CandidateT = TypeVar("CandidateT")


class DeliberationStopReason(str, Enum):
    OFF = "OFF"
    ONE_CYCLE = "ONE_CYCLE"
    CANDIDATE_STABLE = "CANDIDATE_STABLE"
    IMPROVEMENT_BELOW_THRESHOLD = "IMPROVEMENT_BELOW_THRESHOLD"
    AMBIGUITY_RESOLVED = "AMBIGUITY_RESOLVED"
    COMPUTE_BUDGET_EXHAUSTED = "COMPUTE_BUDGET_EXHAUSTED"
    MAX_CYCLES = "MAX_CYCLES"


@dataclass(frozen=True, slots=True)
class DeliberationSignals:
    prediction_error: float = 0.0
    learning_value: float = 0.0
    transfer_potential: float = 0.0
    explanatory_potential: float = 0.0
    ambiguity: float = 0.0
    candidate_confidence: float = 1.0
    context_contradiction: float = 0.0
    alternative_count: int = 0


@dataclass(frozen=True, slots=True)
class DeliberationBudget:
    minimum_cycles: int
    maximum_cycles: int
    compute_budget: int


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    score: float
    ambiguity: float
    prediction_consistency: float = 0.0
    contradiction_penalty: float = 0.0
    future_option_support: float = 0.0
    strategy_reliability: float = 0.0
    trajectory_efficiency: float = 0.0


@dataclass(frozen=True, slots=True)
class DeliberationCycle(Generic[CandidateT]):
    cycle: int
    candidate: CandidateT
    evaluation: CandidateEvaluation
    operator: ReasoningOperator


@dataclass(frozen=True, slots=True)
class DeliberationResult(Generic[CandidateT]):
    initial_candidate: CandidateT
    best_candidate: CandidateT
    final_candidate: CandidateT
    best_cycle: int
    cycles: tuple[DeliberationCycle[CandidateT], ...]
    stop_reason: DeliberationStopReason
    workspace: ReasoningWorkspace

    @property
    def improvement(self) -> float:
        if not self.cycles:
            return 0.0
        initial = self.cycles[0].evaluation.score
        best = max(row.evaluation.score for row in self.cycles)
        return float(best - initial)


def adaptive_budget(
    signals: DeliberationSignals,
    *,
    minimum_cycles: int,
    maximum_cycles: int,
    compute_budget: int,
) -> DeliberationBudget:
    if minimum_cycles <= 0 or maximum_cycles < minimum_cycles or compute_budget <= 0:
        raise ValueError("invalid deliberation budget bounds")
    difficulty = (
        max(0.0, float(signals.prediction_error))
        + max(0.0, float(signals.learning_value))
        + max(0.0, float(signals.transfer_potential))
        + max(0.0, float(signals.explanatory_potential))
        + max(0.0, float(signals.ambiguity))
        + max(0.0, 1.0 - float(signals.candidate_confidence))
        + max(0.0, float(signals.context_contradiction))
        + min(1.0, max(0, int(signals.alternative_count)) / 8.0)
    )
    normalized = min(1.0, difficulty / 4.0)
    span = maximum_cycles - minimum_cycles
    cycles = minimum_cycles + int(round(normalized * span))
    return DeliberationBudget(minimum_cycles, max(minimum_cycles, min(maximum_cycles, cycles)), compute_budget)


class RecursiveDeliberator(Generic[CandidateT]):
    def __init__(
        self,
        *,
        evaluator: Callable[[CandidateT, ReasoningWorkspace], CandidateEvaluation],
        reasoner: Callable[[CandidateT, ReasoningWorkspace], tuple[ReasoningOperator, ReasoningWorkspace]],
        refiner: Callable[[CandidateT, ReasoningWorkspace], CandidateT],
        improvement_threshold: float = 0.001,
        ambiguity_threshold: float = 0.10,
        stability_cycles: int = 2,
    ) -> None:
        if improvement_threshold < 0 or ambiguity_threshold < 0 or stability_cycles <= 0:
            raise ValueError("invalid deliberation thresholds")
        self.evaluator = evaluator
        self.reasoner = reasoner
        self.refiner = refiner
        self.improvement_threshold = float(improvement_threshold)
        self.ambiguity_threshold = float(ambiguity_threshold)
        self.stability_cycles = int(stability_cycles)

    def run(
        self,
        initial_candidate: CandidateT,
        *,
        workspace: ReasoningWorkspace | None = None,
        budget: DeliberationBudget,
        mode: str = "adaptive",
    ) -> DeliberationResult[CandidateT]:
        if mode not in {"off", "one_cycle", "fixed", "adaptive"}:
            raise ValueError("unsupported deliberation mode")

        state = workspace or ReasoningWorkspace()
        candidate = initial_candidate
        initial_eval = self.evaluator(candidate, state)
        cycles: list[DeliberationCycle[CandidateT]] = [
            DeliberationCycle(0, candidate, initial_eval, ReasoningOperator.RETRIEVE_SUPPORT)
        ]
        best_candidate = candidate
        best_evaluation = initial_eval
        best_cycle = 0

        if mode == "off":
            return DeliberationResult(
                initial_candidate,
                best_candidate,
                candidate,
                best_cycle,
                tuple(cycles),
                DeliberationStopReason.OFF,
                state,
            )

        maximum_cycles = 1 if mode == "one_cycle" else int(budget.maximum_cycles)
        stable = 0
        compute_used = 0
        stop_reason = DeliberationStopReason.MAX_CYCLES

        for cycle_index in range(1, maximum_cycles + 1):
            operator, next_state = self.reasoner(candidate, state)
            compute_used += 1
            candidate_next = self.refiner(candidate, next_state)
            evaluation = self.evaluator(candidate_next, next_state)
            cycles.append(DeliberationCycle(cycle_index, candidate_next, evaluation, operator))

            delta = evaluation.score - cycles[-2].evaluation.score
            if evaluation.score > best_evaluation.score:
                best_candidate = candidate_next
                best_evaluation = evaluation
                best_cycle = cycle_index

            stable = stable + 1 if candidate_next == candidate else 0
            candidate = candidate_next
            state = next_state

            if mode == "one_cycle":
                stop_reason = DeliberationStopReason.ONE_CYCLE
                break
            if compute_used >= budget.compute_budget:
                stop_reason = DeliberationStopReason.COMPUTE_BUDGET_EXHAUSTED
                break
            if cycle_index >= budget.minimum_cycles:
                if evaluation.ambiguity <= self.ambiguity_threshold:
                    stop_reason = DeliberationStopReason.AMBIGUITY_RESOLVED
                    break
                if stable >= self.stability_cycles:
                    stop_reason = DeliberationStopReason.CANDIDATE_STABLE
                    break
                if delta >= 0.0 and delta < self.improvement_threshold:
                    stop_reason = DeliberationStopReason.IMPROVEMENT_BELOW_THRESHOLD
                    break

        return DeliberationResult(
            initial_candidate,
            best_candidate,
            candidate,
            best_cycle,
            tuple(cycles),
            stop_reason,
            state,
        )
