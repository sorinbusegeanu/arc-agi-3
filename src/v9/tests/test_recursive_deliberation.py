from __future__ import annotations

from v9.cognition.deliberation import (
    CandidateEvaluation,
    DeliberationBudget,
    DeliberationSignals,
    DeliberationStopReason,
    RecursiveDeliberator,
    adaptive_budget,
)
from v9.cognition.reasoning_workspace import ReasoningOperator, ReasoningWorkspace


def test_adaptive_budget_allocates_more_cycles_to_difficult_cases() -> None:
    easy = adaptive_budget(
        DeliberationSignals(candidate_confidence=1.0),
        minimum_cycles=1,
        maximum_cycles=6,
        compute_budget=64,
    )
    hard = adaptive_budget(
        DeliberationSignals(
            prediction_error=1.0,
            learning_value=1.0,
            transfer_potential=1.0,
            explanatory_potential=1.0,
            ambiguity=1.0,
            candidate_confidence=0.0,
            context_contradiction=1.0,
            alternative_count=8,
        ),
        minimum_cycles=1,
        maximum_cycles=6,
        compute_budget=64,
    )
    assert easy.maximum_cycles == 1
    assert hard.maximum_cycles == 6


def test_recursive_deliberation_keeps_best_candidate_from_earlier_cycle() -> None:
    def evaluator(candidate: int, workspace: ReasoningWorkspace) -> CandidateEvaluation:
        scores = {0: 0.0, 1: 2.0, 2: 1.0}
        return CandidateEvaluation(score=scores[candidate], ambiguity=0.5)

    def reasoner(candidate: int, workspace: ReasoningWorkspace):
        return (
            ReasoningOperator.COMPARE_STRATEGIES,
            workspace.advance(operator=ReasoningOperator.COMPARE_STRATEGIES),
        )

    def refiner(candidate: int, workspace: ReasoningWorkspace) -> int:
        return min(2, candidate + 1)

    result = RecursiveDeliberator(
        evaluator=evaluator,
        reasoner=reasoner,
        refiner=refiner,
        improvement_threshold=0.0,
        ambiguity_threshold=0.0,
        stability_cycles=2,
    ).run(
        0,
        budget=DeliberationBudget(1, 2, 8),
        mode="fixed",
    )

    assert result.final_candidate == 2
    assert result.best_candidate == 1
    assert result.best_cycle == 1
    assert result.improvement == 2.0


def test_recursive_deliberation_stops_when_candidate_is_stable() -> None:
    def evaluator(candidate: int, workspace: ReasoningWorkspace) -> CandidateEvaluation:
        return CandidateEvaluation(score=float(candidate), ambiguity=0.5)

    def reasoner(candidate: int, workspace: ReasoningWorkspace):
        return (
            ReasoningOperator.CHECK_TRAJECTORY_EFFICIENCY,
            workspace.advance(operator=ReasoningOperator.CHECK_TRAJECTORY_EFFICIENCY),
        )

    result = RecursiveDeliberator(
        evaluator=evaluator,
        reasoner=reasoner,
        refiner=lambda candidate, _workspace: candidate,
        improvement_threshold=0.0,
        ambiguity_threshold=0.0,
        stability_cycles=2,
    ).run(
        3,
        budget=DeliberationBudget(1, 6, 8),
        mode="adaptive",
    )

    assert result.stop_reason is DeliberationStopReason.CANDIDATE_STABLE
    assert len(result.cycles) == 3


def test_one_cycle_mode_is_reproducible() -> None:
    def evaluator(candidate: int, workspace: ReasoningWorkspace) -> CandidateEvaluation:
        return CandidateEvaluation(score=float(candidate), ambiguity=0.5)

    def reasoner(candidate: int, workspace: ReasoningWorkspace):
        return (
            ReasoningOperator.SIMULATE_CONSEQUENCE,
            workspace.advance(operator=ReasoningOperator.SIMULATE_CONSEQUENCE),
        )

    deliberator = RecursiveDeliberator(
        evaluator=evaluator,
        reasoner=reasoner,
        refiner=lambda candidate, _workspace: candidate + 1,
    )
    budget = DeliberationBudget(1, 6, 64)

    left = deliberator.run(1, budget=budget, mode="one_cycle")
    right = deliberator.run(1, budget=budget, mode="one_cycle")

    assert left == right
    assert left.stop_reason is DeliberationStopReason.ONE_CYCLE
    assert left.final_candidate == 2
