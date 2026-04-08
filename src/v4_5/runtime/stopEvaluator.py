from __future__ import annotations

from dataclasses import dataclass

from v4_5.runtime.types import ExecutedPrefixResult, LiveObservationSnapshot


@dataclass(frozen=True)
class StopEvaluation:
    should_stop: bool
    stop_reason: str
    level_transition: bool = False
    game_completion: bool = False
    terminal_success: bool = False
    terminal_failure: bool = False
    no_decision_dead_end: bool = False


class StopEvaluator:
    def evaluate(
        self,
        *,
        snapshot: LiveObservationSnapshot,
        executed_prefix_result: ExecutedPrefixResult | None,
        steps_executed: int,
        max_steps: int | None,
        level_steps_executed: int = 0,
        max_actions_per_level: int | None = None,
        max_levels: int | None = None,
        no_decision_rounds: int = 0,
        no_decision_threshold: int = 2,
    ) -> StopEvaluation:
        terminal_status = snapshot.terminal_status
        if terminal_status == "success":
            return StopEvaluation(True, "terminal_win", terminal_success=True, game_completion=True)
        if terminal_status == "failure":
            return StopEvaluation(True, "terminal_fail", terminal_failure=True)
        if max_levels is not None and snapshot.levels_completed >= int(max_levels):
            return StopEvaluation(True, "configured_level_limit_reached", game_completion=True)
        if max_steps is not None and int(steps_executed) >= int(max_steps):
            return StopEvaluation(True, "step_budget_exhausted")
        if max_actions_per_level is not None and int(level_steps_executed) >= int(max_actions_per_level):
            return StopEvaluation(True, "level_action_budget_exhausted")
        if no_decision_rounds >= int(no_decision_threshold):
            return StopEvaluation(True, "no_decision_dead_end", no_decision_dead_end=True)
        if executed_prefix_result is not None and executed_prefix_result.game_completion:
            return StopEvaluation(True, "terminal_win", terminal_success=True, game_completion=True)
        if executed_prefix_result is not None and executed_prefix_result.level_transition:
            return StopEvaluation(False, "continue", level_transition=True)
        return StopEvaluation(False, "continue")
