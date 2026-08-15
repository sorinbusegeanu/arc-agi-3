from __future__ import annotations

from v8.planning import PlanSelection, Planner
from v8.strategies import StrategyEvidence


class ReplanningController:
    def __init__(self) -> None:
        self.planner = Planner()

    def replan_same_outcome(
        self,
        current: PlanSelection,
        *,
        context_signature: int,
        available_actions: tuple[int, ...],
        strategies: tuple[StrategyEvidence, ...],
    ) -> PlanSelection | None:
        return self.planner.replan(
            current,
            context_signature=context_signature,
            available_actions=available_actions,
            strategies=strategies,
        )
