from __future__ import annotations

from dataclasses import dataclass

from v8.model import MemoryUid
from v8.planning import PlanSelection, Planner
from v8.strategies import StrategyEvidence


@dataclass(frozen=True, slots=True)
class ReplanningTrial:
    primary_strategy_uid: MemoryUid
    alternative_strategy_uid: MemoryUid
    outcome_uid: MemoryUid
    primary_invalidated: bool
    alternative_selected: bool
    outcome_preserved: bool
    recovery_succeeded: bool

    @property
    def valid_recovery(self) -> bool:
        return bool(
            self.primary_invalidated
            and self.alternative_selected
            and self.outcome_preserved
            and self.recovery_succeeded
            and self.primary_strategy_uid != self.alternative_strategy_uid
        )


class ReplanningController:
    def __init__(self) -> None:
        self.planner = Planner()
        self._trials: list[ReplanningTrial] = []

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

    def record_trial(
        self,
        *,
        primary_strategy_uid: MemoryUid,
        alternative_strategy_uid: MemoryUid,
        outcome_uid: MemoryUid,
        primary_invalidated: bool,
        alternative_selected: bool,
        outcome_preserved: bool,
        recovery_succeeded: bool,
    ) -> ReplanningTrial:
        trial = ReplanningTrial(
            primary_strategy_uid,
            alternative_strategy_uid,
            outcome_uid,
            bool(primary_invalidated),
            bool(alternative_selected),
            bool(outcome_preserved),
            bool(recovery_succeeded),
        )
        self._trials.append(trial)
        return trial

    def valid_trials(self) -> tuple[ReplanningTrial, ...]:
        return tuple(trial for trial in self._trials if trial.valid_recovery)
