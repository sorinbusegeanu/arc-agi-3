from __future__ import annotations

from dataclasses import dataclass

from v8.model import MemoryUid, stable_u64
from v8.strategies import StrategyEvidence


@dataclass(frozen=True, slots=True)
class PlanSelection:
    outcome_uid: MemoryUid
    strategy_uid: MemoryUid
    action_id: int
    score: float


class Planner:
    """Select an outcome and strategy independently so strategy failure can replan."""

    def select(
        self,
        *,
        context_signature: int,
        available_actions: tuple[int, ...],
        strategies: tuple[StrategyEvidence, ...],
        preferred_outcomes: tuple[MemoryUid, ...] = (),
        excluded_strategies: frozenset[MemoryUid] = frozenset(),
    ) -> PlanSelection | None:
        context_bucket = stable_u64(int(context_signature), person=b"v8-context")
        preferred = set(preferred_outcomes)
        candidates = []
        available = {int(action) for action in available_actions}
        for strategy in strategies:
            if strategy.uid in excluded_strategies:
                continue
            if int(strategy.action_id) not in available:
                continue
            if int(strategy.context_bucket) != int(context_bucket):
                continue
            preference_bonus = 0.25 if strategy.outcome_uid in preferred else 0.0
            efficiency = 1.0 / max(1e-9, float(strategy.mean_cost))
            score = float(strategy.reliability) + 0.10 * efficiency + preference_bonus
            candidates.append((score, strategy))
        if not candidates:
            return None
        score, strategy = max(candidates, key=lambda item: (item[0], item[1].attempts, -item[1].action_id))
        return PlanSelection(strategy.outcome_uid, strategy.uid, strategy.action_id, score)

    def replan(
        self,
        current: PlanSelection,
        *,
        context_signature: int,
        available_actions: tuple[int, ...],
        strategies: tuple[StrategyEvidence, ...],
    ) -> PlanSelection | None:
        alternatives = tuple(strategy for strategy in strategies if strategy.outcome_uid == current.outcome_uid)
        return self.select(
            context_signature=context_signature,
            available_actions=available_actions,
            strategies=alternatives,
            preferred_outcomes=(current.outcome_uid,),
            excluded_strategies=frozenset({current.strategy_uid}),
        )
