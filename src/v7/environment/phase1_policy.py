from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import Iterable

from v7.environment.cognition import (
    ContextualActionDecision,
    ContextualActionScorer,
    DecisionContext,
    LocalCognitionOverlay,
)
from v7.memory.ids import MemoryId
from v7.memory.planning import PersistentPlanningGraph, StrategyProcedure
from v7.memory.read_view import MemoryReadView


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True, slots=True)
class Phase1ActionDecision:
    action_id: int
    score: float
    support: object
    exploration_score: float
    failure_risk: float
    contradiction_risk: float
    future_reachability: float
    memory_confidence: float
    persistent_reachability: float
    persistent_success_reachability: float
    persistent_failure_risk: float
    dead_end_risk: float
    option_loss_risk: float


@dataclass(frozen=True, slots=True)
class Phase1Selection:
    decision: Phase1ActionDecision
    mode: str
    strategy_id: int | None = None
    effective_epsilon: float = 0.0


class Phase1ActionScorer:
    """Augment contextual v7 scores with persistent planning evidence.

    Exploration remains a separate signal. The returned score is an
    exploitation/memory score rather than exploration already mixed into the
    same scalar.
    """

    def __init__(self, planning: PersistentPlanningGraph) -> None:
        self.planning = planning
        self.base = ContextualActionScorer()

    def score_actions(
        self,
        *,
        view: MemoryReadView,
        contexts: DecisionContext,
        actions: Iterable[int],
        overlay: LocalCognitionOverlay,
    ) -> tuple[Phase1ActionDecision, ...]:
        base_rows = self.base.score_actions(
            view=view,
            contexts=contexts,
            actions=actions,
            overlay=overlay,
        )
        output: list[Phase1ActionDecision] = []
        for row in base_rows:
            planning_input = view.score_inputs(
                context_signature=contexts.planning_signature,
                action_ids=(row.action_id,),
            )[0]
            signal = self.planning.evaluate(
                view,
                planning_input.contingency_ids,
                depth=3,
                max_nodes=64,
            )

            # The base scorer includes +0.08*exploration. Remove that term so
            # exploration and exploitation can be arbitrated explicitly.
            memory_score = float(row.score) - 0.08 * float(row.exploration_score)
            memory_score += 0.12 * signal.success_reachability
            memory_score += 0.08 * signal.reachability
            memory_score -= 0.20 * signal.failure_risk
            memory_score -= 0.12 * signal.stall_risk
            memory_score -= 0.10 * signal.option_loss_risk

            failure = max(float(row.failure_risk), signal.failure_risk)
            reachability = max(
                float(row.future_reachability),
                signal.reachability,
                signal.success_reachability,
            )
            support = int(getattr(row.support, "contextual_support", 0) or 0)
            local_support = int(getattr(row.support, "local_support", 0) or 0)
            support_confidence = 1.0 - math.exp(-max(0, support + local_support) / 4.0)
            semantic_confidence = max(
                _clamp01(reachability),
                _clamp01(failure),
                _clamp01(row.contradiction_risk),
                _clamp01(signal.success_reachability),
            )
            confidence = _clamp01(
                0.65 * support_confidence + 0.35 * semantic_confidence
            )

            output.append(
                Phase1ActionDecision(
                    action_id=int(row.action_id),
                    score=float(memory_score),
                    support=row.support,
                    exploration_score=float(row.exploration_score),
                    failure_risk=failure,
                    contradiction_risk=float(row.contradiction_risk),
                    future_reachability=reachability,
                    memory_confidence=confidence,
                    persistent_reachability=signal.reachability,
                    persistent_success_reachability=signal.success_reachability,
                    persistent_failure_risk=signal.failure_risk,
                    dead_end_risk=signal.stall_risk,
                    option_loss_risk=signal.option_loss_risk,
                )
            )
        return tuple(sorted(output, key=lambda item: item.action_id))


class StrategyExecutionCursor:
    """Execute a validated successful M6 procedure step-by-step until it diverges."""

    def __init__(
        self,
        *,
        start_confidence: float = 0.60,
        continue_confidence: float = 0.50,
        risk_abort_threshold: float = 0.65,
    ) -> None:
        self.start_confidence = float(start_confidence)
        self.continue_confidence = float(continue_confidence)
        self.risk_abort_threshold = float(risk_abort_threshold)
        self.strategy_id: MemoryId | None = None
        self.position = 0

    @property
    def active(self) -> bool:
        return self.strategy_id is not None

    def reset(self) -> None:
        self.strategy_id = None
        self.position = 0

    def recommend(
        self,
        *,
        view: MemoryReadView,
        planning: PersistentPlanningGraph,
        contexts: DecisionContext,
        decisions: Iterable[Phase1ActionDecision],
    ) -> tuple[Phase1ActionDecision, MemoryId] | None:
        rows = {int(row.action_id): row for row in decisions}
        current_context = int(contexts.planning_signature)

        if self.strategy_id is not None:
            procedure = planning.strategies.get(self.strategy_id)
            if procedure is None or self.position >= len(procedure.steps):
                self.reset()
            else:
                step = procedure.steps[self.position]
                decision = rows.get(int(step.action_id))
                confidence = self._strategy_confidence(view, self.strategy_id)
                if (
                    int(step.context_signature) == current_context
                    and decision is not None
                    and confidence >= self.continue_confidence
                    and max(decision.failure_risk, decision.dead_end_risk)
                    < self.risk_abort_threshold
                ):
                    return decision, self.strategy_id
                self.reset()

        best: tuple[float, int, MemoryId, Phase1ActionDecision] | None = None
        for decision in rows.values():
            for raw_id in getattr(decision.support, "strategy_ids", ()) or ():
                strategy_id = MemoryId(int(raw_id))
                procedure = planning.strategies.get(strategy_id)
                if procedure is None or not procedure.steps:
                    continue
                first = procedure.steps[0]
                if (
                    int(first.context_signature) != current_context
                    or int(first.action_id) != int(decision.action_id)
                ):
                    continue
                confidence = self._strategy_confidence(view, strategy_id)
                if confidence < self.start_confidence:
                    continue
                if max(decision.failure_risk, decision.dead_end_risk) >= self.risk_abort_threshold:
                    continue
                candidate = (confidence, -len(procedure.steps), strategy_id, decision)
                if best is None or candidate[:2] > best[:2] or (
                    candidate[:2] == best[:2] and int(strategy_id) < int(best[2])
                ):
                    best = candidate

        if best is None:
            return None
        self.strategy_id = best[2]
        self.position = 0
        return best[3], best[2]

    def observe_outcome(
        self,
        *,
        selected_strategy_id: int | None,
        terminal_polarity: int,
        next_context_signature: int | None,
    ) -> None:
        if self.strategy_id is None:
            return
        if selected_strategy_id is None or int(selected_strategy_id) != int(self.strategy_id):
            self.reset()
            return
        if int(terminal_polarity) != 0:
            self.reset()
            return
        self.position += 1
        # Context mismatch is checked before executing the next step. We can
        # reject immediately when the expected context is already known.
        # The procedure itself is looked up on the next recommend() call.
        if next_context_signature is None:
            self.reset()

    @staticmethod
    def _strategy_confidence(view: MemoryReadView, strategy_id: MemoryId) -> float:
        node = view.nodes.get(strategy_id)
        if node is None:
            return 0.0
        support = 1.0 - math.exp(-max(0, int(node.support_count)) / 2.0)
        score = view.scores.get(strategy_id)
        semantic = 0.0
        if score is not None:
            semantic = max(
                _clamp01(score.significance),
                _clamp01(score.learning_value),
                _clamp01(max(0.0, score.future_option_delta)),
            )
        return _clamp01(0.5 * support + 0.5 * semantic)


def select_phase1_action(
    *,
    view: MemoryReadView,
    planning: PersistentPlanningGraph,
    cursor: StrategyExecutionCursor,
    contexts: DecisionContext,
    decisions: Iterable[Phase1ActionDecision],
    rng: Random,
    epsilon: float,
) -> Phase1Selection:
    rows = tuple(decisions)
    if not rows:
        raise ValueError("environment returned no available actions")

    strategy = cursor.recommend(
        view=view,
        planning=planning,
        contexts=contexts,
        decisions=rows,
    )
    if strategy is not None:
        decision, strategy_id = strategy
        return Phase1Selection(
            decision=decision,
            mode="strategy",
            strategy_id=int(strategy_id),
            effective_epsilon=0.0,
        )

    memory = min(rows, key=lambda item: (-item.score, item.action_id))
    exploration = min(
        rows, key=lambda item: (-item.exploration_score, item.action_id)
    )
    confidence = _clamp01(memory.memory_confidence)
    effective_epsilon = _clamp01(float(epsilon)) * (1.0 - confidence) ** 2

    if confidence >= 0.78:
        selected = memory
        mode = "memory"
    elif effective_epsilon > 0.0 and rng.random() < effective_epsilon:
        selected = exploration
        mode = "exploration"
    elif exploration.exploration_score >= 0.75 and confidence < 0.35:
        selected = exploration
        mode = "exploration"
    else:
        temperature = 0.08 + 0.32 * (1.0 - confidence)
        maximum = max(float(item.score) for item in rows)
        weights = [
            math.exp((float(item.score) - maximum) / max(1e-6, temperature))
            for item in rows
        ]
        total = sum(weights)
        if not math.isfinite(total) or total <= 0.0:
            selected = memory
        else:
            selected = rng.choices(list(rows), weights=weights, k=1)[0]
        mode = "memory" if selected.action_id == memory.action_id else "stochastic"

    # If the chosen memory action is the first step of a sufficiently strong
    # stored procedure, start the cursor now so subsequent steps are executed
    # procedurally instead of independently rescored.
    if mode != "exploration":
        started = cursor.recommend(
            view=view,
            planning=planning,
            contexts=contexts,
            decisions=(selected,),
        )
        if started is not None and started[0].action_id == selected.action_id:
            selected, strategy_id = started
            return Phase1Selection(
                decision=selected,
                mode="strategy",
                strategy_id=int(strategy_id),
                effective_epsilon=effective_epsilon,
            )

    return Phase1Selection(
        decision=selected,
        mode=mode,
        strategy_id=None,
        effective_epsilon=effective_epsilon,
    )
