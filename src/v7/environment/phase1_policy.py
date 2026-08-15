from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import Iterable

from v7.environment.ablation import CognitionAblation, ablated
from v7.environment.cognition import (
    ContextualActionDecision,
    ContextualActionScorer,
    DecisionContext,
    LocalCognitionOverlay,
)
from v7.memory.concept_validation import ConceptValidationStatus
from v7.memory.developmental_policy import DevelopmentStage, profile_for_view
from v7.memory.ids import MemoryId
from v7.memory.planning import PersistentPlanningGraph
from v7.memory.read_view import MemoryReadView
from v7.memory.semantic_relations import TYPE_RELATIONAL_WORLD_MODEL
from v7.memory.status import memory_is_active


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _local_policy_confidence(
    confidence: float,
    *,
    signature_count: int,
    context_rank: int,
    local_support: int,
) -> float:
    """Cap shared-memory confidence until the current lane confirms it locally."""
    value = _clamp01(confidence)
    if int(signature_count) < 5 or int(local_support) >= 2:
        return value
    specificity = _clamp01(int(context_rank) / 4.0)
    if int(local_support) <= 0:
        cap = 0.55 + 0.10 * specificity
    else:
        cap = 0.65 + 0.10 * specificity
    return min(value, cap)


def _is_transfer_frontier(
    contexts: DecisionContext,
    decision: object,
) -> bool:
    """Return whether a decision is broad or not yet locally confirmed."""
    if len(contexts.signatures) < 5:
        return False
    support = getattr(decision, "support", None)
    context_rank = int(getattr(support, "context_rank", 0) or 0)
    local_support = int(getattr(support, "local_support", 0) or 0)
    # The confidence cap considers two local observations sufficient. Strategy
    # start must use the same threshold rather than becoming deterministic at 1.
    return context_rank < 3 or local_support < 2


def _transfer_probe_strength(
    view: MemoryReadView,
    concept_ids: Iterable[int],
) -> float:
    """Bounded priority for testing transferable active concepts."""
    best = 0.0
    for raw_memory_id in concept_ids:
        memory_id = MemoryId(int(raw_memory_id))
        node = view.nodes.get(memory_id)
        if not memory_is_active(node):
            continue
        assert node is not None
        flags = int(node.status_flags)
        if flags & int(ConceptValidationStatus.TRANSFER_REJECTED):
            continue
        score = view.scores.get(memory_id)
        if score is None:
            continue
        semantic = _clamp01(
            max(
                float(score.transfer_prior),
                float(score.learning_value),
                float(score.explanatory_potential),
            )
        )
        if flags & int(
            ConceptValidationStatus.TRUSTED
            | ConceptValidationStatus.TRANSFER_VALIDATED
        ):
            status_weight = 1.0
        elif flags & int(ConceptValidationStatus.TRANSFER_CANDIDATE):
            status_weight = 0.90
        elif flags & int(ConceptValidationStatus.STRUCTURAL_SUPPORTED):
            status_weight = 0.75
        elif flags & int(ConceptValidationStatus.CANDIDATE):
            status_weight = 0.50
        else:
            status_weight = 0.25
        best = max(best, status_weight * semantic)
    return _clamp01(best)


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
    future_option_score_component: float = 0.0
    development_stage: str = "CONTROL"


@dataclass(frozen=True, slots=True)
class Phase1Selection:
    decision: Phase1ActionDecision
    mode: str
    strategy_id: int | None = None
    effective_epsilon: float = 0.0
    development_stage: str = "CONTROL"


class Phase1ActionScorer:
    """Contextual exploitation score augmented by persistent planning."""

    def __init__(
        self,
        planning: PersistentPlanningGraph,
        *,
        ablation_mask: int = 0,
    ) -> None:
        self.planning = planning
        self.ablation_mask = int(ablation_mask)
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
        profile = profile_for_view(view)
        stage_name = profile.stage.name
        planning_depth = (
            3
            if ablated(
                self.ablation_mask,
                CognitionAblation.DEVELOPMENTAL_POLICY,
            )
            else int(profile.planning_depth)
        )
        output: list[Phase1ActionDecision] = []
        for row in base_rows:
            signal = self._planning_signal(
                view=view,
                contexts=contexts,
                row=row,
                depth=planning_depth,
            )

            memory_score = float(row.score) - 0.08 * float(row.exploration_score)
            future_option_component = float(
                getattr(row, "future_option_score_component", 0.0)
            )
            future_option_ablated = ablated(
                self.ablation_mask,
                CognitionAblation.FUTURE_OPTION,
            )
            if future_option_ablated:
                memory_score -= future_option_component

            if ablated(
                self.ablation_mask,
                CognitionAblation.FUNCTIONAL_ROLES,
            ):
                role_strength = self.base._memory_strength(  # noqa: SLF001
                    view,
                    getattr(row.support, "role_ids", ()) or (),
                )
                memory_score -= 0.08 * role_strength

            if ablated(
                self.ablation_mask,
                CognitionAblation.RELATIONAL_WORLD_MODELS,
            ):
                all_worlds = tuple(
                    int(value)
                    for value in getattr(row.support, "world_model_ids", ()) or ()
                )
                transition_worlds = tuple(
                    memory_id
                    for memory_id in all_worlds
                    if (
                        memory_is_active(view.nodes.get(MemoryId(memory_id)))
                        and int(view.nodes[MemoryId(memory_id)].type_id)
                        != TYPE_RELATIONAL_WORLD_MODEL
                    )
                )
                all_strength = self.base._memory_strength(  # noqa: SLF001
                    view,
                    all_worlds,
                )
                transition_strength = self.base._memory_strength(  # noqa: SLF001
                    view,
                    transition_worlds,
                )
                memory_score -= 0.10 * max(
                    0.0,
                    all_strength - transition_strength,
                )

            if not ablated(
                self.ablation_mask,
                CognitionAblation.PERSISTENT_PLANNING,
            ):
                memory_score += 0.12 * signal.success_reachability
                memory_score += 0.08 * signal.reachability
                memory_score -= 0.20 * signal.failure_risk
                memory_score -= 0.12 * signal.stall_risk
                option_loss_component = -0.10 * signal.option_loss_risk
                future_option_component += option_loss_component
                if not future_option_ablated:
                    memory_score += option_loss_component

            failure = max(float(row.failure_risk), signal.failure_risk)
            reachability = max(
                float(row.future_reachability),
                signal.reachability,
                signal.success_reachability,
            )
            support = int(getattr(row.support, "contextual_support", 0) or 0)
            local_support = int(getattr(row.support, "local_support", 0) or 0)
            context_rank = int(getattr(row.support, "context_rank", 0) or 0)
            support_confidence = 1.0 - math.exp(
                -max(0, support + local_support) / 4.0
            )
            semantic_confidence = max(
                _clamp01(reachability),
                _clamp01(failure),
                _clamp01(row.contradiction_risk),
                _clamp01(signal.success_reachability),
                _clamp01(row.prediction_confidence),
                _clamp01(row.completion_likelihood),
            )
            confidence = _clamp01(
                0.65 * support_confidence + 0.35 * semantic_confidence
            )

            transfer_probe = _transfer_probe_strength(
                view,
                getattr(row.support, "concept_ids", ()) or (),
            )
            exploration_score = float(row.exploration_score)
            if len(contexts.signatures) >= 5 and context_rank <= 2:
                exploration_score = _clamp01(
                    exploration_score + 0.30 * transfer_probe
                )

            output.append(
                Phase1ActionDecision(
                    action_id=int(row.action_id),
                    score=float(memory_score),
                    support=row.support,
                    exploration_score=exploration_score,
                    failure_risk=failure,
                    contradiction_risk=float(row.contradiction_risk),
                    future_reachability=reachability,
                    memory_confidence=confidence,
                    persistent_reachability=signal.reachability,
                    persistent_success_reachability=signal.success_reachability,
                    persistent_failure_risk=signal.failure_risk,
                    dead_end_risk=signal.stall_risk,
                    option_loss_risk=signal.option_loss_risk,
                    future_option_score_component=future_option_component,
                    development_stage=stage_name,
                )
            )
        return tuple(sorted(output, key=lambda item: item.action_id))

    def _planning_signal(
        self,
        *,
        view: MemoryReadView,
        contexts: DecisionContext,
        row: ContextualActionDecision,
        depth: int,
    ):
        if ablated(
            self.ablation_mask,
            CognitionAblation.PERSISTENT_PLANNING,
        ):
            from v7.memory.planning import PlanningSignal

            return PlanningSignal()
        planning_input = view.score_inputs(
            context_signature=contexts.planning_signature,
            action_ids=(row.action_id,),
        )[0]
        return self.planning.evaluate(
            view,
            planning_input.contingency_ids,
            depth=max(1, int(depth)),
            max_nodes=64,
        )


class StrategyExecutionCursor:
    """Execute a validated successful M6 procedure step-by-step until divergence."""

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
                if (
                    max(decision.failure_risk, decision.dead_end_risk)
                    >= self.risk_abort_threshold
                ):
                    continue
                candidate = (
                    confidence,
                    -len(procedure.steps),
                    strategy_id,
                    decision,
                )
                if best is None or candidate[:2] > best[:2] or (
                    candidate[:2] == best[:2]
                    and int(strategy_id) < int(best[2])
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
        if selected_strategy_id is None or int(selected_strategy_id) != int(
            self.strategy_id
        ):
            self.reset()
            return
        if int(terminal_polarity) != 0:
            self.reset()
            return
        self.position += 1
        if next_context_signature is None:
            self.reset()

    @staticmethod
    def _strategy_confidence(
        view: MemoryReadView,
        strategy_id: MemoryId,
    ) -> float:
        node = view.nodes.get(strategy_id)
        if not memory_is_active(node):
            return 0.0
        assert node is not None
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
    ablation_mask: int = 0,
) -> Phase1Selection:
    rows = tuple(decisions)
    if not rows:
        raise ValueError("environment returned no available actions")

    profile = profile_for_view(view)
    stage_name = profile.stage.name
    strategy_enabled = not ablated(
        ablation_mask,
        CognitionAblation.STRATEGY_EXECUTION,
    )
    if strategy_enabled:
        was_active = cursor.active
        strategy = cursor.recommend(
            view=view,
            planning=planning,
            contexts=contexts,
            decisions=rows,
        )
        if strategy is not None:
            decision, strategy_id = strategy
            if was_active or not _is_transfer_frontier(contexts, decision):
                return Phase1Selection(
                    decision=decision,
                    mode="strategy",
                    strategy_id=int(strategy_id),
                    effective_epsilon=0.0,
                    development_stage=stage_name,
                )
            cursor.reset()
    else:
        cursor.reset()

    memory = min(rows, key=lambda item: (-item.score, item.action_id))
    exploration = min(
        rows,
        key=lambda item: (-item.exploration_score, item.action_id),
    )
    support = getattr(memory, "support", None)
    context_rank = int(getattr(support, "context_rank", 0) or 0)
    local_support = int(getattr(support, "local_support", 0) or 0)
    confidence = _local_policy_confidence(
        memory.memory_confidence,
        signature_count=len(contexts.signatures),
        context_rank=context_rank,
        local_support=local_support,
    )
    transfer_frontier = _is_transfer_frontier(contexts, memory)
    developmental_multiplier = (
        1.0
        if ablated(
            ablation_mask,
            CognitionAblation.DEVELOPMENTAL_POLICY,
        )
        else float(profile.exploration_multiplier)
    )
    if transfer_frontier and not ablated(
        ablation_mask,
        CognitionAblation.DEVELOPMENTAL_POLICY,
    ):
        developmental_multiplier = max(1.0, developmental_multiplier)
    base_epsilon = _clamp01(float(epsilon))
    effective_epsilon = _clamp01(
        base_epsilon * developmental_multiplier * (1.0 - confidence) ** 2
    )
    if transfer_frontier and base_epsilon > 0.0:
        effective_epsilon = max(
            effective_epsilon,
            min(base_epsilon, 0.08),
        )

    high_confidence_threshold = (
        0.78
        if ablated(
            ablation_mask,
            CognitionAblation.DEVELOPMENTAL_POLICY,
        )
        else 0.82
        if profile.stage <= DevelopmentStage.CONTINGENCY
        else 0.76
        if profile.stage <= DevelopmentStage.TRANSFER
        else 0.68
    )

    if confidence >= high_confidence_threshold:
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
        if (
            profile.stage >= DevelopmentStage.PLANNING
            and not ablated(
                ablation_mask,
                CognitionAblation.DEVELOPMENTAL_POLICY,
            )
            and not transfer_frontier
        ):
            temperature *= 0.70
        maximum = max(float(item.score) for item in rows)
        weights = [
            math.exp(
                (float(item.score) - maximum) / max(1e-6, temperature)
            )
            for item in rows
        ]
        total = sum(weights)
        if not math.isfinite(total) or total <= 0.0:
            selected = memory
        else:
            selected = rng.choices(list(rows), weights=weights, k=1)[0]
        mode = (
            "memory"
            if selected.action_id == memory.action_id
            else "stochastic"
        )

    if (
        strategy_enabled
        and mode != "exploration"
        and not _is_transfer_frontier(contexts, selected)
    ):
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
                development_stage=stage_name,
            )

    return Phase1Selection(
        decision=selected,
        mode=mode,
        strategy_id=None,
        effective_epsilon=effective_epsilon,
        development_stage=stage_name,
    )
