from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from typing import Iterable, Mapping

from v7.memory.ids import MemoryId
from v7.memory.models import MemoryNode, MemoryScore, NodeMutation
from v7.memory.read_view import MemoryReadView
from v7.memory.status import MemoryStatus
from v7.memory.writer import CanonicalMemoryWriter


@dataclass(frozen=True, slots=True)
class LifecyclePolicy:
    promote_threshold: float = 0.55
    retain_threshold: float = 0.25
    minimum_promotion_support: int = 2
    replay_prediction_error: float = 0.50
    replay_learning_value: float = 0.50
    replay_transfer_prior: float = 0.40
    replay_empirical_transfer: float = 0.50
    replay_explanatory_potential: float = 0.40
    replay_contradiction_severity: float = 0.50
    replay_limit: int = 256
    significance_weight: float = 0.20
    prediction_error_weight: float = 0.25
    learning_value_weight: float = 0.25
    transfer_prior_weight: float = 0.075
    empirical_transfer_weight: float = 0.075
    explanatory_weight: float = 0.15
    support_weight: float = 0.05

    def __post_init__(self) -> None:
        if self.minimum_promotion_support < 1:
            raise ValueError("minimum_promotion_support must be positive")
        if self.replay_limit < 0:
            raise ValueError("replay_limit must be non-negative")
        for value in (self.promote_threshold, self.retain_threshold):
            if value < 0:
                raise ValueError("lifecycle thresholds must be non-negative")
        if self.retain_threshold > self.promote_threshold:
            raise ValueError("retain_threshold cannot exceed promote_threshold")


@dataclass(frozen=True, slots=True)
class LifecycleDecision:
    memory_id: MemoryId
    fitness: float
    promote: bool
    demote: bool
    replay: bool
    previous_flags: int
    next_flags: int
    empirical_transfer: float = 0.0
    contradiction_severity: float = 0.0


@dataclass(frozen=True, slots=True)
class ReplayRequest:
    memory_id: MemoryId
    priority: float
    reason_mask: int


class ReplayQueue:
    """Deterministic bounded replay frontier; highest priority survives."""

    def __init__(self, *, limit: int = 256) -> None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        self.limit = int(limit)
        self._requests: dict[MemoryId, ReplayRequest] = {}

    def push(self, request: ReplayRequest) -> None:
        current = self._requests.get(request.memory_id)
        if current is None or request.priority > current.priority or (
            request.priority == current.priority and request.reason_mask > current.reason_mask
        ):
            self._requests[request.memory_id] = request
        if len(self._requests) > self.limit:
            ordered = sorted(
                self._requests.values(),
                key=lambda item: (-item.priority, int(item.memory_id)),
            )[: self.limit]
            self._requests = {item.memory_id: item for item in ordered}

    def snapshot(self) -> tuple[ReplayRequest, ...]:
        return tuple(
            sorted(
                self._requests.values(),
                key=lambda item: (-item.priority, int(item.memory_id)),
            )
        )

    def pop_all(self) -> tuple[ReplayRequest, ...]:
        rows = self.snapshot()
        self._requests.clear()
        return rows


class MemoryLifecycleController:
    """Evaluate retention, replay and promotion state from one immutable generation."""

    REPLAY_PE = 1 << 0
    REPLAY_LV = 1 << 1
    REPLAY_TRANSFER_PRIOR = 1 << 2
    REPLAY_EXPLANATORY = 1 << 3
    REPLAY_TRANSFER_EMPIRICAL = 1 << 4
    REPLAY_CONTRADICTION = 1 << 5

    def __init__(self, policy: LifecyclePolicy | None = None) -> None:
        self.policy = policy or LifecyclePolicy()
        self.replay_queue = ReplayQueue(limit=self.policy.replay_limit)

    def fitness(
        self,
        node: MemoryNode,
        score: MemoryScore | None,
        *,
        empirical_transfer: float = 0.0,
    ) -> float:
        score = score or MemoryScore(memory_id=node.memory_id)
        p = self.policy
        support_term = min(
            1.0,
            log1p(max(0, node.support_count)) / log1p(8.0),
        )
        empirical = min(1.0, max(0.0, float(empirical_transfer)))
        return (
            p.significance_weight * max(0.0, score.significance)
            + p.prediction_error_weight * max(0.0, score.prediction_error)
            + p.learning_value_weight * max(0.0, score.learning_value)
            + p.transfer_prior_weight * max(0.0, score.transfer_prior)
            + p.empirical_transfer_weight * empirical
            + p.explanatory_weight * max(0.0, score.explanatory_potential)
            + p.support_weight * support_term
        )

    def _replay_reason(
        self,
        score: MemoryScore | None,
        empirical_transfer: float,
        contradiction_severity: float,
    ) -> tuple[int, float]:
        p = self.policy
        reason = 0
        priority = 0.0
        if score is not None:
            if score.prediction_error >= p.replay_prediction_error:
                reason |= self.REPLAY_PE
                priority = max(priority, score.prediction_error)
            if score.learning_value >= p.replay_learning_value:
                reason |= self.REPLAY_LV
                priority = max(priority, score.learning_value)
            if score.transfer_prior >= p.replay_transfer_prior:
                reason |= self.REPLAY_TRANSFER_PRIOR
                priority = max(priority, score.transfer_prior)
            if score.explanatory_potential >= p.replay_explanatory_potential:
                reason |= self.REPLAY_EXPLANATORY
                priority = max(priority, score.explanatory_potential)
        empirical = min(1.0, max(0.0, float(empirical_transfer)))
        if empirical >= p.replay_empirical_transfer:
            reason |= self.REPLAY_TRANSFER_EMPIRICAL
            priority = max(priority, empirical)
        contradiction = max(0.0, float(contradiction_severity))
        if contradiction >= p.replay_contradiction_severity:
            reason |= self.REPLAY_CONTRADICTION
            priority = max(priority, contradiction)
        return reason, priority

    def evaluate(
        self,
        view: MemoryReadView,
        memory_ids: Iterable[MemoryId] | None = None,
        *,
        empirical_transfer: Mapping[MemoryId, float] | None = None,
        contradiction_severity: Mapping[MemoryId, float] | None = None,
    ) -> tuple[LifecycleDecision, ...]:
        ids = tuple(
            sorted(
                memory_ids if memory_ids is not None else view.nodes.keys(),
                key=int,
            )
        )
        transfer = empirical_transfer or {}
        contradictions = contradiction_severity or {}
        decisions: list[LifecycleDecision] = []
        for memory_id in ids:
            node = view.nodes.get(memory_id)
            if node is None:
                continue
            score = view.scores.get(memory_id)
            empirical = float(transfer.get(memory_id, 0.0))
            contradiction = float(contradictions.get(memory_id, 0.0))
            fitness = self.fitness(node, score, empirical_transfer=empirical)
            promote = (
                node.support_count >= self.policy.minimum_promotion_support
                and fitness >= self.policy.promote_threshold
            )
            demote = not promote and fitness < self.policy.retain_threshold
            replay_reason, replay_priority = self._replay_reason(
                score,
                empirical,
                contradiction,
            )
            replay = replay_reason != 0
            flags = int(node.status_flags) | int(MemoryStatus.ACTIVE)
            if promote:
                flags = (flags | int(MemoryStatus.PROMOTED)) & ~int(
                    MemoryStatus.DEMOTED
                )
            elif demote:
                flags = (flags | int(MemoryStatus.DEMOTED)) & ~int(
                    MemoryStatus.PROMOTED | MemoryStatus.ACTIVE
                )
            if replay:
                flags |= int(MemoryStatus.REPLAY_QUEUED)
                self.replay_queue.push(
                    ReplayRequest(memory_id, replay_priority, replay_reason)
                )
            else:
                flags &= ~int(MemoryStatus.REPLAY_QUEUED)
            decisions.append(
                LifecycleDecision(
                    memory_id,
                    fitness,
                    promote,
                    demote,
                    replay,
                    int(node.status_flags),
                    flags,
                    empirical,
                    contradiction,
                )
            )
        return tuple(decisions)

    def apply(
        self,
        view: MemoryReadView,
        *,
        writer: CanonicalMemoryWriter,
        memory_ids: Iterable[MemoryId] | None = None,
        empirical_transfer: Mapping[MemoryId, float] | None = None,
        contradiction_severity: Mapping[MemoryId, float] | None = None,
    ) -> tuple[LifecycleDecision, ...]:
        decisions = self.evaluate(
            view,
            memory_ids,
            empirical_transfer=empirical_transfer,
            contradiction_severity=contradiction_severity,
        )
        mutations = []
        for decision in decisions:
            if decision.next_flags == decision.previous_flags:
                continue
            node = view.nodes[decision.memory_id]
            mutations.append(
                NodeMutation(
                    decision.memory_id,
                    node.level,
                    node.type_id,
                    support_delta=0,
                    status_flags=decision.next_flags,
                )
            )
        if mutations:
            writer.apply_mutation_batch(mutations)
        return decisions
