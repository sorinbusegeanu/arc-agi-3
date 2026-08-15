from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from typing import Iterable, Mapping

from v7.memory.ids import MemoryId
from v7.memory.models import MemoryNode, MemoryScore, NodeMutation
from v7.memory.read_view import MemoryReadView
from v7.memory.state import CognitiveState, GateId, GateValidationState, is_gate_validated
from v7.memory.status import MemoryStatus, memory_cognitive_state, memory_validation_state
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
    probe_after_low_windows: int = 2
    quarantine_after_low_windows: int = 4
    quarantine_after_harm_windows: int = 2
    retire_after_low_windows: int = 8
    retire_after_harm_windows: int = 4
    reactivate_after_positive_windows: int = 2

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
        if min(
            self.probe_after_low_windows,
            self.quarantine_after_low_windows,
            self.quarantine_after_harm_windows,
            self.retire_after_low_windows,
            self.retire_after_harm_windows,
            self.reactivate_after_positive_windows,
        ) < 1:
            raise ValueError("lifecycle window thresholds must be positive")
        if self.quarantine_after_low_windows < self.probe_after_low_windows:
            raise ValueError("quarantine window cannot precede probe-only window")
        if self.retire_after_low_windows < self.quarantine_after_low_windows:
            raise ValueError("retirement cannot precede low-utility quarantine")
        if self.retire_after_harm_windows < self.quarantine_after_harm_windows:
            raise ValueError("retirement cannot precede harm quarantine")


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
    previous_cognitive_state: int = int(CognitiveState.ACTIVE)
    next_cognitive_state: int = int(CognitiveState.ACTIVE)

    @property
    def retired(self) -> bool:
        return (
            self.previous_cognitive_state != int(CognitiveState.RETIRED)
            and self.next_cognitive_state == int(CognitiveState.RETIRED)
        )


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
    """Evaluate retention/replay without treating fitness as scientific validation."""

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

    @staticmethod
    def _scientifically_valid(node: MemoryNode) -> bool:
        if int(getattr(node, "gate_id", GateId.NONE)) == int(GateId.NONE):
            return True
        state = memory_validation_state(node)
        return state is not None and is_gate_validated(state)

    def _window_state(
        self,
        node: MemoryNode,
        fitness: float,
        *,
        window: tuple[int, int, int] | None,
        legacy_demote: bool,
    ) -> CognitiveState:
        current = memory_cognitive_state(node) or CognitiveState.ACTIVE
        validation = memory_validation_state(node) or GateValidationState.VALIDATED
        if current == CognitiveState.RETIRED:
            return current
        if validation == GateValidationState.REJECTED:
            return CognitiveState.QUARANTINED
        if not self._scientifically_valid(node):
            return (
                CognitiveState.QUARANTINED
                if current == CognitiveState.QUARANTINED
                else CognitiveState.PROBE_ONLY
            )
        if window is None:
            return CognitiveState.QUARANTINED if legacy_demote else CognitiveState.ACTIVE
        low_windows, harm_windows, positive_windows = (int(value) for value in window)
        p = self.policy
        if (
            harm_windows >= p.retire_after_harm_windows
            or low_windows >= p.retire_after_low_windows
        ):
            return CognitiveState.RETIRED
        if harm_windows >= p.quarantine_after_harm_windows:
            return CognitiveState.QUARANTINED
        if low_windows >= p.quarantine_after_low_windows:
            return CognitiveState.QUARANTINED
        if low_windows >= p.probe_after_low_windows:
            return CognitiveState.PROBE_ONLY
        if positive_windows >= p.reactivate_after_positive_windows:
            return CognitiveState.ACTIVE
        if current in {CognitiveState.PROBE_ONLY, CognitiveState.QUARANTINED}:
            return current
        return CognitiveState.ACTIVE

    def evaluate(
        self,
        view: MemoryReadView,
        memory_ids: Iterable[MemoryId] | None = None,
        *,
        empirical_transfer: Mapping[MemoryId, float] | None = None,
        contradiction_severity: Mapping[MemoryId, float] | None = None,
        lifecycle_windows: Mapping[MemoryId, tuple[int, int, int]] | None = None,
    ) -> tuple[LifecycleDecision, ...]:
        ids = tuple(
            sorted(
                memory_ids if memory_ids is not None else view.nodes.keys(),
                key=int,
            )
        )
        transfer = empirical_transfer or {}
        contradictions = contradiction_severity or {}
        windows = lifecycle_windows
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
            legacy_demote = not promote and fitness < self.policy.retain_threshold
            next_cognitive = self._window_state(
                node,
                fitness,
                window=None if windows is None else windows.get(memory_id, (0, 0, 0)),
                legacy_demote=legacy_demote,
            )
            previous_cognitive = memory_cognitive_state(node) or CognitiveState.ACTIVE
            if next_cognitive == CognitiveState.RETIRED:
                promote = False
            demote = (
                previous_cognitive == CognitiveState.ACTIVE
                and next_cognitive != CognitiveState.ACTIVE
            )
            replay_reason, replay_priority = self._replay_reason(
                score,
                empirical,
                contradiction,
            )
            replay = (
                next_cognitive != CognitiveState.RETIRED
                and (replay_reason != 0 or next_cognitive == CognitiveState.PROBE_ONLY)
            )
            flags = int(node.status_flags)
            if next_cognitive == CognitiveState.ACTIVE:
                flags |= int(MemoryStatus.ACTIVE)
                flags &= ~int(MemoryStatus.DEMOTED)
            else:
                flags |= int(MemoryStatus.DEMOTED)
                flags &= ~int(MemoryStatus.ACTIVE)
            if promote:
                flags |= int(MemoryStatus.PROMOTED)
            elif demote or next_cognitive == CognitiveState.RETIRED:
                flags &= ~int(MemoryStatus.PROMOTED)
            if replay:
                flags |= int(MemoryStatus.REPLAY_QUEUED)
                self.replay_queue.push(
                    ReplayRequest(memory_id, max(replay_priority, fitness), replay_reason)
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
                    int(previous_cognitive),
                    int(next_cognitive),
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
        lifecycle_windows: Mapping[MemoryId, tuple[int, int, int]] | None = None,
    ) -> tuple[LifecycleDecision, ...]:
        decisions = self.evaluate(
            view,
            memory_ids,
            empirical_transfer=empirical_transfer,
            contradiction_severity=contradiction_severity,
            lifecycle_windows=lifecycle_windows,
        )
        mutations = []
        for decision in decisions:
            node = view.nodes[decision.memory_id]
            if (
                decision.next_flags == decision.previous_flags
                and int(getattr(node, "cognitive_state", CognitiveState.ACTIVE))
                == decision.next_cognitive_state
            ):
                continue
            mutations.append(
                NodeMutation(
                    decision.memory_id,
                    node.level,
                    node.type_id,
                    support_delta=0,
                    status_flags=decision.next_flags,
                    cognitive_state=decision.next_cognitive_state,
                )
            )
        if mutations:
            writer.apply_mutation_batch(mutations)
        return decisions
