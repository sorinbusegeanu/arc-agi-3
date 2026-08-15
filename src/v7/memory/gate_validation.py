from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.models import MemoryNode, NodeMutation
from v7.memory.read_view import MemoryReadView
from v7.memory.state import (
    CognitiveState,
    GateId,
    GateValidationState,
    gate_for_identity,
)
from v7.memory.status import ConceptValidationStatus, MemoryStatus
from v7.memory.writer import CanonicalMemoryWriter


@dataclass(frozen=True, slots=True)
class GatePolicy:
    minimum_support: int
    minimum_trials: int
    minimum_targets: int
    minimum_causal_gain: float
    trusted_support: int
    trusted_trials: int
    trusted_targets: int
    trusted_causal_gain: float


DEFAULT_GATE_POLICIES: Mapping[GateId, GatePolicy] = {
    GateId.G01: GatePolicy(2, 2, 1, 0.05, 4, 4, 2, 0.10),
    GateId.G12: GatePolicy(2, 2, 2, 0.05, 4, 4, 3, 0.10),
    GateId.G23C: GatePolicy(2, 2, 2, 0.05, 4, 4, 3, 0.10),
    GateId.G23R: GatePolicy(2, 2, 2, 0.05, 4, 4, 3, 0.10),
    GateId.G34: GatePolicy(3, 2, 1, 0.05, 4, 4, 2, 0.10),
    GateId.G45: GatePolicy(3, 3, 1, 0.05, 5, 5, 2, 0.10),
    GateId.G56: GatePolicy(3, 3, 2, 0.08, 5, 5, 3, 0.12),
}


@dataclass(frozen=True, slots=True)
class GateTrialSummary:
    trials: int = 0
    successes: int = 0
    independent_targets: int = 0
    mean_causal_gain: float = 0.0
    mean_transfer_score: float = 0.0
    positive_terminal_gain: float = 0.0
    prediction_gain: float = 0.0
    planning_gain: float = 0.0
    future_option_gain: float = 0.0
    efficiency_gain: float = 0.0

    @property
    def success_rate(self) -> float:
        return 0.0 if self.trials <= 0 else self.successes / self.trials


@dataclass(frozen=True, slots=True)
class GateValidationDecision:
    memory_id: MemoryId
    gate_id: GateId
    trials: int
    successes: int
    independent_targets: int
    mean_causal_gain: float
    structural_candidate: bool
    probe_eligible: bool
    tested: bool
    validated: bool
    rejected: bool
    trusted: bool
    previous_validation_state: GateValidationState
    next_validation_state: GateValidationState
    previous_cognitive_state: CognitiveState
    next_cognitive_state: CognitiveState
    dependency_satisfied: bool = True


class EmpiricalGateValidator:
    """Validate M1-M6 memories with gate-specific held-out causal evidence."""

    def __init__(
        self,
        policies: Mapping[GateId, GatePolicy] | None = None,
    ) -> None:
        merged = dict(DEFAULT_GATE_POLICIES)
        if policies:
            merged.update({GateId(int(key)): value for key, value in policies.items()})
        self.policies = merged

    @staticmethod
    def _node_gate(node: MemoryNode) -> GateId:
        explicit = int(getattr(node, "gate_id", 0) or 0)
        if explicit:
            try:
                return GateId(explicit)
            except ValueError:
                pass
        return gate_for_identity(node.level, node.type_id)

    @staticmethod
    def _validation_state(node: MemoryNode) -> GateValidationState:
        raw = int(getattr(node, "validation_state", GateValidationState.VALIDATED))
        try:
            return GateValidationState(raw)
        except ValueError:
            return GateValidationState.VALIDATED

    @staticmethod
    def _cognitive_state(node: MemoryNode) -> CognitiveState:
        raw = int(getattr(node, "cognitive_state", CognitiveState.ACTIVE))
        try:
            return CognitiveState(raw)
        except ValueError:
            return CognitiveState.ACTIVE

    def evaluate(
        self,
        view: MemoryReadView,
        *,
        gate_summaries: Mapping[MemoryId, GateTrialSummary],
        memory_ids: Iterable[MemoryId] | None = None,
        parent_validity: Mapping[MemoryId, bool] | None = None,
    ) -> tuple[GateValidationDecision, ...]:
        ids = tuple(sorted(memory_ids if memory_ids is not None else view.nodes, key=int))
        dependencies = parent_validity or {}
        decisions: list[GateValidationDecision] = []
        for memory_id in ids:
            node = view.nodes.get(memory_id)
            if node is None:
                continue
            gate = self._node_gate(node)
            if gate == GateId.NONE or gate not in self.policies:
                continue
            policy = self.policies[gate]
            summary = gate_summaries.get(memory_id, GateTrialSummary())
            support = max(0, int(node.support_count))
            dependency_satisfied = bool(dependencies.get(memory_id, True))
            structural = support >= policy.minimum_support
            probe = structural and dependency_satisfied
            tested = (
                probe
                and summary.trials >= policy.minimum_trials
                and summary.independent_targets >= policy.minimum_targets
            )
            validated = tested and summary.mean_causal_gain >= policy.minimum_causal_gain
            trusted = (
                validated
                and support >= policy.trusted_support
                and summary.trials >= policy.trusted_trials
                and summary.independent_targets >= policy.trusted_targets
                and summary.mean_causal_gain >= policy.trusted_causal_gain
            )
            rejected = tested and not validated

            if trusted:
                next_validation = GateValidationState.TRUSTED
            elif validated:
                next_validation = GateValidationState.VALIDATED
            elif rejected:
                next_validation = GateValidationState.REJECTED
            elif tested:
                next_validation = GateValidationState.TRANSFER_TESTED
            elif probe:
                next_validation = GateValidationState.PROBE_ELIGIBLE
            else:
                next_validation = GateValidationState.STRUCTURAL_CANDIDATE

            previous_cognitive = self._cognitive_state(node)
            if not dependency_satisfied:
                next_cognitive = CognitiveState.QUARANTINED
            elif next_validation in {
                GateValidationState.VALIDATED,
                GateValidationState.TRUSTED,
            }:
                # Validation unlocks cognition. Later lifecycle demotion can
                # still suppress a scientifically valid memory independently.
                next_cognitive = CognitiveState.ACTIVE
            elif next_validation == GateValidationState.REJECTED:
                next_cognitive = CognitiveState.QUARANTINED
            elif previous_cognitive == CognitiveState.RETIRED:
                next_cognitive = CognitiveState.RETIRED
            else:
                next_cognitive = CognitiveState.PROBE_ONLY

            decisions.append(
                GateValidationDecision(
                    memory_id=memory_id,
                    gate_id=gate,
                    trials=int(summary.trials),
                    successes=int(summary.successes),
                    independent_targets=int(summary.independent_targets),
                    mean_causal_gain=float(summary.mean_causal_gain),
                    structural_candidate=structural,
                    probe_eligible=probe,
                    tested=tested,
                    validated=validated,
                    rejected=rejected,
                    trusted=trusted,
                    previous_validation_state=self._validation_state(node),
                    next_validation_state=next_validation,
                    previous_cognitive_state=previous_cognitive,
                    next_cognitive_state=next_cognitive,
                    dependency_satisfied=dependency_satisfied,
                )
            )
        return tuple(decisions)

    @staticmethod
    def _compatibility_flags(node: MemoryNode, decision: GateValidationDecision) -> int:
        flags = int(node.status_flags)
        if decision.next_cognitive_state == CognitiveState.ACTIVE:
            flags |= int(MemoryStatus.ACTIVE)
            flags &= ~int(MemoryStatus.DEMOTED)
        else:
            flags &= ~int(MemoryStatus.ACTIVE)
            flags |= int(MemoryStatus.DEMOTED)
        if decision.validated or decision.trusted:
            flags |= int(MemoryStatus.PROMOTED)
        else:
            flags &= ~int(MemoryStatus.PROMOTED)

        if node.level == MemoryLevel.M4:
            mask = int(
                ConceptValidationStatus.CANDIDATE
                | ConceptValidationStatus.STRUCTURAL_SUPPORTED
                | ConceptValidationStatus.TRANSFER_CANDIDATE
                | ConceptValidationStatus.TRANSFER_VALIDATED
                | ConceptValidationStatus.TRANSFER_REJECTED
                | ConceptValidationStatus.TRUSTED
            )
            flags &= ~mask
            if decision.structural_candidate:
                flags |= int(ConceptValidationStatus.CANDIDATE)
            if decision.probe_eligible:
                flags |= int(ConceptValidationStatus.STRUCTURAL_SUPPORTED)
            if decision.probe_eligible and not decision.tested:
                flags |= int(ConceptValidationStatus.TRANSFER_CANDIDATE)
            if decision.validated:
                flags |= int(ConceptValidationStatus.TRANSFER_VALIDATED)
            if decision.rejected:
                flags |= int(ConceptValidationStatus.TRANSFER_REJECTED)
            if decision.trusted:
                flags |= int(ConceptValidationStatus.TRUSTED)
        return flags

    def apply(
        self,
        decisions: Iterable[GateValidationDecision],
        *,
        writer: CanonicalMemoryWriter,
    ) -> int:
        current_nodes = getattr(writer, "_nodes")
        mutations: list[NodeMutation] = []
        for decision in decisions:
            current = current_nodes.get(decision.memory_id)
            if current is None:
                continue
            flags = self._compatibility_flags(current, decision)
            next_cognitive = decision.next_cognitive_state
            # Lifecycle and validation share one mutable generation. A memory
            # that was already scientifically valid may have just been moved
            # to PROBE_ONLY/QUARANTINED by persistent low utility; do not
            # reactivate it merely because validation remains valid.
            current_validation = self._validation_state(current)
            current_cognitive = self._cognitive_state(current)
            if (
                current_validation in {
                    GateValidationState.VALIDATED,
                    GateValidationState.TRUSTED,
                }
                and decision.next_validation_state
                in {
                    GateValidationState.VALIDATED,
                    GateValidationState.TRUSTED,
                }
                and current_cognitive
                in {CognitiveState.PROBE_ONLY, CognitiveState.QUARANTINED}
            ):
                next_cognitive = current_cognitive
                flags &= ~int(MemoryStatus.ACTIVE)
                flags |= int(MemoryStatus.DEMOTED)
            if (
                int(current.validation_state) == int(decision.next_validation_state)
                and int(current.cognitive_state) == int(next_cognitive)
                and int(current.gate_id) == int(decision.gate_id)
                and int(current.status_flags) == flags
            ):
                continue
            mutations.append(
                NodeMutation(
                    current.memory_id,
                    current.level,
                    current.type_id,
                    support_delta=0,
                    status_flags=flags,
                    cognitive_state=int(next_cognitive),
                    validation_state=int(decision.next_validation_state),
                    gate_id=int(decision.gate_id),
                )
            )
        return writer.apply_mutation_batch(mutations) if mutations else 0


__all__ = [
    "DEFAULT_GATE_POLICIES",
    "EmpiricalGateValidator",
    "GatePolicy",
    "GateTrialSummary",
    "GateValidationDecision",
]
