from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import Iterable, Mapping

from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.models import NodeMutation
from v7.memory.read_view import MemoryReadView
from v7.memory.writer import CanonicalMemoryWriter


class ConceptValidationStatus(IntFlag):
    CANDIDATE = 1 << 8
    TRANSFER_VALIDATED = 1 << 9
    TRANSFER_REJECTED = 1 << 10


@dataclass(frozen=True, slots=True)
class ConceptValidationPolicy:
    minimum_support: int = 2
    minimum_trials: int = 2
    empirical_transfer_threshold: float = 0.50

    def __post_init__(self) -> None:
        if self.minimum_support < 1 or self.minimum_trials < 1:
            raise ValueError("minimum counts must be positive")
        if not 0.0 <= self.empirical_transfer_threshold <= 1.0:
            raise ValueError("empirical_transfer_threshold must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ConceptValidationDecision:
    memory_id: MemoryId
    transfer_trials: int
    empirical_transfer: float | None
    candidate: bool
    validated: bool
    rejected: bool
    previous_flags: int
    next_flags: int


class EmpiricalConceptValidator:
    """Validate M4 concept status using retrospective unseen-environment transfer only."""

    def __init__(self, policy: ConceptValidationPolicy | None = None) -> None:
        self.policy = policy or ConceptValidationPolicy()

    def evaluate(
        self,
        view: MemoryReadView,
        *,
        transfer_summary: Mapping[MemoryId, tuple[int, int, float]],
        memory_ids: Iterable[MemoryId] | None = None,
    ) -> tuple[ConceptValidationDecision, ...]:
        ids = tuple(sorted(memory_ids if memory_ids is not None else view.nodes, key=int))
        result: list[ConceptValidationDecision] = []
        for memory_id in ids:
            node = view.nodes.get(memory_id)
            if node is None or node.level != MemoryLevel.M4:
                continue
            total, successes, _mean_score = transfer_summary.get(memory_id, (0, 0, 0.0))
            empirical = None if total <= 0 else successes / total
            candidate = node.support_count >= self.policy.minimum_support
            tested = total >= self.policy.minimum_trials
            validated = candidate and tested and empirical is not None and empirical >= self.policy.empirical_transfer_threshold
            rejected = candidate and tested and not validated
            flags = int(node.status_flags)
            if candidate:
                flags |= int(ConceptValidationStatus.CANDIDATE)
            else:
                flags &= ~int(ConceptValidationStatus.CANDIDATE)
            if validated:
                flags |= int(ConceptValidationStatus.TRANSFER_VALIDATED)
                flags &= ~int(ConceptValidationStatus.TRANSFER_REJECTED)
            elif rejected:
                flags |= int(ConceptValidationStatus.TRANSFER_REJECTED)
                flags &= ~int(ConceptValidationStatus.TRANSFER_VALIDATED)
            result.append(ConceptValidationDecision(memory_id, total, empirical, candidate, validated, rejected, int(node.status_flags), flags))
        return tuple(result)

    def apply(self, decisions: Iterable[ConceptValidationDecision], *, view: MemoryReadView, writer: CanonicalMemoryWriter) -> int:
        mutations = []
        for decision in decisions:
            if decision.next_flags == decision.previous_flags:
                continue
            node = view.nodes[decision.memory_id]
            mutations.append(NodeMutation(node.memory_id, node.level, node.type_id, support_delta=0, status_flags=decision.next_flags))
        return writer.apply_mutation_batch(mutations) if mutations else 0
