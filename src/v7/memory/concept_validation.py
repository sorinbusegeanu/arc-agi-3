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
    STRUCTURAL_SUPPORTED = 1 << 11
    TRANSFER_CANDIDATE = 1 << 12
    TRUSTED = 1 << 13


@dataclass(frozen=True, slots=True)
class ConceptValidationPolicy:
    minimum_support: int = 2
    structural_support: int = 3
    minimum_trials: int = 2
    empirical_transfer_threshold: float = 0.50
    trusted_support: int = 4
    trusted_trials: int = 4
    trusted_transfer_threshold: float = 0.67

    def __post_init__(self) -> None:
        if min(
            self.minimum_support,
            self.structural_support,
            self.minimum_trials,
            self.trusted_support,
            self.trusted_trials,
        ) < 1:
            raise ValueError("minimum counts must be positive")
        if self.structural_support < self.minimum_support:
            raise ValueError("structural_support cannot be below minimum_support")
        if self.trusted_support < self.structural_support:
            raise ValueError("trusted_support cannot be below structural_support")
        if self.trusted_trials < self.minimum_trials:
            raise ValueError("trusted_trials cannot be below minimum_trials")
        for value in (
            self.empirical_transfer_threshold,
            self.trusted_transfer_threshold,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("transfer thresholds must be in [0, 1]")
        if self.trusted_transfer_threshold < self.empirical_transfer_threshold:
            raise ValueError(
                "trusted_transfer_threshold cannot be below validation threshold"
            )


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
    structural_supported: bool = False
    transfer_candidate: bool = False
    trusted: bool = False


class EmpiricalConceptValidator:
    """Stage M4 concepts from structural candidate to trusted transfer concept."""

    def __init__(self, policy: ConceptValidationPolicy | None = None) -> None:
        self.policy = policy or ConceptValidationPolicy()

    def evaluate(
        self,
        view: MemoryReadView,
        *,
        transfer_summary: Mapping[MemoryId, tuple[int, int, float]],
        memory_ids: Iterable[MemoryId] | None = None,
    ) -> tuple[ConceptValidationDecision, ...]:
        ids = tuple(
            sorted(memory_ids if memory_ids is not None else view.nodes, key=int)
        )
        result: list[ConceptValidationDecision] = []
        p = self.policy
        for memory_id in ids:
            node = view.nodes.get(memory_id)
            if node is None or node.level != MemoryLevel.M4:
                continue
            total, successes, _mean_score = transfer_summary.get(
                memory_id, (0, 0, 0.0)
            )
            empirical = None if total <= 0 else successes / total
            candidate = int(node.support_count) >= p.minimum_support
            structural_supported = int(node.support_count) >= p.structural_support
            tested = total >= p.minimum_trials
            validated = (
                structural_supported
                and tested
                and empirical is not None
                and empirical >= p.empirical_transfer_threshold
            )
            trusted = (
                validated
                and int(node.support_count) >= p.trusted_support
                and total >= p.trusted_trials
                and empirical is not None
                and empirical >= p.trusted_transfer_threshold
            )
            rejected = structural_supported and tested and not validated
            transfer_candidate = structural_supported and not tested

            flags = int(node.status_flags)
            flags = self._set_flag(
                flags,
                ConceptValidationStatus.CANDIDATE,
                candidate,
            )
            flags = self._set_flag(
                flags,
                ConceptValidationStatus.STRUCTURAL_SUPPORTED,
                structural_supported,
            )
            flags = self._set_flag(
                flags,
                ConceptValidationStatus.TRANSFER_CANDIDATE,
                transfer_candidate,
            )
            flags = self._set_flag(
                flags,
                ConceptValidationStatus.TRANSFER_VALIDATED,
                validated,
            )
            flags = self._set_flag(
                flags,
                ConceptValidationStatus.TRUSTED,
                trusted,
            )
            flags = self._set_flag(
                flags,
                ConceptValidationStatus.TRANSFER_REJECTED,
                rejected,
            )
            if rejected:
                flags &= ~int(
                    ConceptValidationStatus.TRANSFER_VALIDATED
                    | ConceptValidationStatus.TRUSTED
                    | ConceptValidationStatus.TRANSFER_CANDIDATE
                )

            result.append(
                ConceptValidationDecision(
                    memory_id=memory_id,
                    transfer_trials=total,
                    empirical_transfer=empirical,
                    candidate=candidate,
                    validated=validated,
                    rejected=rejected,
                    previous_flags=int(node.status_flags),
                    next_flags=flags,
                    structural_supported=structural_supported,
                    transfer_candidate=transfer_candidate,
                    trusted=trusted,
                )
            )
        return tuple(result)

    @staticmethod
    def _set_flag(flags: int, flag: ConceptValidationStatus, enabled: bool) -> int:
        if enabled:
            return int(flags) | int(flag)
        return int(flags) & ~int(flag)

    def apply(
        self,
        decisions: Iterable[ConceptValidationDecision],
        *,
        view: MemoryReadView,
        writer: CanonicalMemoryWriter,
    ) -> int:
        mutations = []
        for decision in decisions:
            if decision.next_flags == decision.previous_flags:
                continue
            node = view.nodes[decision.memory_id]
            mutations.append(
                NodeMutation(
                    node.memory_id,
                    node.level,
                    node.type_id,
                    support_delta=0,
                    status_flags=decision.next_flags,
                )
            )
        return writer.apply_mutation_batch(mutations) if mutations else 0
