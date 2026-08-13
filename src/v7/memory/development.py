from __future__ import annotations

from dataclasses import dataclass

from v7.memory.concept_validation import ConceptValidationDecision, EmpiricalConceptValidator
from v7.memory.evidence_lifecycle import EvidenceLifecycleStore
from v7.memory.evidence_store import EvidenceStore
from v7.memory.lifecycle import LifecycleDecision, MemoryLifecycleController
from v7.memory.lifecycle_runtime import LifecycleRunStats, MemoryLifecycleRuntime
from v7.memory.read_view import MemoryReadView
from v7.memory.writer import CanonicalMemoryWriter


@dataclass(frozen=True, slots=True)
class DevelopmentalLifecycleResult:
    lifecycle: tuple[LifecycleDecision, ...]
    lifecycle_stats: LifecycleRunStats
    concepts: tuple[ConceptValidationDecision, ...]
    concept_mutations: int


class DevelopmentalLifecycleRuntime:
    """One ordered lifecycle pass over a published generation.

    Prospective memory fitness drives retention/replay/promotion pressure first.
    Empirical held-out transfer then validates or rejects M4 concept candidates.
    Historical evidence remains in append-only stores and is never substituted for
    the online transfer prior stored in MemoryScore.
    """

    def __init__(
        self,
        *,
        evidence_lifecycle: EvidenceLifecycleStore,
        evidence_store: EvidenceStore | None = None,
        lifecycle: MemoryLifecycleController | None = None,
        concept_validator: EmpiricalConceptValidator | None = None,
    ) -> None:
        self.evidence_lifecycle = evidence_lifecycle
        self.lifecycle_runtime = MemoryLifecycleRuntime(
            lifecycle,
            evidence_store=evidence_store,
            evidence_lifecycle=evidence_lifecycle,
        )
        self.concept_validator = concept_validator or EmpiricalConceptValidator()

    def run(self, view: MemoryReadView, *, writer: CanonicalMemoryWriter) -> DevelopmentalLifecycleResult:
        lifecycle_decisions, stats = self.lifecycle_runtime.run(view, writer=writer)
        summaries = self.evidence_lifecycle.transfer_summary(view.nodes.keys())
        concept_decisions = self.concept_validator.evaluate(view, transfer_summary=summaries)
        mutations = self.concept_validator.apply(concept_decisions, view=view, writer=writer)
        return DevelopmentalLifecycleResult(lifecycle_decisions, stats, concept_decisions, mutations)
