from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any

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
    def __init__(self, *, evidence_lifecycle: EvidenceLifecycleStore, evidence_store: EvidenceStore | None = None, lifecycle: MemoryLifecycleController | None = None, concept_validator: EmpiricalConceptValidator | None = None) -> None:
        self.evidence_lifecycle = evidence_lifecycle
        self.lifecycle_runtime = MemoryLifecycleRuntime(lifecycle, evidence_store=evidence_store, evidence_lifecycle=evidence_lifecycle)
        self.concept_validator = concept_validator or EmpiricalConceptValidator()

    def run(self, view: MemoryReadView, *, writer: CanonicalMemoryWriter) -> DevelopmentalLifecycleResult:
        lifecycle_decisions, stats = self.lifecycle_runtime.run(view, writer=writer)
        summaries = self.evidence_lifecycle.transfer_summary(view.nodes.keys())
        concept_decisions = self.concept_validator.evaluate(view, transfer_summary=summaries)
        mutations = self.concept_validator.apply(concept_decisions, view=view, writer=writer)
        return DevelopmentalLifecycleResult(lifecycle_decisions, stats, concept_decisions, mutations)


@dataclass(frozen=True, slots=True)
class ScientificMismatch:
    path: str
    expected: Any
    observed: Any


@dataclass(frozen=True, slots=True)
class ScientificComparison:
    matched: bool
    mismatches: tuple[ScientificMismatch, ...]


class ScientificArtifactComparator:
    """Compare exported reference artifacts without importing another runtime."""

    def compare(self, expected: Any, observed: Any) -> ScientificComparison:
        mismatches: list[ScientificMismatch] = []
        self._walk("$", expected, observed, mismatches)
        return ScientificComparison(not mismatches, tuple(mismatches))

    def _walk(self, path: str, expected: Any, observed: Any, mismatches: list[ScientificMismatch]) -> None:
        if isinstance(expected, dict) and isinstance(observed, dict):
            if set(expected) != set(observed):
                mismatches.append(ScientificMismatch(path + ".keys", tuple(sorted(expected)), tuple(sorted(observed))))
            for key in sorted(set(expected) & set(observed), key=str):
                self._walk(f"{path}.{key}", expected[key], observed[key], mismatches)
            return
        if isinstance(expected, (list, tuple)) and isinstance(observed, (list, tuple)):
            if len(expected) != len(observed):
                mismatches.append(ScientificMismatch(path + ".length", len(expected), len(observed)))
                return
            for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
                self._walk(f"{path}[{index}]", left, right, mismatches)
            return
        if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
            if not isclose(float(expected), float(observed), rel_tol=1e-9, abs_tol=1e-12):
                mismatches.append(ScientificMismatch(path, expected, observed))
            return
        if expected != observed:
            mismatches.append(ScientificMismatch(path, expected, observed))
