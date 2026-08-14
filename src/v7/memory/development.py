from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any

from v7.memory.concept_validation import (
    ConceptValidationDecision,
    EmpiricalConceptValidator,
)
from v7.memory.developmental_policy import focused_replay_ids, profile_for_view
from v7.memory.evidence_lifecycle import EvidenceLifecycleStore
from v7.memory.evidence_store import EvidenceRecord, EvidenceStore
from v7.memory.evidence_types import EvidenceType
from v7.memory.lifecycle import LifecycleDecision, MemoryLifecycleController, MemoryStatus
from v7.memory.lifecycle_runtime import LifecycleRunStats, MemoryLifecycleRuntime
from v7.memory.models import NodeMutation
from v7.memory.read_view import MemoryReadView
from v7.memory.writer import CanonicalMemoryWriter


@dataclass(frozen=True, slots=True)
class DevelopmentalLifecycleResult:
    lifecycle: tuple[LifecycleDecision, ...]
    lifecycle_stats: LifecycleRunStats
    concepts: tuple[ConceptValidationDecision, ...]
    concept_mutations: int
    development_stage: str = "CONTROL"
    developmental_replay_mutations: int = 0


class DevelopmentalLifecycleRuntime:
    def __init__(
        self,
        *,
        evidence_lifecycle: EvidenceLifecycleStore,
        evidence_store: EvidenceStore | None = None,
        lifecycle: MemoryLifecycleController | None = None,
        concept_validator: EmpiricalConceptValidator | None = None,
    ) -> None:
        self.evidence_lifecycle = evidence_lifecycle
        self.evidence_store = evidence_store
        self.lifecycle_runtime = MemoryLifecycleRuntime(
            lifecycle,
            evidence_store=evidence_store,
            evidence_lifecycle=evidence_lifecycle,
        )
        self.concept_validator = concept_validator or EmpiricalConceptValidator()

    def run(
        self,
        view: MemoryReadView,
        *,
        writer: CanonicalMemoryWriter,
    ) -> DevelopmentalLifecycleResult:
        profile = profile_for_view(view)
        lifecycle_decisions, stats = self.lifecycle_runtime.run(view, writer=writer)
        summaries = self.evidence_lifecycle.transfer_summary(view.nodes.keys())
        concept_decisions = self.concept_validator.evaluate(
            view,
            transfer_summary=summaries,
        )
        mutations = self.concept_validator.apply(
            concept_decisions,
            view=view,
            writer=writer,
        )
        replay_mutations = self._apply_developmental_replay(
            view,
            writer=writer,
            stage_name=profile.stage.name,
        )
        if self.evidence_store is not None and concept_decisions:
            generation = int(writer.mutable_generation_id)
            self.evidence_store.append_evidence_batch(
                EvidenceRecord(
                    memory_id=d.memory_id,
                    evidence_type=int(EvidenceType.CONCEPT_VALIDATION),
                    generation_id=generation,
                    payload={
                        "transfer_trials": int(d.transfer_trials),
                        "empirical_transfer": d.empirical_transfer,
                        "candidate": bool(d.candidate),
                        "structural_supported": bool(d.structural_supported),
                        "transfer_candidate": bool(d.transfer_candidate),
                        "validated": bool(d.validated),
                        "trusted": bool(d.trusted),
                        "rejected": bool(d.rejected),
                        "previous_flags": int(d.previous_flags),
                        "next_flags": int(d.next_flags),
                        "development_stage": profile.stage.name,
                        "validation_source_games": list(
                            self.evidence_lifecycle.provenance_source_games(d.memory_id)
                        ),
                    },
                )
                for d in concept_decisions
            )
        return DevelopmentalLifecycleResult(
            lifecycle_decisions,
            stats,
            concept_decisions,
            mutations,
            profile.stage.name,
            replay_mutations,
        )

    def _apply_developmental_replay(
        self,
        view: MemoryReadView,
        *,
        writer: CanonicalMemoryWriter,
        stage_name: str,
    ) -> int:
        profile = profile_for_view(view)
        focus = focused_replay_ids(view, profile=profile, limit=64)
        writer_nodes = getattr(writer, "_nodes")
        mutations: list[NodeMutation] = []
        records: list[EvidenceRecord] = []
        generation = int(writer.mutable_generation_id)
        for memory_id in focus:
            current = writer_nodes.get(memory_id)
            if current is None:
                continue
            if int(current.status_flags) & int(MemoryStatus.REPLAY_QUEUED):
                continue
            next_flags = int(current.status_flags) | int(MemoryStatus.REPLAY_QUEUED)
            mutations.append(
                NodeMutation(
                    memory_id,
                    current.level,
                    current.type_id,
                    support_delta=0,
                    status_flags=next_flags,
                )
            )
            if self.evidence_store is not None:
                records.append(
                    EvidenceRecord(
                        memory_id=memory_id,
                        evidence_type=int(EvidenceType.REPLAY),
                        generation_id=generation,
                        payload={
                            "reason": "developmental_focus",
                            "development_stage": stage_name,
                            "memory_level": int(current.level),
                        },
                    )
                )
        applied = writer.apply_mutation_batch(mutations) if mutations else 0
        if self.evidence_store is not None and records:
            self.evidence_store.append_evidence_batch(records)
        return applied


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
    def compare(self, expected: Any, observed: Any) -> ScientificComparison:
        mismatches: list[ScientificMismatch] = []
        self._walk("$", expected, observed, mismatches)
        return ScientificComparison(not mismatches, tuple(mismatches))

    def _walk(
        self,
        path: str,
        expected: Any,
        observed: Any,
        mismatches: list[ScientificMismatch],
    ) -> None:
        if isinstance(expected, dict) and isinstance(observed, dict):
            if set(expected) != set(observed):
                mismatches.append(
                    ScientificMismatch(
                        path + ".keys",
                        tuple(sorted(expected)),
                        tuple(sorted(observed)),
                    )
                )
            for key in sorted(set(expected) & set(observed), key=str):
                self._walk(
                    f"{path}.{key}",
                    expected[key],
                    observed[key],
                    mismatches,
                )
            return
        if isinstance(expected, (list, tuple)) and isinstance(
            observed, (list, tuple)
        ):
            if len(expected) != len(observed):
                mismatches.append(
                    ScientificMismatch(
                        path + ".length", len(expected), len(observed)
                    )
                )
                return
            for index, (left, right) in enumerate(
                zip(expected, observed, strict=True)
            ):
                self._walk(
                    f"{path}[{index}]", left, right, mismatches
                )
            return
        if isinstance(expected, (int, float)) and isinstance(
            observed, (int, float)
        ):
            if not isclose(
                float(expected),
                float(observed),
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                mismatches.append(
                    ScientificMismatch(path, expected, observed)
                )
            return
        if expected != observed:
            mismatches.append(ScientificMismatch(path, expected, observed))
