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
from v7.memory.gate_validation import (
    EmpiricalGateValidator,
    GateTrialSummary,
    GateValidationDecision,
)
from v7.memory.lifecycle import LifecycleDecision, MemoryLifecycleController, MemoryStatus
from v7.memory.lifecycle_runtime import LifecycleRunStats, MemoryLifecycleRuntime
from v7.memory.models import NodeMutation
from v7.memory.read_view import MemoryReadView
from v7.memory.state import GateId, gate_for_identity
from v7.memory.writer import CanonicalMemoryWriter


@dataclass(frozen=True, slots=True)
class DevelopmentalLifecycleResult:
    lifecycle: tuple[LifecycleDecision, ...]
    lifecycle_stats: LifecycleRunStats
    concepts: tuple[ConceptValidationDecision, ...]
    concept_mutations: int
    development_stage: str = "CONTROL"
    developmental_replay_mutations: int = 0
    gates: tuple[GateValidationDecision, ...] = ()
    gate_mutations: int = 0


class DevelopmentalLifecycleRuntime:
    def __init__(
        self,
        *,
        evidence_lifecycle: EvidenceLifecycleStore,
        evidence_store: EvidenceStore | None = None,
        lifecycle: MemoryLifecycleController | None = None,
        concept_validator: EmpiricalConceptValidator | None = None,
        gate_validator: EmpiricalGateValidator | None = None,
    ) -> None:
        self.evidence_lifecycle = evidence_lifecycle
        self.evidence_store = evidence_store
        self.lifecycle_runtime = MemoryLifecycleRuntime(
            lifecycle,
            evidence_store=evidence_store,
            evidence_lifecycle=evidence_lifecycle,
        )
        self.concept_validator = concept_validator or EmpiricalConceptValidator()
        self.gate_validator = gate_validator or EmpiricalGateValidator()

    def run(
        self,
        view: MemoryReadView,
        *,
        writer: CanonicalMemoryWriter,
    ) -> DevelopmentalLifecycleResult:
        profile = profile_for_view(view)
        lifecycle_decisions, stats = self.lifecycle_runtime.run(view, writer=writer)

        # Existing M4 held-out validation remains a compatibility surface for
        # reports/tests while the generic validator becomes authoritative.
        formation_generations = {
            memory_id: int(node.created_generation)
            for memory_id, node in view.nodes.items()
            if int(node.level) == 4
        }
        summaries = self.evidence_lifecycle.heldout_transfer_summary(
            formation_generations.keys(),
            formation_generations=formation_generations,
        )
        concept_decisions = self.concept_validator.evaluate(
            view,
            transfer_summary=summaries,
        )
        concept_mutations = self.concept_validator.apply(
            concept_decisions,
            view=view,
            writer=writer,
        )

        # Freeze construction scope once, then validate every reusable level
        # through the same empirical gate framework.
        gate_ids = []
        for memory_id, node in view.nodes.items():
            gate = gate_for_identity(node.level, node.type_id)
            if gate == GateId.NONE:
                continue
            self.evidence_lifecycle.freeze_candidate_scope(
                memory_id,
                int(node.created_generation),
            )
            gate_ids.append(memory_id)
        gate_summaries = self.evidence_lifecycle.gate_trial_summary(gate_ids)

        # Legacy held-out M4 transfer records are upgraded to a causal summary
        # only when no genuine gate trial exists yet. New evidence always wins.
        for memory_id, (total, successes, mean_score) in summaries.items():
            if memory_id in gate_summaries or total <= 0:
                continue
            gate_summaries[memory_id] = GateTrialSummary(
                trials=int(total),
                successes=int(successes),
                independent_targets=max(1, min(int(total), 2)),
                mean_causal_gain=float(mean_score),
                mean_transfer_score=float(mean_score),
                positive_terminal_gain=(
                    float(successes) / float(total) if total > 0 else 0.0
                ),
            )
        gate_decisions = self.gate_validator.evaluate(
            view,
            gate_summaries=gate_summaries,
            memory_ids=gate_ids,
        )
        gate_mutations = self.gate_validator.apply(
            gate_decisions,
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
                        "heldout_validation": True,
                        "formation_generation": formation_generations.get(d.memory_id),
                        "validation_source_games": list(
                            self.evidence_lifecycle.provenance_source_games_at(
                                d.memory_id,
                                formation_generations.get(
                                    d.memory_id,
                                    int(view.generation_id),
                                ),
                            )
                        ),
                    },
                )
                for d in concept_decisions
            )
        if self.evidence_store is not None and gate_decisions:
            generation = int(writer.mutable_generation_id)
            self.evidence_store.append_evidence_batch(
                EvidenceRecord(
                    memory_id=d.memory_id,
                    evidence_type=int(EvidenceType.GATE_VALIDATION),
                    generation_id=generation,
                    payload={
                        "gate_id": int(d.gate_id),
                        "trials": int(d.trials),
                        "successes": int(d.successes),
                        "independent_targets": int(d.independent_targets),
                        "mean_causal_gain": float(d.mean_causal_gain),
                        "structural_candidate": bool(d.structural_candidate),
                        "probe_eligible": bool(d.probe_eligible),
                        "tested": bool(d.tested),
                        "validated": bool(d.validated),
                        "trusted": bool(d.trusted),
                        "rejected": bool(d.rejected),
                        "previous_validation_state": int(d.previous_validation_state),
                        "next_validation_state": int(d.next_validation_state),
                        "previous_cognitive_state": int(d.previous_cognitive_state),
                        "next_cognitive_state": int(d.next_cognitive_state),
                    },
                )
                for d in gate_decisions
            )
        return DevelopmentalLifecycleResult(
            lifecycle_decisions,
            stats,
            concept_decisions,
            concept_mutations,
            profile.stage.name,
            replay_mutations,
            gate_decisions,
            gate_mutations,
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
