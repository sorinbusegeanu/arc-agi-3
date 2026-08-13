from __future__ import annotations

from typing import Iterable

from v7.derivation.scientific import (
    EpisodeEvidence,
    ScientificDerivationKernels,
    TYPE_CONTINGENCY,
    TYPE_ROLE,
)
from v7.memory.canonical import CanonicalCandidateMutation, CanonicalMemoryKey
from v7.memory.evidence_lifecycle import (
    ContradictionRecord,
    EvidenceLifecycleStore,
    ProvenanceRecord,
    TransferTrialRecord,
)
from v7.memory.evidence_store import EvidenceRecord, EvidenceStore
from v7.memory.evidence_types import EvidenceType
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.indexes.cognition import (
    ActionAggregateDelta,
    ContingencyIndexMutation,
    RoleConceptIndexMutation,
    RoleIndexMutation,
)
from v7.memory.writer import CanonicalMemoryWriter


class MemoryLearningPipeline:
    """Evidence-to-memory pipeline with multi-scale contextual M1 learning."""

    def __init__(
        self,
        writer: CanonicalMemoryWriter,
        evidence_lifecycle: EvidenceLifecycleStore | None = None,
        evidence_store: EvidenceStore | None = None,
    ) -> None:
        self.writer = writer
        self.evidence_lifecycle = evidence_lifecycle
        self.evidence_store = evidence_store

    def _record_parents(
        self, memory_id: MemoryId, parents: Iterable[MemoryId]
    ) -> None:
        if self.evidence_lifecycle is None:
            return
        generation_id = int(self.writer.mutable_generation_id)
        existing = set(self.evidence_lifecycle.provenance_parents(memory_id))
        rows = tuple(
            ProvenanceRecord(
                memory_id=memory_id,
                parent_memory_id=parent_id,
                generation_id=generation_id,
            )
            for parent_id in sorted(set(parents) - existing, key=int)
        )
        if rows:
            self.evidence_lifecycle.append_provenance(rows)

    @staticmethod
    def _contexts(evidence: EpisodeEvidence) -> tuple[int, ...]:
        values = tuple(
            int(value)
            for value in getattr(evidence, "context_signatures", ()) or ()
        )
        return values or (int(evidence.context_signature),)

    @staticmethod
    def _m1_candidate(
        evidence: EpisodeEvidence, context_signature: int
    ) -> CanonicalCandidateMutation:
        polarity = int(getattr(evidence, "terminal_polarity", 0) or 0)
        if polarity > 0:
            significance = 1.0
        elif polarity < 0:
            significance = 0.0
        else:
            significance = 0.5 if bool(evidence.success) else 0.25
        key = CanonicalMemoryKey(
            MemoryLevel.M1,
            TYPE_CONTINGENCY,
            (
                int(context_signature),
                int(evidence.action_id),
                int(evidence.outcome_signature),
            ),
        )
        return CanonicalCandidateMutation(
            key=key,
            support_delta=1,
            significance=significance,
            prediction_error=max(0.0, float(evidence.prediction_error)),
            learning_value=max(0.0, float(evidence.prediction_error)),
            future_option_delta=float(evidence.future_option_delta),
        )

    def observe_episode(self, evidence: EpisodeEvidence) -> MemoryId:
        return self.observe_batch((evidence,))[0]

    def observe_batch(self, rows: Iterable[EpisodeEvidence]) -> tuple[MemoryId, ...]:
        episodes = tuple(rows)
        if not episodes:
            return ()

        expanded: list[tuple[EpisodeEvidence, int, CanonicalCandidateMutation]] = []
        for evidence in episodes:
            for context in self._contexts(evidence):
                expanded.append(
                    (evidence, context, self._m1_candidate(evidence, context))
                )
        resolved = self.writer.apply_canonical_candidate_batch(
            candidate for _evidence, _context, candidate in expanded
        )
        self.writer.apply_contingency_index_batch(
            ContingencyIndexMutation(
                context,
                int(evidence.action_id),
                resolved[candidate.key],
            )
            for evidence, context, candidate in expanded
        )

        memory_ids: list[MemoryId] = []
        for evidence in episodes:
            context = self._contexts(evidence)[-1]
            memory_ids.append(
                resolved[self._m1_candidate(evidence, context).key]
            )

        self.writer.apply_action_aggregate_batch(
            ActionAggregateDelta(
                action_id=int(evidence.action_id),
                future_option_sum_delta=float(evidence.future_option_delta),
                future_option_count_delta=1,
                positive_count_delta=1
                if int(getattr(evidence, "terminal_polarity", 0) or 0) > 0
                else 0,
                negative_count_delta=1
                if int(getattr(evidence, "terminal_polarity", 0) or 0) < 0
                else 0,
                failure_count_delta=1
                if int(getattr(evidence, "terminal_polarity", 0) or 0) < 0
                else 0,
                contradiction_count_delta=1
                if float(evidence.prediction_error) > 0
                else 0,
            )
            for evidence in episodes
        )

        generation_id = int(self.writer.mutable_generation_id)
        if self.evidence_lifecycle is not None:
            self.evidence_lifecycle.append_provenance(
                ProvenanceRecord(
                    memory_id=memory_id,
                    generation_id=generation_id,
                    source_game=evidence.source_game,
                    source_context=evidence.source_context,
                    source_global_step=evidence.source_global_step,
                )
                for evidence, memory_id in zip(
                    episodes, memory_ids, strict=True
                )
            )
            self.evidence_lifecycle.append_contradictions(
                ContradictionRecord(
                    memory_id=memory_id,
                    generation_id=generation_id,
                    severity=float(evidence.prediction_error),
                    source_game=evidence.source_game,
                    source_context=evidence.source_context,
                    source_global_step=evidence.source_global_step,
                    payload={
                        "action_id": int(evidence.action_id),
                        "outcome_signature": int(evidence.outcome_signature),
                    },
                )
                for evidence, memory_id in zip(
                    episodes, memory_ids, strict=True
                )
                if float(evidence.prediction_error) > 0
            )

        if self.evidence_store is not None:
            self.evidence_store.append_evidence_batch(
                EvidenceRecord(
                    memory_id=memory_id,
                    evidence_type=int(EvidenceType.EPISODE),
                    generation_id=generation_id,
                    source_game=evidence.source_game,
                    source_context=evidence.source_context,
                    source_global_step=evidence.source_global_step,
                    payload={
                        "context_signature": int(evidence.context_signature),
                        "context_signatures": list(self._contexts(evidence)),
                        "next_context_signatures": [
                            int(value)
                            for value in getattr(
                                evidence, "next_context_signatures", ()
                            )
                            or ()
                        ],
                        "exact_context_signature": getattr(
                            evidence, "exact_context_signature", None
                        ),
                        "structural_context_signature": getattr(
                            evidence, "structural_context_signature", None
                        ),
                        "action_id": int(evidence.action_id),
                        "outcome_signature": int(evidence.outcome_signature),
                        "raw_transition_signature": getattr(
                            evidence, "raw_transition_signature", None
                        ),
                        "success": bool(evidence.success),
                        "prediction_error": float(evidence.prediction_error),
                        "future_option_delta": float(
                            evidence.future_option_delta
                        ),
                        "raw_action_option_delta": float(
                            evidence.raw_action_option_delta
                        ),
                        "carrier_signature": None
                        if evidence.carrier_signature is None
                        else int(evidence.carrier_signature),
                        "decision_role_ids": [
                            int(value) for value in evidence.decision_role_ids
                        ],
                        "decision_concept_ids": [
                            int(value) for value in evidence.decision_concept_ids
                        ],
                        "decision_world_model_ids": [
                            int(value)
                            for value in getattr(
                                evidence, "decision_world_model_ids", ()
                            )
                            or ()
                        ],
                        "decision_strategy_ids": [
                            int(value)
                            for value in getattr(
                                evidence, "decision_strategy_ids", ()
                            )
                            or ()
                        ],
                        "terminal_polarity": int(evidence.terminal_polarity),
                        "changed_cells": int(
                            getattr(evidence, "changed_cells", 0) or 0
                        ),
                        "decision_score": float(evidence.decision_score),
                        "max_action_score": float(evidence.max_action_score),
                        "memory_guided": bool(evidence.memory_guided),
                    },
                )
                for evidence, memory_id in zip(
                    episodes, memory_ids, strict=True
                )
            )

        if self.evidence_lifecycle is not None:
            transfer_records: list[TransferTrialRecord] = []
            for evidence in episodes:
                if evidence.terminal_polarity != 0 and evidence.source_game:
                    transfer_records.extend(
                        self._transfer_records(
                            evidence, target_game=evidence.source_game
                        )
                    )
            self.evidence_lifecycle.append_transfer_trials(transfer_records)
        return tuple(memory_ids)

    def _transfer_records(
        self, evidence: EpisodeEvidence, *, target_game: str
    ) -> list[TransferTrialRecord]:
        if self.evidence_lifecycle is None:
            return []
        records: list[TransferTrialRecord] = []
        ids = tuple(
            sorted(
                set(
                    (
                        *evidence.decision_role_ids,
                        *evidence.decision_concept_ids,
                    )
                )
            )
        )
        for raw_memory_id in ids:
            memory_id = MemoryId(int(raw_memory_id))
            source_games = tuple(
                game
                for game in self.evidence_lifecycle.provenance_source_games(
                    memory_id
                )
                if game != target_game
            )
            if not source_games:
                continue
            if self.evidence_lifecycle.transfer_trial_exists(
                memory_id,
                target_game=target_game,
                source_global_step=evidence.source_global_step,
            ):
                continue
            records.append(
                TransferTrialRecord(
                    memory_id=memory_id,
                    generation_id=int(self.writer.mutable_generation_id),
                    source_game=source_games[0],
                    target_game=target_game,
                    success=evidence.terminal_polarity > 0,
                    score=1.0 if evidence.terminal_polarity > 0 else 0.0,
                    payload={
                        "source_games": list(source_games),
                        "source_game_count": len(source_games),
                        "target_context": evidence.source_context,
                        "source_global_step": evidence.source_global_step,
                        "future_option_delta": float(
                            evidence.future_option_delta
                        ),
                        "carrier_signature": None
                        if evidence.carrier_signature is None
                        else int(evidence.carrier_signature),
                    },
                )
            )
        return records

    def derive_m2(
        self,
        *,
        action_id: int,
        member_ids: Iterable[MemoryId],
        outcome_class: int,
    ) -> MemoryId:
        members = tuple(member_ids)
        candidate = ScientificDerivationKernels.m2_family(
            action_id=action_id,
            member_ids=members,
            outcome_class=outcome_class,
        )
        memory_id = self.writer.apply_canonical_candidate_batch((candidate,))[
            candidate.key
        ]
        self._record_parents(memory_id, candidate.parents)
        return memory_id

    def derive_m3(
        self,
        *,
        family_id: MemoryId,
        context_class: int,
        action_id: int,
        member_ids: Iterable[MemoryId],
    ) -> MemoryId:
        members = tuple(sorted(set(member_ids), key=int))
        if not members:
            raise ValueError("M3 role requires supporting members")
        key = CanonicalMemoryKey(
            MemoryLevel.M3,
            TYPE_ROLE,
            (int(family_id), int(action_id), int(context_class)),
        )
        candidate = CanonicalCandidateMutation(
            key=key,
            support_delta=len(members),
            parents=(family_id, *members),
            explanatory_potential=min(1.0, len(members) / 4.0),
        )
        memory_id = self.writer.apply_canonical_candidate_batch((candidate,))[key]
        self.writer.apply_role_index_batch(
            (
                RoleIndexMutation(
                    int(context_class), int(action_id), memory_id, family_id
                ),
            )
        )
        self._record_parents(memory_id, candidate.parents)
        return memory_id

    def derive_m4(
        self, *, role_ids: Iterable[MemoryId], relation_signature: int
    ) -> MemoryId:
        roles = tuple(role_ids)
        candidate = ScientificDerivationKernels.m4_concept(
            role_ids=roles, relation_signature=relation_signature
        )
        memory_id = self.writer.apply_canonical_candidate_batch((candidate,))[
            candidate.key
        ]
        self.writer.apply_role_concept_index_batch(
            tuple(
                RoleConceptIndexMutation(role_id, memory_id)
                for role_id in sorted(set(roles), key=int)
            )
        )
        self._record_parents(memory_id, candidate.parents)
        return memory_id

    def derive_m5(
        self,
        *,
        concept_ids: Iterable[MemoryId],
        transition_signature: int,
    ) -> MemoryId:
        concepts = tuple(concept_ids)
        candidate = ScientificDerivationKernels.m5_world_model(
            concept_ids=concepts, transition_signature=transition_signature
        )
        memory_id = self.writer.apply_canonical_candidate_batch((candidate,))[
            candidate.key
        ]
        self._record_parents(memory_id, candidate.parents)
        return memory_id

    def derive_m6(
        self,
        *,
        world_model_ids: Iterable[MemoryId],
        action_signature: int,
        efficiency_gain: float,
    ) -> MemoryId:
        models = tuple(world_model_ids)
        candidate = ScientificDerivationKernels.m6_strategy(
            world_model_ids=models,
            action_signature=action_signature,
            efficiency_gain=efficiency_gain,
        )
        memory_id = self.writer.apply_canonical_candidate_batch((candidate,))[
            candidate.key
        ]
        self._record_parents(memory_id, candidate.parents)
        return memory_id
