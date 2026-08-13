from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from v7.memory.canonical import CanonicalCandidateMutation, CanonicalMemoryKey
from v7.memory.ids import MemoryId, MemoryLevel

TYPE_CONTINGENCY = 100
TYPE_FAMILY = 200
TYPE_ROLE = 300
TYPE_CONCEPT = 400
TYPE_WORLD_MODEL = 500
TYPE_STRATEGY = 600


@dataclass(frozen=True, slots=True)
class EpisodeEvidence:
    context_signature: int
    action_id: int
    outcome_signature: int
    success: bool
    prediction_error: float = 0.0
    future_option_delta: float = 0.0
    source_game: str | None = None
    source_context: str | None = None
    source_global_step: int | None = None


class ScientificDerivationKernels:
    """Deterministic clean-break M1-M6 semantic constructors.

    Canonical identities represent abstractions, while supporting MemoryIds remain
    provenance/support. New evidence therefore does not create a new family, concept
    or world-model identity solely because its support population grew.
    """

    @staticmethod
    def m1_from_episode(evidence: EpisodeEvidence) -> CanonicalCandidateMutation:
        key = CanonicalMemoryKey(MemoryLevel.M1, TYPE_CONTINGENCY, (int(evidence.context_signature), int(evidence.action_id), int(evidence.outcome_signature)))
        return CanonicalCandidateMutation(
            key=key,
            support_delta=1,
            significance=1.0 if evidence.success else 0.5,
            prediction_error=max(0.0, float(evidence.prediction_error)),
            learning_value=max(0.0, float(evidence.prediction_error)),
            future_option_delta=float(evidence.future_option_delta),
        )

    @staticmethod
    def m2_family(*, action_id: int, member_ids: Iterable[MemoryId], outcome_class: int) -> CanonicalCandidateMutation:
        members = tuple(sorted(set(member_ids), key=int))
        if not members:
            raise ValueError("M2 family requires at least one M1 member")
        key = CanonicalMemoryKey(MemoryLevel.M2, TYPE_FAMILY, (int(action_id), int(outcome_class)))
        return CanonicalCandidateMutation(key=key, support_delta=len(members), parents=members, transfer_prior=min(1.0, len(members) / 4.0))

    @staticmethod
    def m3_role(*, family_id: MemoryId, context_class: int, action_id: int, member_ids: Iterable[MemoryId]) -> CanonicalCandidateMutation:
        members = tuple(sorted(set(member_ids), key=int))
        if not members:
            raise ValueError("M3 role requires supporting members")
        key = CanonicalMemoryKey(MemoryLevel.M3, TYPE_ROLE, (int(family_id), int(context_class), int(action_id)))
        return CanonicalCandidateMutation(key=key, support_delta=len(members), parents=(family_id, *members), explanatory_potential=min(1.0, len(members) / 4.0))

    @staticmethod
    def m4_concept(*, role_ids: Iterable[MemoryId], relation_signature: int) -> CanonicalCandidateMutation:
        roles = tuple(sorted(set(role_ids), key=int))
        if len(roles) < 2:
            raise ValueError("M4 concept requires at least two roles")
        key = CanonicalMemoryKey(MemoryLevel.M4, TYPE_CONCEPT, (int(relation_signature),))
        return CanonicalCandidateMutation(key=key, support_delta=len(roles), parents=roles, explanatory_potential=min(1.0, len(roles) / 3.0), transfer_prior=min(1.0, len(roles) / 5.0))

    @staticmethod
    def m5_world_model(*, concept_ids: Iterable[MemoryId], transition_signature: int) -> CanonicalCandidateMutation:
        concepts = tuple(sorted(set(concept_ids), key=int))
        if len(concepts) < 2:
            raise ValueError("M5 world model requires at least two concepts")
        key = CanonicalMemoryKey(MemoryLevel.M5, TYPE_WORLD_MODEL, (int(transition_signature),))
        return CanonicalCandidateMutation(key=key, support_delta=len(concepts), parents=concepts, explanatory_potential=min(1.0, 0.25 * len(concepts)), transfer_prior=min(1.0, 0.20 * len(concepts)))

    @staticmethod
    def m6_strategy(*, world_model_ids: Iterable[MemoryId], action_signature: int, efficiency_gain: float) -> CanonicalCandidateMutation:
        models = tuple(sorted(set(world_model_ids), key=int))
        if not models:
            raise ValueError("M6 strategy requires world-model support")
        gain = float(efficiency_gain)
        key = CanonicalMemoryKey(MemoryLevel.M6, TYPE_STRATEGY, (int(action_signature), _support_signature(models)))
        return CanonicalCandidateMutation(key=key, support_delta=len(models), parents=models, significance=max(0.0, gain), learning_value=max(0.0, gain), future_option_delta=gain)

    @classmethod
    def derive_level(cls, level: MemoryLevel, payload: Mapping[str, object]) -> CanonicalCandidateMutation:
        if level == MemoryLevel.M2:
            return cls.m2_family(action_id=int(payload["action_id"]), member_ids=payload["member_ids"], outcome_class=int(payload["outcome_class"]))
        if level == MemoryLevel.M3:
            return cls.m3_role(family_id=payload["family_id"], context_class=int(payload["context_class"]), action_id=int(payload["action_id"]), member_ids=payload["member_ids"])
        if level == MemoryLevel.M4:
            return cls.m4_concept(role_ids=payload["role_ids"], relation_signature=int(payload["relation_signature"]))
        if level == MemoryLevel.M5:
            return cls.m5_world_model(concept_ids=payload["concept_ids"], transition_signature=int(payload["transition_signature"]))
        if level == MemoryLevel.M6:
            return cls.m6_strategy(world_model_ids=payload["world_model_ids"], action_signature=int(payload["action_signature"]), efficiency_gain=float(payload["efficiency_gain"]))
        raise ValueError("derive_level targets M2-M6")


def _support_signature(memory_ids: tuple[MemoryId, ...]) -> int:
    value = 1469598103934665603
    for memory_id in memory_ids:
        value ^= int(memory_id)
        value = (value * 1099511628211) & ((1 << 63) - 1)
    return value
