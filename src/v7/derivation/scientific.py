from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from v7.memory.canonical import CanonicalMemoryKey
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


@dataclass(frozen=True, slots=True)
class SemanticCandidate:
    key: CanonicalMemoryKey
    support: int
    parents: tuple[MemoryId, ...] = ()
    significance: float = 0.0
    prediction_error: float = 0.0
    learning_value: float = 0.0
    transfer_prior: float = 0.0
    explanatory_potential: float = 0.0
    future_option_delta: float = 0.0


class ScientificDerivationKernels:
    """Deterministic clean-break M1-M6 semantic constructors.

    These kernels encode the v0.3 hierarchy without importing v6 runtime code.
    They operate on canonical structural signatures so parallel workers can emit
    duplicate candidates safely and the writer can resolve them deterministically.
    """

    @staticmethod
    def m1_from_episode(evidence: EpisodeEvidence) -> SemanticCandidate:
        key = CanonicalMemoryKey(
            MemoryLevel.M1,
            TYPE_CONTINGENCY,
            (int(evidence.context_signature), int(evidence.action_id), int(evidence.outcome_signature)),
        )
        return SemanticCandidate(
            key=key,
            support=1,
            significance=1.0 if evidence.success else 0.5,
            prediction_error=max(0.0, float(evidence.prediction_error)),
            learning_value=max(0.0, float(evidence.prediction_error)),
            future_option_delta=float(evidence.future_option_delta),
        )

    @staticmethod
    def m2_family(*, action_id: int, member_ids: Iterable[MemoryId], outcome_class: int) -> SemanticCandidate:
        members = tuple(sorted(set(member_ids), key=int))
        if not members:
            raise ValueError("M2 family requires at least one M1 member")
        key = CanonicalMemoryKey(MemoryLevel.M2, TYPE_FAMILY, (int(action_id), int(outcome_class), *(int(v) for v in members)))
        return SemanticCandidate(key=key, support=len(members), parents=members, transfer_prior=min(1.0, len(members) / 4.0))

    @staticmethod
    def m3_role(*, family_id: MemoryId, context_class: int, action_id: int, member_ids: Iterable[MemoryId]) -> SemanticCandidate:
        members = tuple(sorted(set(member_ids), key=int))
        if not members:
            raise ValueError("M3 role requires supporting members")
        key = CanonicalMemoryKey(MemoryLevel.M3, TYPE_ROLE, (int(family_id), int(context_class), int(action_id)))
        return SemanticCandidate(key=key, support=len(members), parents=(family_id, *members), explanatory_potential=min(1.0, len(members) / 4.0))

    @staticmethod
    def m4_concept(*, role_ids: Iterable[MemoryId], relation_signature: int) -> SemanticCandidate:
        roles = tuple(sorted(set(role_ids), key=int))
        if len(roles) < 2:
            raise ValueError("M4 concept requires at least two roles")
        key = CanonicalMemoryKey(MemoryLevel.M4, TYPE_CONCEPT, (int(relation_signature), *(int(v) for v in roles)))
        return SemanticCandidate(key=key, support=len(roles), parents=roles, explanatory_potential=min(1.0, len(roles) / 3.0), transfer_prior=min(1.0, len(roles) / 5.0))

    @staticmethod
    def m5_world_model(*, concept_ids: Iterable[MemoryId], transition_signature: int) -> SemanticCandidate:
        concepts = tuple(sorted(set(concept_ids), key=int))
        if len(concepts) < 2:
            raise ValueError("M5 world model requires at least two concepts")
        key = CanonicalMemoryKey(MemoryLevel.M5, TYPE_WORLD_MODEL, (int(transition_signature), *(int(v) for v in concepts)))
        return SemanticCandidate(key=key, support=len(concepts), parents=concepts, explanatory_potential=min(1.0, 0.25 * len(concepts)), transfer_prior=min(1.0, 0.20 * len(concepts)))

    @staticmethod
    def m6_strategy(*, world_model_ids: Iterable[MemoryId], action_signature: int, efficiency_gain: float) -> SemanticCandidate:
        models = tuple(sorted(set(world_model_ids), key=int))
        if not models:
            raise ValueError("M6 strategy requires world-model support")
        gain = float(efficiency_gain)
        key = CanonicalMemoryKey(MemoryLevel.M6, TYPE_STRATEGY, (int(action_signature), *(int(v) for v in models)))
        return SemanticCandidate(key=key, support=len(models), parents=models, significance=max(0.0, gain), learning_value=max(0.0, gain), future_option_delta=gain)

    @classmethod
    def derive_level(cls, level: MemoryLevel, payload: Mapping[str, object]) -> SemanticCandidate:
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
