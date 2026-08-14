from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Iterable, Mapping

from v7.memory.canonical import CanonicalCandidateMutation, CanonicalMemoryKey
from v7.memory.ids import MemoryId, MemoryLevel

TYPE_CONTINGENCY = 100
TYPE_FAMILY = 200
TYPE_ROLE = 300
TYPE_CONTEXTUAL_ROLE = 301
TYPE_CARRIER = 302
TYPE_CONCEPT = 400
TYPE_WORLD_MODEL = 500
TYPE_STRATEGY = 600

_MASK63 = (1 << 63) - 1


def world_transition_signature(
    prior: Iterable[int], action_id: int, current: Iterable[int]
) -> int:
    prior_values = tuple(int(value) for value in prior)
    current_values = tuple(int(value) for value in current)
    digest = blake2b(digest_size=8)
    digest.update(b"world-transition-v3")
    digest.update(str(prior_values).encode("ascii"))
    digest.update(str(int(action_id)).encode("ascii"))
    digest.update(str(current_values).encode("ascii"))
    return int.from_bytes(digest.digest(), "little") & _MASK63


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
    carrier_signature: int | None = None
    decision_role_ids: tuple[int, ...] = ()
    decision_concept_ids: tuple[int, ...] = ()
    terminal_polarity: int = 0
    raw_action_option_delta: float = 0.0
    decision_score: float = 0.0
    max_action_score: float = 0.0
    memory_guided: bool = False


class ScientificDerivationKernels:
    @staticmethod
    def m1_from_episode(evidence: EpisodeEvidence) -> CanonicalCandidateMutation:
        key = CanonicalMemoryKey(
            MemoryLevel.M1,
            TYPE_CONTINGENCY,
            (
                int(evidence.context_signature),
                int(evidence.action_id),
                int(evidence.outcome_signature),
            ),
        )
        polarity = int(evidence.terminal_polarity)
        significance = (
            1.0
            if polarity > 0
            else 0.0
            if polarity < 0
            else 0.5
            if evidence.success
            else 0.25
        )
        return CanonicalCandidateMutation(
            key=key,
            support_delta=1,
            significance=significance,
            prediction_error=max(0.0, float(evidence.prediction_error)),
            learning_value=max(0.0, float(evidence.prediction_error)),
            future_option_delta=float(evidence.future_option_delta),
        )

    @staticmethod
    def m2_family(
        *, action_id: int, member_ids: Iterable[MemoryId], outcome_class: int
    ) -> CanonicalCandidateMutation:
        """Create an action-independent transformation family.

        ``action_id`` remains in the API because callers naturally discover the
        family while grouping action-conditioned M1 evidence, but it is not
        part of canonical M2 identity. This allows the same normalized
        transformation to generalize across distinct actions.
        """
        del action_id
        members = tuple(sorted(set(member_ids), key=int))
        if not members:
            raise ValueError("M2 family requires at least one M1 member")
        key = CanonicalMemoryKey(
            MemoryLevel.M2,
            TYPE_FAMILY,
            (int(outcome_class),),
        )
        return CanonicalCandidateMutation(
            key=key,
            support_delta=len(members),
            parents=members,
            transfer_prior=min(1.0, len(members) / 4.0),
        )

    @staticmethod
    def m3_carrier(
        *,
        carrier_signature: int,
        parent_ids: Iterable[MemoryId],
        support_count: int,
        prediction_lift: float,
        compression_gain: float,
    ) -> CanonicalCandidateMutation:
        parents = tuple(sorted(set(parent_ids), key=int))
        if not parents or int(support_count) < 1:
            raise ValueError("carrier requires supporting lower-level evidence")
        key = CanonicalMemoryKey(
            MemoryLevel.M3,
            TYPE_CARRIER,
            (int(carrier_signature),),
        )
        return CanonicalCandidateMutation(
            key=key,
            support_delta=int(support_count),
            parents=parents,
            learning_value=max(0.0, min(1.0, float(prediction_lift))),
            explanatory_potential=max(0.0, min(1.0, float(compression_gain))),
        )

    @staticmethod
    def m3_contextual_role(
        *,
        family_id: MemoryId,
        carrier_id: MemoryId,
        context_class: int,
        action_id: int,
        member_ids: Iterable[MemoryId],
    ) -> CanonicalCandidateMutation:
        members = tuple(sorted(set(member_ids), key=int))
        if not members:
            raise ValueError("contextual role requires supporting members")
        key = CanonicalMemoryKey(
            MemoryLevel.M3,
            TYPE_CONTEXTUAL_ROLE,
            (
                int(family_id),
                int(carrier_id),
                int(action_id),
                int(context_class),
            ),
        )
        return CanonicalCandidateMutation(
            key=key,
            support_delta=len(members),
            parents=(family_id, carrier_id, *members),
            explanatory_potential=min(1.0, len(members) / 4.0),
        )

    @staticmethod
    def m3_functional_role(
        *,
        function_signature: int,
        instance_ids: Iterable[MemoryId],
        support_count: int | None = None,
        transfer_prior: float = 0.0,
        explanatory_potential: float = 0.0,
    ) -> CanonicalCandidateMutation:
        instances = tuple(sorted(set(instance_ids), key=int))
        if not instances:
            raise ValueError("functional role requires contextual role instances")
        support = len(instances) if support_count is None else max(1, int(support_count))
        key = CanonicalMemoryKey(
            MemoryLevel.M3,
            TYPE_ROLE,
            (int(function_signature),),
        )
        return CanonicalCandidateMutation(
            key=key,
            support_delta=support,
            parents=instances,
            transfer_prior=max(0.0, min(1.0, float(transfer_prior))),
            explanatory_potential=max(
                0.0, min(1.0, float(explanatory_potential))
            ),
        )

    @staticmethod
    def m3_role(
        *,
        family_id: MemoryId,
        context_class: int,
        action_id: int,
        member_ids: Iterable[MemoryId],
    ) -> CanonicalCandidateMutation:
        """Backward-compatible kernel for direct tests/callers.

        Production online derivation uses ``m3_contextual_role`` followed by
        ``m3_functional_role``. This helper keeps the older public kernel shape
        while producing a deterministic functional-role key.
        """
        members = tuple(sorted(set(member_ids), key=int))
        if not members:
            raise ValueError("M3 role requires supporting members")
        signature = _support_signature(
            tuple(
                MemoryId(int(value))
                for value in (
                    int(family_id),
                    int(action_id),
                    int(context_class),
                )
            )
        )
        key = CanonicalMemoryKey(MemoryLevel.M3, TYPE_ROLE, (signature,))
        return CanonicalCandidateMutation(
            key=key,
            support_delta=len(members),
            parents=(family_id, *members),
            explanatory_potential=min(1.0, len(members) / 4.0),
        )

    @staticmethod
    def m4_concept(
        *, role_ids: Iterable[MemoryId], relation_signature: int
    ) -> CanonicalCandidateMutation:
        roles = tuple(sorted(set(role_ids), key=int))
        if len(roles) < 2:
            raise ValueError("M4 concept requires at least two roles")
        key = CanonicalMemoryKey(
            MemoryLevel.M4,
            TYPE_CONCEPT,
            (int(relation_signature),),
        )
        return CanonicalCandidateMutation(
            key=key,
            support_delta=len(roles),
            parents=roles,
            explanatory_potential=min(1.0, len(roles) / 3.0),
            transfer_prior=min(1.0, len(roles) / 5.0),
        )

    @staticmethod
    def m5_world_model(
        *, concept_ids: Iterable[MemoryId], transition_signature: int
    ) -> CanonicalCandidateMutation:
        concepts = tuple(sorted(set(concept_ids), key=int))
        if len(concepts) < 2:
            raise ValueError("M5 world model requires at least two concepts")
        key = CanonicalMemoryKey(
            MemoryLevel.M5,
            TYPE_WORLD_MODEL,
            (int(transition_signature),),
        )
        return CanonicalCandidateMutation(
            key=key,
            support_delta=len(concepts),
            parents=concepts,
            explanatory_potential=min(1.0, 0.25 * len(concepts)),
            transfer_prior=min(1.0, 0.20 * len(concepts)),
        )

    @staticmethod
    def m6_strategy(
        *,
        world_model_ids: Iterable[MemoryId],
        action_signature: int,
        efficiency_gain: float,
    ) -> CanonicalCandidateMutation:
        models = tuple(sorted(set(world_model_ids), key=int))
        if not models:
            raise ValueError("M6 strategy requires world-model support")
        gain = float(efficiency_gain)
        key = CanonicalMemoryKey(
            MemoryLevel.M6,
            TYPE_STRATEGY,
            (int(action_signature), _support_signature(models)),
        )
        return CanonicalCandidateMutation(
            key=key,
            support_delta=len(models),
            parents=models,
            significance=max(0.0, gain),
            learning_value=max(0.0, gain),
            future_option_delta=gain,
        )

    @classmethod
    def derive_level(
        cls, level: MemoryLevel, payload: Mapping[str, object]
    ) -> CanonicalCandidateMutation:
        if level == MemoryLevel.M2:
            return cls.m2_family(
                action_id=int(payload.get("action_id", 0)),
                member_ids=payload["member_ids"],
                outcome_class=int(payload["outcome_class"]),
            )
        if level == MemoryLevel.M3:
            return cls.m3_role(
                family_id=payload["family_id"],
                context_class=int(payload["context_class"]),
                action_id=int(payload["action_id"]),
                member_ids=payload["member_ids"],
            )
        if level == MemoryLevel.M4:
            return cls.m4_concept(
                role_ids=payload["role_ids"],
                relation_signature=int(payload["relation_signature"]),
            )
        if level == MemoryLevel.M5:
            return cls.m5_world_model(
                concept_ids=payload["concept_ids"],
                transition_signature=int(payload["transition_signature"]),
            )
        if level == MemoryLevel.M6:
            return cls.m6_strategy(
                world_model_ids=payload["world_model_ids"],
                action_signature=int(payload["action_signature"]),
                efficiency_gain=float(payload["efficiency_gain"]),
            )
        raise ValueError("derive_level targets M2-M6")


def _support_signature(memory_ids: tuple[MemoryId, ...]) -> int:
    value = 1469598103934665603
    for memory_id in memory_ids:
        value ^= int(memory_id)
        value = (value * 1099511628211) & ((1 << 63) - 1)
    return value
