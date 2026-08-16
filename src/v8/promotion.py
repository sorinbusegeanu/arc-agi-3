from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from v8.arena import EdgeRecord, NodeRecord
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    stable_u64,
)


@dataclass(frozen=True, slots=True)
class FormationCandidate:
    uid: MemoryUid
    level: MemoryLevel
    memory_type: MemoryType
    key_parts: tuple[int, ...]
    parents: tuple[MemoryUid, ...]
    support: int
    significance: float
    learning_value: float
    transfer_prior: float
    explanatory_reach: float
    future_option_delta: float
    cognitive_state: int
    validation_state: int
    evidence_kind: str
    evidence_value: float


class EvidenceGatedPromotionEngine:
    """Derive M2-M7 only from accumulated canonical lower-level state.

    Raw events are intentionally absent from this API.  Every candidate is justified
    by already-published lower-level memories and therefore has explicit provenance.
    """

    def __init__(
        self,
        *,
        min_contingency_support: int = 3,
        min_family_members: int = 2,
        min_family_compression: float = 1.0,
        min_carrier_family_support: int = 3,
        min_carrier_persistence: int = 2,
        min_concept_explanatory: float = 1.0,
        min_concept_transfer_prior: float = 0.25,
    ) -> None:
        self.min_contingency_support = int(min_contingency_support)
        self.min_family_members = int(min_family_members)
        self.min_family_compression = float(min_family_compression)
        self.min_carrier_family_support = int(min_carrier_family_support)
        self.min_carrier_persistence = int(min_carrier_persistence)
        self.min_concept_explanatory = float(min_concept_explanatory)
        self.min_concept_transfer_prior = float(min_concept_transfer_prior)

    @staticmethod
    def _future_bucket(value: float) -> int:
        return 1 if float(value) > 1e-9 else -1 if float(value) < -1e-9 else 0

    @staticmethod
    def _admissible(row: NodeRecord) -> bool:
        return int(row.cognitive_state) not in {
            int(CognitiveState.QUARANTINED),
            int(CognitiveState.RETIRE_PENDING),
            int(CognitiveState.RETIRED),
        }

    @staticmethod
    def _children(
        edges: tuple[EdgeRecord, ...],
        *,
        relation_types: tuple[RelationType, ...] = (RelationType.EXPLAINS,),
    ) -> dict[MemoryUid, tuple[MemoryUid, ...]]:
        allowed = {int(value) for value in relation_types}
        grouped: dict[MemoryUid, set[MemoryUid]] = defaultdict(set)
        for edge in edges:
            if int(edge.relation_type) in allowed:
                grouped[edge.source_uid].add(edge.target_uid)
        return {uid: tuple(sorted(values)) for uid, values in grouped.items()}

    def propose(
        self,
        nodes: tuple[NodeRecord, ...],
        edges: tuple[EdgeRecord, ...],
        *,
        budget: int = 256,
    ) -> tuple[FormationCandidate, ...]:
        limit = max(0, int(budget))
        if limit == 0:
            return ()
        by_uid = {row.uid: row for row in nodes}
        children = self._children(edges)
        result: list[FormationCandidate] = []

        # M2: multiple established M1 contingencies with a common declared
        # action/transformation bucket.  The raw event's family signature is not
        # consulted here.
        m1_groups: dict[tuple[int, int], list[NodeRecord]] = defaultdict(list)
        stable_m1 = [
            row
            for row in nodes
            if int(row.level) == int(MemoryLevel.M1)
            and int(row.memory_type) == int(MemoryType.CONTINGENCY)
            and row.support_count >= self.min_contingency_support
            and len(row.key_parts) >= 4
            and self._admissible(row)
        ]
        for row in stable_m1:
            action_token = int(row.key_parts[1])
            transformation_bucket = int(row.key_parts[2]) & 0xFFFF
            m1_groups[(action_token, transformation_bucket)].append(row)
        for key, members in sorted(m1_groups.items()):
            if len(members) < self.min_family_members:
                continue
            total_support = sum(max(0, int(row.support_count)) for row in members)
            compression = float(total_support - len(members))
            if compression <= self.min_family_compression:
                continue
            consistency = min(1.0, total_support / max(1.0, 2.0 * len(members)))
            uid = MemoryUid.from_key(MemoryLevel.M2, MemoryType.FAMILY, key)
            result.append(
                FormationCandidate(
                    uid,
                    MemoryLevel.M2,
                    MemoryType.FAMILY,
                    key,
                    tuple(sorted(row.uid for row in members)),
                    total_support,
                    consistency,
                    min(1.0, compression / max(1.0, total_support)),
                    0.0,
                    float(len(members)),
                    sum(row.future_option_delta * row.support_count for row in members)
                    / max(1, total_support),
                    int(CognitiveState.PROBATION),
                    int(ValidationState.STRUCTURAL),
                    "family_compression",
                    min(1.0, compression / max(1.0, total_support)),
                )
            )
            if len(result) >= limit:
                return tuple(result)

        # M3 carrier hypotheses: derive persistent latent carriers from an existing
        # M2 family's explained M1 transitions.  Carrier identity is based on the
        # recurring context transition, never a raw carrier signature.
        for family in sorted(
            (
                row
                for row in nodes
                if int(row.level) == int(MemoryLevel.M2)
                and int(row.memory_type) == int(MemoryType.FAMILY)
                and row.support_count >= self.min_carrier_family_support
                and self._admissible(row)
            ),
            key=lambda row: row.uid,
        ):
            family_token = stable_u64(family.uid.hi, family.uid.lo, person=b"v8.2-family")
            parent_rows = [
                by_uid[uid]
                for uid in children.get(family.uid, ())
                if uid in by_uid and int(by_uid[uid].level) == int(MemoryLevel.M1)
            ]
            family_support = max(1, sum(max(0, row.support_count) for row in parent_rows))
            for parent in sorted(parent_rows, key=lambda row: row.uid):
                if parent.support_count < self.min_carrier_persistence or len(parent.key_parts) < 4:
                    continue
                carrier_token = stable_u64(
                    int(parent.key_parts[0]),
                    int(parent.key_parts[3]),
                    person=b"v8.2-carrier",
                )
                future_bucket = self._future_bucket(parent.future_option_delta)
                key = (int(family_token), int(carrier_token), int(future_bucket))
                uid = MemoryUid.from_key(MemoryLevel.M3, MemoryType.CARRIER, key)
                persistence = min(1.0, parent.support_count / max(2.0, family_support))
                compression_gain = max(0.0, parent.support_count - 1.0) / max(1.0, parent.support_count)
                utility = max(persistence, compression_gain)
                result.append(
                    FormationCandidate(
                        uid,
                        MemoryLevel.M3,
                        MemoryType.CARRIER,
                        key,
                        (family.uid, parent.uid),
                        int(parent.support_count),
                        utility,
                        compression_gain,
                        0.0,
                        utility,
                        parent.future_option_delta,
                        int(CognitiveState.PROBATION),
                        int(ValidationState.STRUCTURAL),
                        "carrier_candidate",
                        utility,
                    )
                )
                if len(result) >= limit:
                    return tuple(result)

        # M4 concept candidates: explicit CB/ER/TP_prior gate over established roles.
        for role in sorted(
            (
                row
                for row in nodes
                if int(row.level) == int(MemoryLevel.M3)
                and int(row.memory_type) == int(MemoryType.ROLE)
                and row.support_count >= 2
                and self._admissible(row)
            ),
            key=lambda row: row.uid,
        ):
            compression = max(0.0, float(role.support_count - 1))
            explanatory = max(float(role.explanatory_reach), float(role.support_count))
            transfer_prior = max(float(role.transfer_prior), min(1.0, role.game_evidence_count / 2.0))
            if (
                compression <= 0.0
                or explanatory <= self.min_concept_explanatory
                or transfer_prior <= self.min_concept_transfer_prior
            ):
                continue
            role_key = tuple(int(v) for v in role.key_parts[:2])
            key = role_key if role_key else (int(role.uid.lo),)
            uid = MemoryUid.from_key(MemoryLevel.M4, MemoryType.CONCEPT, key)
            result.append(
                FormationCandidate(
                    uid,
                    MemoryLevel.M4,
                    MemoryType.CONCEPT,
                    key,
                    (role.uid,),
                    int(role.support_count),
                    min(1.0, role.significance),
                    min(1.0, compression / max(1.0, role.support_count)),
                    min(1.0, transfer_prior),
                    explanatory,
                    role.future_option_delta,
                    int(CognitiveState.CANDIDATE),
                    int(ValidationState.STRUCTURAL),
                    "concept_candidate",
                    min(1.0, explanatory / 4.0),
                )
            )
            if len(result) >= limit:
                return tuple(result)

        # M5 consequence structures: only established concept candidates/active
        # concepts can seed consequence memory.
        for concept in sorted(
            (
                row
                for row in nodes
                if int(row.level) == int(MemoryLevel.M4)
                and int(row.memory_type) == int(MemoryType.CONCEPT)
                and row.support_count >= 2
                and self._admissible(row)
            ),
            key=lambda row: row.uid,
        ):
            future_bucket = self._future_bucket(concept.future_option_delta)
            family_token = int(concept.key_parts[0]) if concept.key_parts else int(concept.uid.lo)
            consequence_token = stable_u64(
                family_token,
                future_bucket,
                person=b"v8.2-consequence",
            )
            key = (
                int(concept.uid.hi),
                int(concept.uid.lo),
                int(consequence_token),
                int(future_bucket),
            )
            uid = MemoryUid.from_key(MemoryLevel.M5, MemoryType.CONSEQUENCE, key)
            result.append(
                FormationCandidate(
                    uid,
                    MemoryLevel.M5,
                    MemoryType.CONSEQUENCE,
                    key,
                    (concept.uid,),
                    int(concept.support_count),
                    min(1.0, concept.significance),
                    min(1.0, concept.learning_value),
                    min(1.0, concept.transfer_prior),
                    max(1.0, concept.explanatory_reach),
                    concept.future_option_delta,
                    int(CognitiveState.PROBATION),
                    int(ValidationState.STRUCTURAL),
                    "consequence_structure",
                    min(1.0, max(1.0, concept.explanatory_reach) / 4.0),
                )
            )
            if len(result) >= limit:
                return tuple(result)

        # M6 fine outcome variants: consequence-derived and terminal-label free.
        for consequence in sorted(
            (
                row
                for row in nodes
                if int(row.level) == int(MemoryLevel.M5)
                and int(row.memory_type) == int(MemoryType.CONSEQUENCE)
                and row.support_count >= 2
                and len(row.key_parts) >= 4
                and self._admissible(row)
            ),
            key=lambda row: row.uid,
        ):
            future_bucket = int(row_future := int(consequence.key_parts[3]))
            consequence_bucket = int(consequence.key_parts[2]) & 0xFFFF
            context_variant = stable_u64(
                int(consequence.key_parts[0]),
                int(consequence.key_parts[1]),
                person=b"v8.2-outcome-context",
            ) & 0xF
            key = (future_bucket, consequence_bucket, int(context_variant))
            uid = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, key)
            result.append(
                FormationCandidate(
                    uid,
                    MemoryLevel.M6,
                    MemoryType.OUTCOME,
                    key,
                    (consequence.uid,),
                    int(consequence.support_count),
                    min(1.0, consequence.significance),
                    min(1.0, consequence.learning_value),
                    min(1.0, consequence.transfer_prior),
                    max(1.0, consequence.explanatory_reach),
                    consequence.future_option_delta,
                    int(CognitiveState.PROBATION),
                    int(ValidationState.STRUCTURAL),
                    "outcome_equivalence",
                    min(1.0, consequence.support_count / 4.0),
                )
            )
            if len(result) >= limit:
                return tuple(result)

        # M7 strategy candidates: require an already represented M6 outcome and an
        # established M1 action/context contingency.  This is intentionally derived
        # from memory state rather than one raw trajectory event.
        m6_rows = [
            row
            for row in nodes
            if int(row.level) == int(MemoryLevel.M6)
            and int(row.memory_type) == int(MemoryType.OUTCOME)
            and row.support_count >= 2
            and self._admissible(row)
        ]
        for outcome in sorted(m6_rows, key=lambda row: row.uid):
            outcome_future = int(outcome.key_parts[0]) if outcome.key_parts else 0
            for contingency in stable_m1:
                if self._future_bucket(contingency.future_option_delta) != outcome_future:
                    continue
                context_bucket = stable_u64(
                    int(contingency.key_parts[0]), person=b"v8-context"
                )
                key = (
                    int(contingency.key_parts[1]),
                    int(outcome.uid.hi),
                    int(outcome.uid.lo),
                    int(context_bucket),
                )
                uid = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, key)
                support = min(int(contingency.support_count), int(outcome.support_count))
                result.append(
                    FormationCandidate(
                        uid,
                        MemoryLevel.M7,
                        MemoryType.STRATEGY,
                        key,
                        (outcome.uid, contingency.uid),
                        max(1, support),
                        min(1.0, contingency.significance),
                        min(1.0, contingency.learning_value),
                        0.0,
                        1.0,
                        contingency.future_option_delta,
                        int(CognitiveState.PROBATION),
                        int(ValidationState.STRUCTURAL),
                        "strategy_reuse",
                        min(1.0, support / 4.0),
                    )
                )
                if len(result) >= limit:
                    return tuple(result)

        return tuple(result)
