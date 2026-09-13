from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .identity import MemoryUid
from .m0_episode import M0Episode
from .model import MemoryLevel, MemoryType
from .provenance import DerivationProvenance


class GroundedRelation(str, Enum):
    WORLD_TRANSFORMED = "WORLD_TRANSFORMED"
    ACTION_CONDITIONED = "ACTION_CONDITIONED"
    SYMBOL_OCCURRED = "SYMBOL_OCCURRED"
    SYMBOL_REPEATED = "SYMBOL_REPEATED"
    SYMBOL_PRECEDES_SYMBOL = "SYMBOL_PRECEDES_SYMBOL"
    SYMBOL_PRECEDES_ACTION = "SYMBOL_PRECEDES_ACTION"
    SYMBOL_FOLLOWS_ACTION = "SYMBOL_FOLLOWS_ACTION"
    PASSIVE_PRECEDES_SETTLED = "PASSIVE_PRECEDES_SETTLED"


@dataclass(frozen=True, slots=True)
class M1GroundedContingency:
    uid: MemoryUid
    relation: GroundedRelation
    provenance: DerivationProvenance
    environment_instance_id: int
    episode_id: int
    grounded_context_signature: int
    executable_action_token: int | None
    realized_transition_signature: int
    grounded_next_context_signature: int
    support: int = 1

    @classmethod
    def build(cls, relation: GroundedRelation, parents: tuple[M0Episode, ...]) -> "M1GroundedContingency":
        if not parents:
            raise ValueError("grounded contingency requires M0 evidence")
        parent_ids = tuple(row.uid for row in parents)
        first = parents[0].provenance
        context = parents[0].context_signature
        action = parents[0].action_id
        transition = parents[0].outcome_signature if parents[0].outcome_signature is not None else parents[0].payload_digest
        next_context = parents[0].next_context_signature if parents[0].next_context_signature is not None else context
        # Parent connectivity and support are provenance/state, not identity.
        uid = MemoryUid.from_key(
            MemoryLevel.M1,
            MemoryType.GROUNDED_CONTINGENCY,
            (hash_relation(relation), first.environment_instance_id, context, -1 if action is None else action, transition, next_context),
        )
        evidence = parent_ids
        return cls(uid, relation, DerivationProvenance(parent_ids, evidence), first.environment_instance_id, first.episode_id.value, context, action, transition, next_context)


def hash_relation(relation: GroundedRelation) -> int:
    from .identity import stable_u64
    return stable_u64(relation.value, person=b"v9-m1-grounded")
