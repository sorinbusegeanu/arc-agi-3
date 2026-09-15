from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .identity import MemoryUid


class RelationType(str, Enum):
    TEMPORAL = "TEMPORAL"
    CO_OCCURS = "CO_OCCURS"
    PROVENANCE = "PROVENANCE"
    DEPENDS_ON = "DEPENDS_ON"
    ENABLES = "ENABLES"
    BLOCKS = "BLOCKS"
    EXPLAINS = "EXPLAINS"
    SIMILAR_TO = "SIMILAR_TO"
    TRANSFER_CORRESPONDENCE = "TRANSFER_CORRESPONDENCE"
    OUTCOME_EQUIVALENT = "OUTCOME_EQUIVALENT"
    LEADS_TO = "LEADS_TO"
    PREFERENCE = "PREFERENCE"
    OBSERVED_IN = "OBSERVED_IN"
    PRECEDES = "PRECEDES"
    FOLLOWS = "FOLLOWS"
    TEMPORALLY_ALIGNED_WITH = "TEMPORALLY_ALIGNED_WITH"
    STRUCTURALLY_CORRESPONDS_TO = "STRUCTURALLY_CORRESPONDS_TO"
    PARTICIPATES_IN = "PARTICIPATES_IN"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    TRANSFER_VALIDATES = "TRANSFER_VALIDATES"
    GROUNDS = "GROUNDS"
    SUPERSEDES = "SUPERSEDES"\n    SYMBOL_OCCURRENCE = "SYMBOL_OCCURRENCE"\n    SYMBOL_PRECEDES_SYMBOL = "SYMBOL_PRECEDES_SYMBOL"\n    SYMBOL_FOLLOWS_SYMBOL = "SYMBOL_FOLLOWS_SYMBOL"\n    CROSS_MODAL_CORRESPONDENCE = "CROSS_MODAL_CORRESPONDENCE"


class EdgeAuthority(str, Enum):
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class RelationEdge:
    source: MemoryUid
    relation: RelationType
    target: MemoryUid
    evidence_uids: tuple[MemoryUid, ...] = ()
    authority: EdgeAuthority = EdgeAuthority.ACTIVE
    object_version: int = 0

    @property
    def key(self) -> tuple[MemoryUid, str, MemoryUid]:
        return self.source, self.relation.value, self.target

