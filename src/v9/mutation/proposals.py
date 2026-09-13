from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from v9.memory.identity import MemoryUid, stable_u64
from v9.memory.model import CanonicalNode
from v9.memory.relations import RelationEdge

from .read_sets import ReadSet


class MutationKind(str, Enum):
    UPSERT_NODE = "UPSERT_NODE"
    UPSERT_EDGE = "UPSERT_EDGE"
    REMOVE_EDGE = "REMOVE_EDGE"
    RETIRE_NODE = "RETIRE_NODE"
    UPDATE_LIFECYCLE = "UPDATE_LIFECYCLE"
    UPDATE_OVERLAY = "UPDATE_OVERLAY"
    UPDATE_DEPENDENCY = "UPDATE_DEPENDENCY"
    UPDATE_VALIDATION = "UPDATE_VALIDATION"
    RESOLVE_EQUIVALENCE = "RESOLVE_EQUIVALENCE"
    PUBLISH_ESTIMATOR = "PUBLISH_ESTIMATOR"


class ProposalClass(str, Enum):
    ADDITIVE = "ADDITIVE"
    STATEFUL = "STATEFUL"


@dataclass(frozen=True, slots=True)
class MutationWrite:
    node: CanonicalNode | None = None
    edge: RelationEdge | None = None
    payload: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if (self.node is None) == (self.edge is None):
            raise ValueError("mutation write must contain exactly one node or edge")


@dataclass(frozen=True, slots=True)
class MutationProposal:
    proposal_uid: int
    mutation_kind: MutationKind
    target_partitions: tuple[int, ...]
    read_set: ReadSet
    evidence_refs: tuple[MemoryUid, ...]
    causal_watermark: int
    writes: tuple[MutationWrite, ...]
    proposal_class: ProposalClass = ProposalClass.ADDITIVE
    source_peer_stable_id: int = 0
    priority_class: int = 0

    @property
    def ordering_key(self) -> tuple[int, int, int, int]:
        return int(self.causal_watermark), int(self.priority_class), int(self.source_peer_stable_id), int(self.proposal_uid)

    @classmethod
    def build(cls, mutation_kind: MutationKind, *, target_partitions: tuple[int, ...], read_set: ReadSet, evidence_refs: tuple[MemoryUid, ...], causal_watermark: int, writes: tuple[MutationWrite, ...], proposal_class: ProposalClass | None = None, source_peer_stable_id: int = 0, priority_class: int = 0) -> "MutationProposal":
        partitions = tuple(sorted(set(int(value) for value in target_partitions)))
        if not partitions or any(value < 0 for value in partitions):
            raise ValueError("proposal partitions must be non-negative")
        if not writes:
            raise ValueError("proposal requires writes")
        category = proposal_class or (ProposalClass.ADDITIVE if mutation_kind in {MutationKind.UPSERT_NODE, MutationKind.UPSERT_EDGE} else ProposalClass.STATEFUL)
        if category is ProposalClass.STATEFUL and not read_set.dependencies:
            raise ValueError("stateful mutations require an authoritative read set")
        write_keys = tuple(
            f"n:{write.node.uid.hi}:{write.node.uid.lo}:{write.node.structural_key}" if write.node is not None
            else f"e:{write.edge.source.hi}:{write.edge.source.lo}:{write.edge.relation.value}:{write.edge.target.hi}:{write.edge.target.lo}"
            for write in writes
        )
        uid = stable_u64(
            mutation_kind.value, category.value, int(causal_watermark), int(source_peer_stable_id), int(priority_class), *partitions,
            *(f"{row.ref.kind}:{row.ref.uid_hi}:{row.ref.uid_lo}:{row.version}" for row in read_set.dependencies),
            *(f"{ref.hi}:{ref.lo}" for ref in evidence_refs),
            *write_keys,
            person=b"v9-mutation",
        )
        return cls(uid, mutation_kind, partitions, read_set, tuple(sorted(set(evidence_refs))), int(causal_watermark), writes, category, int(source_peer_stable_id), int(priority_class))
