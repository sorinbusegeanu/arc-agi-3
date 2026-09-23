from __future__ import annotations

from threading import RLock
from typing import Any

from v9.memory.identity import MemoryUid
from v9.memory.model import CanonicalNode, MemoryLevel, MemoryType
from v9.memory.relations import EdgeAuthority, RelationEdge, RelationType
from v9.mutation.proposals import MutationKind, MutationProposal
from v9.mutation.transactions import MutationOutcome, MutationResult, TransactionCoordinator
from v9.mutation.versions import ObjectRef, VersionTable

from .read_view import ReadView


def node_ref(uid: MemoryUid) -> ObjectRef:
    return ObjectRef("node", uid.hi, uid.lo)


def edge_ref(edge: RelationEdge) -> ObjectRef:
    from v9.memory.identity import stable_u64
    return ObjectRef("edge", stable_u64(edge.source.hi, edge.source.lo, edge.relation.value, edge.target.hi, edge.target.lo, person=b"v9-edge"))


class CanonicalGraph:
    SCHEMA_VERSION = 1

    def __init__(self, partition_count: int, *, node_capacity_per_partition: int = 250_000, edge_capacity_per_partition: int = 500_000, applied_proposal_capacity: int = 65_536) -> None:
        self.partition_count = int(partition_count)
        self.node_capacity_per_partition = int(node_capacity_per_partition)
        self.edge_capacity_per_partition = int(edge_capacity_per_partition)
        self.applied_proposal_capacity = int(applied_proposal_capacity)
        if min(self.partition_count, self.node_capacity_per_partition, self.edge_capacity_per_partition, self.applied_proposal_capacity) <= 0:
            raise ValueError("canonical graph capacities must be positive")
        self.generation = 0
        self.nodes: dict[MemoryUid, CanonicalNode] = {}
        self.payloads: dict[MemoryUid, dict[str, Any]] = {}
        self.edges: dict[tuple[MemoryUid, str, MemoryUid], RelationEdge] = {}
        self.versions = VersionTable()
        self.applied_proposals: set[int] = set()
        self._applied_order: list[int] = []
        self._coordinator = TransactionCoordinator(partition_count)
        self._publication_lock = RLock()

    def publish(self, proposal: MutationProposal) -> MutationResult:
        if any(partition >= self.partition_count for partition in proposal.target_partitions):
            return MutationResult(proposal.proposal_uid, MutationOutcome.INVALID, self.generation)
        required_partitions: set[int] = set()
        for write in proposal.writes:
            if write.node is not None:
                required_partitions.add(write.node.uid.shard(self.partition_count))
            elif write.edge is not None:
                required_partitions.add(write.edge.source.shard(self.partition_count))
                required_partitions.add(write.edge.target.shard(self.partition_count))
        if not required_partitions.issubset(proposal.target_partitions):
            return MutationResult(proposal.proposal_uid, MutationOutcome.INVALID, self.generation)
        with self._coordinator.locked(proposal.target_partitions), self._publication_lock:
            if proposal.proposal_uid in self.applied_proposals:
                return MutationResult(proposal.proposal_uid, MutationOutcome.ACCEPTED, self.generation)
            if not proposal.read_set.valid(self.versions):
                return MutationResult(proposal.proposal_uid, MutationOutcome.STALE_READ_SET, self.generation)
            next_nodes = dict(self.nodes)
            next_payloads = {key: dict(value) for key, value in self.payloads.items()}
            next_edges = dict(self.edges)
            refs: list[ObjectRef] = []
            for write in proposal.writes:
                if write.node is not None:
                    ref = node_ref(write.node.uid)
                    current = next_nodes.get(write.node.uid)
                    if current is not None and (
                        current.level != write.node.level
                        or current.memory_type != write.node.memory_type
                        or current.structural_key != write.node.structural_key
                    ):
                        return MutationResult(proposal.proposal_uid, MutationOutcome.INVALID, self.generation)
                    owner_count = sum(uid.shard(self.partition_count) == write.node.uid.shard(self.partition_count) for uid in next_nodes)
                    if write.node.uid not in next_nodes and owner_count >= self.node_capacity_per_partition:
                        return MutationResult(proposal.proposal_uid, MutationOutcome.INVALID, self.generation)
                    next_nodes[write.node.uid] = write.node
                    incoming_payload = dict(write.payload or {})
                    if current is not None and proposal.proposal_class.value == "ADDITIVE":
                        merged_payload = dict(next_payloads.get(write.node.uid, {}))
                        merged_payload.update(incoming_payload)
                        if "parents" in incoming_payload:
                            merged_payload["parents"] = sorted({tuple(map(int, row)) for row in next_payloads.get(write.node.uid, {}).get("parents", [])} | {tuple(map(int, row)) for row in incoming_payload.get("parents", [])})
                        incoming_payload = merged_payload
                    next_payloads[write.node.uid] = incoming_payload
                    refs.append(ref)
                elif write.edge is not None:
                    ref = edge_ref(write.edge)
                    if proposal.mutation_kind is MutationKind.REMOVE_EDGE:
                        next_edges.pop(write.edge.key, None)
                    else:
                        owner = write.edge.source.shard(self.partition_count)
                        owner_count = sum(edge.source.shard(self.partition_count) == owner for edge in next_edges.values())
                        if write.edge.key not in next_edges and owner_count >= self.edge_capacity_per_partition:
                            return MutationResult(proposal.proposal_uid, MutationOutcome.INVALID, self.generation)
                        next_edges[write.edge.key] = write.edge
                    refs.append(ref)
            self.nodes, self.payloads, self.edges = next_nodes, next_payloads, next_edges
            for ref in refs:
                self.versions.bump(ref)
            self.applied_proposals.add(proposal.proposal_uid)
            self._applied_order.append(proposal.proposal_uid)
            if len(self._applied_order) > self.applied_proposal_capacity:
                expired = self._applied_order.pop(0)
                self.applied_proposals.discard(expired)
            self.generation += 1
            return MutationResult(proposal.proposal_uid, MutationOutcome.ACCEPTED, self.generation)

    def read_view(self) -> ReadView:
        with self._publication_lock:
            return ReadView.build(self.generation, self.nodes, self.payloads, self.edges, dict(self.versions._versions))

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION, "partition_count": self.partition_count,
            "node_capacity_per_partition": self.node_capacity_per_partition,
            "edge_capacity_per_partition": self.edge_capacity_per_partition,
            "applied_proposal_capacity": self.applied_proposal_capacity,
            "generation": self.generation,
            "nodes": [{"hi": uid.hi, "lo": uid.lo, "level": int(node.level), "memory_type": int(node.memory_type), "structural_key": list(node.structural_key), "created_watermark": node.created_watermark, "payload": self.payloads.get(uid, {})} for uid, node in sorted(self.nodes.items())],
            "edges": [{"source_hi": row.source.hi, "source_lo": row.source.lo, "relation": row.relation.value, "target_hi": row.target.hi, "target_lo": row.target.lo, "evidence": [[uid.hi, uid.lo] for uid in row.evidence_uids], "authority": row.authority.value, "object_version": row.object_version} for row in (self.edges[key] for key in sorted(self.edges))],
            "versions": self.versions.state_dict(),
            "applied_proposals": list(self._applied_order),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "CanonicalGraph":
        if int(state.get("schema_version", 0)) != cls.SCHEMA_VERSION:
            raise ValueError("unsupported native v9 graph schema")
        result = cls(int(state["partition_count"]), node_capacity_per_partition=int(state.get("node_capacity_per_partition", 250_000)), edge_capacity_per_partition=int(state.get("edge_capacity_per_partition", 500_000)), applied_proposal_capacity=int(state.get("applied_proposal_capacity", 65_536)))
        result.generation = int(state.get("generation", 0))
        for raw in state.get("nodes", []):
            uid = MemoryUid(int(raw["hi"]), int(raw["lo"]))
            result.nodes[uid] = CanonicalNode(uid, MemoryLevel(int(raw["level"])), MemoryType(int(raw["memory_type"])), tuple(int(v) for v in raw["structural_key"]), int(raw["created_watermark"]))
            result.payloads[uid] = dict(raw.get("payload", {}))
        for raw in state.get("edges", []):
            edge = RelationEdge(MemoryUid(int(raw["source_hi"]), int(raw["source_lo"])), RelationType(str(raw["relation"])), MemoryUid(int(raw["target_hi"]), int(raw["target_lo"])), tuple(MemoryUid(int(v[0]), int(v[1])) for v in raw.get("evidence", [])), EdgeAuthority(str(raw["authority"])), int(raw.get("object_version", 0)))
            result.edges[edge.key] = edge
        result.versions = VersionTable.from_state_dict(list(state.get("versions", [])))
        result._applied_order = [int(value) for value in state.get("applied_proposals", [])][-result.applied_proposal_capacity :]
        result.applied_proposals = set(result._applied_order)
        return result
