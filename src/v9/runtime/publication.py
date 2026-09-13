from __future__ import annotations

from collections import deque
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
        self._applied_order: deque[int] = deque()
        self._node_counts_by_partition = [0] * self.partition_count
        self._edge_counts_by_partition = [0] * self.partition_count
        self._uids_by_level = {level: set() for level in MemoryLevel}
        self._cached_read_view: ReadView | None = None
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

            # Validate and stage only the objects touched by this proposal.  The
            # publication lock keeps readers from observing the later in-place
            # commit, so cloning the complete graph for every write is neither
            # required for atomicity nor affordable as the graph grows.
            node_updates: dict[MemoryUid, tuple[CanonicalNode, dict[str, Any]]] = {}
            edge_updates: dict[tuple[MemoryUid, str, MemoryUid], RelationEdge | None] = {}
            node_count_deltas = [0] * self.partition_count
            edge_count_deltas = [0] * self.partition_count
            refs: list[ObjectRef] = []
            for write in proposal.writes:
                if write.node is not None:
                    ref = node_ref(write.node.uid)
                    staged = node_updates.get(write.node.uid)
                    current = staged[0] if staged is not None else self.nodes.get(write.node.uid)
                    if current is not None and (
                        current.level != write.node.level
                        or current.memory_type != write.node.memory_type
                        or current.structural_key != write.node.structural_key
                    ):
                        return MutationResult(proposal.proposal_uid, MutationOutcome.INVALID, self.generation)
                    owner = write.node.uid.shard(self.partition_count)
                    if current is None:
                        if self._node_counts_by_partition[owner] + node_count_deltas[owner] >= self.node_capacity_per_partition:
                            return MutationResult(proposal.proposal_uid, MutationOutcome.INVALID, self.generation)
                        node_count_deltas[owner] += 1
                    incoming_payload = dict(write.payload or {})
                    if current is not None and proposal.proposal_class.value == "ADDITIVE":
                        current_payload = staged[1] if staged is not None else self.payloads.get(write.node.uid, {})
                        merged_payload = dict(current_payload)
                        merged_payload.update(incoming_payload)
                        if "parents" in incoming_payload:
                            merged_payload["parents"] = sorted({tuple(map(int, row)) for row in current_payload.get("parents", [])} | {tuple(map(int, row)) for row in incoming_payload.get("parents", [])})
                        incoming_payload = merged_payload
                    node_updates[write.node.uid] = (write.node, incoming_payload)
                    refs.append(ref)
                elif write.edge is not None:
                    ref = edge_ref(write.edge)
                    current_edge = edge_updates.get(write.edge.key, self.edges.get(write.edge.key))
                    owner = write.edge.source.shard(self.partition_count)
                    if proposal.mutation_kind is MutationKind.REMOVE_EDGE:
                        if current_edge is not None:
                            edge_count_deltas[owner] -= 1
                        edge_updates[write.edge.key] = None
                    else:
                        if current_edge is None:
                            if self._edge_counts_by_partition[owner] + edge_count_deltas[owner] >= self.edge_capacity_per_partition:
                                return MutationResult(proposal.proposal_uid, MutationOutcome.INVALID, self.generation)
                            edge_count_deltas[owner] += 1
                        edge_updates[write.edge.key] = write.edge
                    refs.append(ref)

            for uid, (node, payload) in node_updates.items():
                if uid not in self.nodes:
                    self._uids_by_level[node.level].add(uid)
                self.nodes[uid] = node
                self.payloads[uid] = payload
            for key, edge in edge_updates.items():
                if edge is None:
                    self.edges.pop(key, None)
                else:
                    self.edges[key] = edge
            for partition, delta in enumerate(node_count_deltas):
                self._node_counts_by_partition[partition] += delta
            for partition, delta in enumerate(edge_count_deltas):
                self._edge_counts_by_partition[partition] += delta
            for ref in refs:
                self.versions.bump(ref)
            if len(self._applied_order) >= self.applied_proposal_capacity:
                expired = self._applied_order.popleft()
                self.applied_proposals.discard(expired)
            self.applied_proposals.add(proposal.proposal_uid)
            self._applied_order.append(proposal.proposal_uid)
            self.generation += 1
            self._cached_read_view = None
            return MutationResult(proposal.proposal_uid, MutationOutcome.ACCEPTED, self.generation)

    def read_view(self) -> ReadView:
        with self._publication_lock:
            if self._cached_read_view is None:
                self._cached_read_view = ReadView.build(self.generation, self.nodes, self.payloads, self.edges, dict(self.versions._versions))
            return self._cached_read_view

    def memory_count(self, level: MemoryLevel | None = None) -> int:
        with self._publication_lock:
            return len(self.nodes) if level is None else len(self._uids_by_level[level])

    def uids_at_level(self, level: MemoryLevel) -> tuple[MemoryUid, ...]:
        with self._publication_lock:
            return tuple(self._uids_by_level[level])

    def state_dict(self) -> dict[str, object]:
        with self._publication_lock:
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
            node = CanonicalNode(uid, MemoryLevel(int(raw["level"])), MemoryType(int(raw["memory_type"])), tuple(int(v) for v in raw["structural_key"]), int(raw["created_watermark"]))
            result.nodes[uid] = node
            result.payloads[uid] = dict(raw.get("payload", {}))
            result._node_counts_by_partition[uid.shard(result.partition_count)] += 1
            result._uids_by_level[node.level].add(uid)
        for raw in state.get("edges", []):
            edge = RelationEdge(MemoryUid(int(raw["source_hi"]), int(raw["source_lo"])), RelationType(str(raw["relation"])), MemoryUid(int(raw["target_hi"]), int(raw["target_lo"])), tuple(MemoryUid(int(v[0]), int(v[1])) for v in raw.get("evidence", [])), EdgeAuthority(str(raw["authority"])), int(raw.get("object_version", 0)))
            result.edges[edge.key] = edge
            result._edge_counts_by_partition[edge.source.shard(result.partition_count)] += 1
        result.versions = VersionTable.from_state_dict(list(state.get("versions", [])))
        applied_order = [int(value) for value in state.get("applied_proposals", [])]
        result._applied_order = deque(applied_order[-result.applied_proposal_capacity :])
        result.applied_proposals = set(result._applied_order)
        return result
