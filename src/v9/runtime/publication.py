from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class RetiredTombstone:
    uid: MemoryUid
    level: MemoryLevel
    memory_type: MemoryType
    structural_key: tuple[int, ...]
    retired_generation: int
    replacement_uid: MemoryUid
    reason: str


class CanonicalGraph:
    SCHEMA_VERSION = 1

    def __init__(self, partition_count: int, *, node_capacity_per_partition: int | None = 250_000, edge_capacity_per_partition: int | None = 500_000, applied_proposal_capacity: int = 65_536) -> None:
        self.partition_count = int(partition_count)
        self.node_capacity_per_partition = None if node_capacity_per_partition is None else int(node_capacity_per_partition)
        self.edge_capacity_per_partition = None if edge_capacity_per_partition is None else int(edge_capacity_per_partition)
        self.applied_proposal_capacity = int(applied_proposal_capacity)
        if self.partition_count <= 0 or self.applied_proposal_capacity <= 0:
            raise ValueError("canonical graph counts must be positive")
        if self.node_capacity_per_partition is not None and self.node_capacity_per_partition <= 0:
            raise ValueError("canonical graph node capacity must be positive or None")
        if self.edge_capacity_per_partition is not None and self.edge_capacity_per_partition <= 0:
            raise ValueError("canonical graph edge capacity must be positive or None")
        self.generation = 0
        self.nodes: dict[MemoryUid, CanonicalNode] = {}
        self.payloads: dict[MemoryUid, dict[str, Any]] = {}
        self.edges: dict[tuple[MemoryUid, str, MemoryUid], RelationEdge] = {}
        self.retired_tombstones: dict[MemoryUid, RetiredTombstone] = {}
        self.versions = VersionTable()
        self.applied_proposals: set[int] = set()
        self._applied_order: deque[int] = deque()
        self._node_counts_by_partition = [0] * self.partition_count
        self._edge_counts_by_partition = [0] * self.partition_count
        self._node_uids_by_partition = [set() for _ in range(self.partition_count)]
        self._edge_keys_by_partition = [set() for _ in range(self.partition_count)]
        self._uids_by_level = {level: set() for level in MemoryLevel}
        self._cached_read_view: ReadView | None = None
        self._coordinator = TransactionCoordinator(partition_count)
        self._publication_lock = RLock()

    def _required_partitions(self, proposal: MutationProposal) -> set[int] | None:
        if any(partition >= self.partition_count for partition in proposal.target_partitions):
            return None
        required: set[int] = set()
        for write in proposal.writes:
            if write.node is not None:
                required.add(write.node.uid.shard(self.partition_count))
            elif write.edge is not None:
                required.add(write.edge.source.shard(self.partition_count))
                required.add(write.edge.target.shard(self.partition_count))
        if not required.issubset(proposal.target_partitions):
            return None
        return required

    def _publish_locked(self, proposal: MutationProposal) -> MutationResult:
        if proposal.proposal_uid in self.applied_proposals:
            return MutationResult(proposal.proposal_uid, MutationOutcome.ACCEPTED, self.generation)
        if not proposal.read_set.valid(self.versions):
            return MutationResult(proposal.proposal_uid, MutationOutcome.STALE_READ_SET, self.generation)
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
                if current is not None and (current.level != write.node.level or current.memory_type != write.node.memory_type or current.structural_key != write.node.structural_key):
                    return MutationResult(proposal.proposal_uid, MutationOutcome.INVALID, self.generation)
                owner = write.node.uid.shard(self.partition_count)
                if current is None:
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
                        edge_count_deltas[owner] += 1
                    edge_updates[write.edge.key] = write.edge
                refs.append(ref)
        for uid, (node, payload) in node_updates.items():
            if uid not in self.nodes:
                owner = uid.shard(self.partition_count)
                self._uids_by_level[node.level].add(uid)
                self._node_uids_by_partition[owner].add(uid)
                self.retired_tombstones.pop(uid, None)
            self.nodes[uid] = node
            self.payloads[uid] = payload
        for key, edge in edge_updates.items():
            owner = key[0].shard(self.partition_count)
            if edge is None:
                self.edges.pop(key, None)
                self._edge_keys_by_partition[owner].discard(key)
            else:
                self.edges[key] = edge
                self._edge_keys_by_partition[owner].add(key)
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

    def publish(self, proposal: MutationProposal) -> MutationResult:
        required = self._required_partitions(proposal)
        if required is None:
            return MutationResult(proposal.proposal_uid, MutationOutcome.INVALID, self.generation)
        with self._coordinator.locked(proposal.target_partitions), self._publication_lock:
            return self._publish_locked(proposal)

    def publish_batch(self, proposals: tuple[MutationProposal, ...]) -> tuple[MutationResult, ...]:
        if not proposals:
            return ()
        valid: list[bool] = []
        partitions: set[int] = set()
        for proposal in proposals:
            required = self._required_partitions(proposal)
            ok = required is not None
            valid.append(ok)
            if ok:
                partitions.update(proposal.target_partitions)
        results: list[MutationResult] = []
        with self._coordinator.locked(tuple(sorted(partitions))), self._publication_lock:
            for proposal, ok in zip(proposals, valid):
                if not ok:
                    results.append(MutationResult(proposal.proposal_uid, MutationOutcome.INVALID, self.generation))
                else:
                    results.append(self._publish_locked(proposal))
        return tuple(results)

    def pressure_ratio(self) -> float:
        with self._publication_lock:
            ratios: list[float] = []
            if self.node_capacity_per_partition is not None:
                ratios.extend(count / self.node_capacity_per_partition for count in self._node_counts_by_partition)
            if self.edge_capacity_per_partition is not None:
                ratios.extend(count / self.edge_capacity_per_partition for count in self._edge_counts_by_partition)
            return max(ratios, default=0.0)

    def provenance_replacements(self, *, maximum_level: MemoryLevel = MemoryLevel.M1) -> dict[MemoryUid, MemoryUid]:
        with self._publication_lock:
            replacements: dict[MemoryUid, MemoryUid] = {}
            for edge in self.edges.values():
                if edge.relation is not RelationType.PROVENANCE:
                    continue
                source = self.nodes.get(edge.source)
                target = self.nodes.get(edge.target)
                if source is None or target is None or target.level > maximum_level or source.level <= target.level:
                    continue
                current_uid = replacements.get(target.uid)
                current = self.nodes.get(current_uid) if current_uid is not None else None
                if current is None or source.level > current.level or (source.level == current.level and source.uid < current.uid):
                    replacements[target.uid] = source.uid
            return replacements

    def retire_nodes_batch(self, plans: tuple[tuple[MemoryUid, MemoryUid, str], ...]) -> tuple[MemoryUid, ...]:
        if not plans:
            return ()
        all_partitions = tuple(range(self.partition_count))
        with self._coordinator.locked(all_partitions), self._publication_lock:
            accepted: dict[MemoryUid, tuple[CanonicalNode, MemoryUid, str]] = {}
            for uid, replacement_uid, reason in plans:
                node = self.nodes.get(uid)
                replacement = self.nodes.get(replacement_uid)
                if node is None or replacement is None:
                    continue
                if node.level > MemoryLevel.M1 or replacement.level <= node.level:
                    continue
                provenance_key = (replacement_uid, RelationType.PROVENANCE.value, uid)
                if provenance_key not in self.edges:
                    continue
                accepted[uid] = (node, replacement_uid, str(reason))
            if not accepted:
                return ()
            retiring = set(accepted)
            incident = [edge for edge in self.edges.values() if edge.source in retiring or edge.target in retiring]
            for edge in incident:
                key = edge.key
                owner = edge.source.shard(self.partition_count)
                self.edges.pop(key, None)
                self._edge_keys_by_partition[owner].discard(key)
                self._edge_counts_by_partition[owner] -= 1
                self.versions.bump(edge_ref(edge))
            retired_generation = self.generation + 1
            for uid, (node, replacement_uid, reason) in accepted.items():
                owner = uid.shard(self.partition_count)
                self.nodes.pop(uid, None)
                self.payloads.pop(uid, None)
                self._node_uids_by_partition[owner].discard(uid)
                self._uids_by_level[node.level].discard(uid)
                self._node_counts_by_partition[owner] -= 1
                self.versions.bump(node_ref(uid))
                self.retired_tombstones[uid] = RetiredTombstone(uid, node.level, node.memory_type, node.structural_key, retired_generation, replacement_uid, reason)
            self.generation = retired_generation
            self._cached_read_view = None
            return tuple(sorted(accepted))

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

    def _partition_state(self, partition: int):
        nodes = [{"hi": uid.hi, "lo": uid.lo, "level": int(self.nodes[uid].level), "memory_type": int(self.nodes[uid].memory_type), "structural_key": list(self.nodes[uid].structural_key), "created_watermark": self.nodes[uid].created_watermark, "payload": self.payloads.get(uid, {})} for uid in sorted(self._node_uids_by_partition[partition])]
        edges = [{"source_hi": self.edges[key].source.hi, "source_lo": self.edges[key].source.lo, "relation": self.edges[key].relation.value, "target_hi": self.edges[key].target.hi, "target_lo": self.edges[key].target.lo, "evidence": [[uid.hi, uid.lo] for uid in self.edges[key].evidence_uids], "authority": self.edges[key].authority.value, "object_version": self.edges[key].object_version} for key in sorted(self._edge_keys_by_partition[partition]) if key in self.edges]
        return nodes, edges

    def state_dict(self) -> dict[str, object]:
        with self._publication_lock:
            workers = max(1, min(self.partition_count, 8))
            if workers == 1:
                parts = [self._partition_state(0)]
            else:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v9-snapshot") as pool:
                    parts = list(pool.map(self._partition_state, range(self.partition_count)))
            nodes = [row for part, _ in parts for row in part]
            edges = [row for _, part in parts for row in part]
            tombstones = [{"hi": uid.hi, "lo": uid.lo, "level": int(row.level), "memory_type": int(row.memory_type), "structural_key": list(row.structural_key), "retired_generation": row.retired_generation, "replacement": [row.replacement_uid.hi, row.replacement_uid.lo], "reason": row.reason} for uid, row in sorted(self.retired_tombstones.items())]
            return {"schema_version": self.SCHEMA_VERSION, "partition_count": self.partition_count, "node_capacity_per_partition": self.node_capacity_per_partition, "edge_capacity_per_partition": self.edge_capacity_per_partition, "applied_proposal_capacity": self.applied_proposal_capacity, "generation": self.generation, "nodes": nodes, "edges": edges, "retired_tombstones": tombstones, "versions": self.versions.state_dict(), "applied_proposals": list(self._applied_order)}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "CanonicalGraph":
        if int(state.get("schema_version", 0)) != cls.SCHEMA_VERSION:
            raise ValueError("unsupported native v9 graph schema")
        raw_node_capacity = state.get("node_capacity_per_partition", 250_000)
        raw_edge_capacity = state.get("edge_capacity_per_partition", 500_000)
        result = cls(
            int(state["partition_count"]),
            node_capacity_per_partition=None if raw_node_capacity is None else int(raw_node_capacity),
            edge_capacity_per_partition=None if raw_edge_capacity is None else int(raw_edge_capacity),
            applied_proposal_capacity=int(state.get("applied_proposal_capacity", 65_536)),
        )
        result.generation = int(state.get("generation", 0))
        for raw in state.get("nodes", []):
            uid = MemoryUid(int(raw["hi"]), int(raw["lo"]))
            node = CanonicalNode(uid, MemoryLevel(int(raw["level"])), MemoryType(int(raw["memory_type"])), tuple(int(v) for v in raw["structural_key"]), int(raw["created_watermark"]))
            result.nodes[uid] = node
            result.payloads[uid] = dict(raw.get("payload", {}))
            owner = uid.shard(result.partition_count)
            result._node_counts_by_partition[owner] += 1
            result._node_uids_by_partition[owner].add(uid)
            result._uids_by_level[node.level].add(uid)
        for raw in state.get("edges", []):
            edge = RelationEdge(MemoryUid(int(raw["source_hi"]), int(raw["source_lo"])), RelationType(str(raw["relation"])), MemoryUid(int(raw["target_hi"]), int(raw["target_lo"])), tuple(MemoryUid(int(v[0]), int(v[1])) for v in raw.get("evidence", [])), EdgeAuthority(str(raw["authority"])), int(raw.get("object_version", 0)))
            result.edges[edge.key] = edge
            owner = edge.source.shard(result.partition_count)
            result._edge_counts_by_partition[owner] += 1
            result._edge_keys_by_partition[owner].add(edge.key)
        for raw in state.get("retired_tombstones", []):
            uid = MemoryUid(int(raw["hi"]), int(raw["lo"]))
            replacement = raw.get("replacement", [0, 0])
            result.retired_tombstones[uid] = RetiredTombstone(uid, MemoryLevel(int(raw["level"])), MemoryType(int(raw["memory_type"])), tuple(int(v) for v in raw.get("structural_key", [])), int(raw.get("retired_generation", result.generation)), MemoryUid(int(replacement[0]), int(replacement[1])), str(raw.get("reason", "retention_compaction")))
        result.versions = VersionTable.from_state_dict(list(state.get("versions", [])))
        applied_order = [int(value) for value in state.get("applied_proposals", [])]
        result._applied_order = deque(applied_order[-result.applied_proposal_capacity :])
        result.applied_proposals = set(result._applied_order)
        return result
