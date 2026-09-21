from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import heapq
from threading import RLock
from types import SimpleNamespace
from typing import Any, Callable

from v9.memory.identity import MemoryUid, stable_u64
from v9.memory.model import CanonicalNode, MemoryLevel, MemoryType
from v9.memory.relations import EdgeAuthority, RelationEdge, RelationType
from v9.mutation.proposals import MutationKind, MutationProposal
from v9.mutation.transactions import MutationOutcome, MutationResult, TransactionCoordinator
from v9.mutation.versions import ObjectRef, VersionTable

from .read_view import ReadView, _cognitively_visible


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

    def __init__(
        self,
        partition_count: int,
        *,
        node_capacity_per_partition: int | None = 250_000,
        edge_capacity_per_partition: int | None = 500_000,
        applied_proposal_capacity: int = 65_536,
    ) -> None:
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
        self._training_m0_reservoir: deque[MemoryUid] = deque(maxlen=65_536)
        self._outgoing_edge_keys: dict[MemoryUid, set[tuple[MemoryUid, str, MemoryUid]]] = {}
        self._incoming_edge_keys: dict[MemoryUid, set[tuple[MemoryUid, str, MemoryUid]]] = {}
        self._provenance_sources_by_target: dict[MemoryUid, set[MemoryUid]] = {}
        self._provenance_targets_by_source: dict[MemoryUid, set[MemoryUid]] = {}
        self._cached_read_view: ReadView | None = None
        self._coordinator = TransactionCoordinator(partition_count)
        self._publication_lock = RLock()
        self._durable_commit: Callable[..., None] | None = None

    def configure_durable_commit(self, callback: Callable[..., None] | None) -> None:
        """Set the explicit WAL/immutable-store owner for accepted publications.

        The callback runs after all proposal validation and staging, but before
        any live graph dictionary is changed.  Raising from it therefore leaves
        the actor-visible graph at its previous complete generation.
        """
        with self._publication_lock:
            self._durable_commit = callback

    def _index_edge(self, edge: RelationEdge) -> None:
        self._outgoing_edge_keys.setdefault(edge.source, set()).add(edge.key)
        self._incoming_edge_keys.setdefault(edge.target, set()).add(edge.key)
        if edge.relation is RelationType.PROVENANCE:
            self._provenance_sources_by_target.setdefault(edge.target, set()).add(edge.source)
            self._provenance_targets_by_source.setdefault(edge.source, set()).add(edge.target)

    def _deindex_edge(self, edge: RelationEdge) -> None:
        outgoing = self._outgoing_edge_keys.get(edge.source)
        if outgoing is not None:
            outgoing.discard(edge.key)
            if not outgoing:
                self._outgoing_edge_keys.pop(edge.source, None)
        incoming = self._incoming_edge_keys.get(edge.target)
        if incoming is not None:
            incoming.discard(edge.key)
            if not incoming:
                self._incoming_edge_keys.pop(edge.target, None)
        if edge.relation is RelationType.PROVENANCE:
            sources = self._provenance_sources_by_target.get(edge.target)
            if sources is not None:
                sources.discard(edge.source)
                if not sources:
                    self._provenance_sources_by_target.pop(edge.target, None)
            targets = self._provenance_targets_by_source.get(edge.source)
            if targets is not None:
                targets.discard(edge.target)
                if not targets:
                    self._provenance_targets_by_source.pop(edge.source, None)

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
                if current is not None and (
                    current.level != write.node.level
                    or current.memory_type != write.node.memory_type
                    or current.structural_key != write.node.structural_key
                ):
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
                        merged_payload["parents"] = sorted(
                            {tuple(map(int, row)) for row in current_payload.get("parents", [])}
                            | {tuple(map(int, row)) for row in incoming_payload.get("parents", [])}
                        )
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
        if self._durable_commit is not None:
            self._durable_commit(
                proposal=proposal,
                node_updates=node_updates,
                node_deletes={},
                edge_updates=edge_updates,
                next_generation=self.generation + 1,
            )
        for uid, (node, payload) in node_updates.items():
            if uid not in self.nodes:
                owner = uid.shard(self.partition_count)
                self._uids_by_level[node.level].add(uid)
                self._node_uids_by_partition[owner].add(uid)
                if node.level is MemoryLevel.M0 and payload.get("action_id") is not None:
                    self._training_m0_reservoir.append(uid)
                self.retired_tombstones.pop(uid, None)
            self.nodes[uid] = node
            self.payloads[uid] = payload
        for key, edge in edge_updates.items():
            owner = key[0].shard(self.partition_count)
            previous = self.edges.get(key)
            if edge is None:
                if previous is not None:
                    self._deindex_edge(previous)
                self.edges.pop(key, None)
                self._edge_keys_by_partition[owner].discard(key)
            else:
                if previous is None:
                    self._index_edge(edge)
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

    def training_m0_reservoir_snapshot(self) -> tuple[MemoryUid, ...]:
        """Return a stable training-reservoir cut for lock-free consumers."""
        with self._publication_lock:
            return tuple(self._training_m0_reservoir)

    def pressure_ratio(self) -> float:
        with self._publication_lock:
            ratios: list[float] = []
            if self.node_capacity_per_partition is not None:
                ratios.extend(count / self.node_capacity_per_partition for count in self._node_counts_by_partition)
            if self.edge_capacity_per_partition is not None:
                ratios.extend(count / self.edge_capacity_per_partition for count in self._edge_counts_by_partition)
            return max(ratios, default=0.0)

    def provenance_replacement(self, uid: MemoryUid, *, maximum_level: MemoryLevel = MemoryLevel.M1) -> MemoryUid | None:
        with self._publication_lock:
            target = self.nodes.get(uid)
            if target is None or target.level > maximum_level:
                return None
            best: CanonicalNode | None = None
            for source_uid in self._provenance_sources_by_target.get(uid, ()):
                source = self.nodes.get(source_uid)
                if source is None or source.level <= target.level:
                    continue
                if best is None or source.level > best.level or (source.level == best.level and source.uid < best.uid):
                    best = source
            return None if best is None else best.uid

    def provenance_replacements(self, *, maximum_level: MemoryLevel = MemoryLevel.M1) -> dict[MemoryUid, MemoryUid]:
        with self._publication_lock:
            replacements: dict[MemoryUid, MemoryUid] = {}
            for target_uid, source_uids in self._provenance_sources_by_target.items():
                target = self.nodes.get(target_uid)
                if target is None or target.level > maximum_level:
                    continue
                best: CanonicalNode | None = None
                for source_uid in source_uids:
                    source = self.nodes.get(source_uid)
                    if source is None or source.level <= target.level:
                        continue
                    if best is None or source.level > best.level or (source.level == best.level and source.uid < best.uid):
                        best = source
                if best is not None:
                    replacements[target_uid] = best.uid
            return replacements

    def replacement_target_uids(self, replacement_uid: MemoryUid) -> tuple[MemoryUid, ...]:
        with self._publication_lock:
            return tuple(self._provenance_targets_by_source.get(replacement_uid, ()))

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
            incident_keys: set[tuple[MemoryUid, str, MemoryUid]] = set()
            for uid in retiring:
                incident_keys.update(self._outgoing_edge_keys.get(uid, ()))
                incident_keys.update(self._incoming_edge_keys.get(uid, ()))
            if self._durable_commit is not None:
                self._durable_commit(
                    proposal=SimpleNamespace(
                        proposal_uid=stable_u64(
                            self.generation,
                            *(uid.hex() for uid in sorted(retiring)),
                            person=b"v9-retire-wal",
                        )
                    ),
                    node_updates={},
                    node_deletes={uid: accepted[uid][0] for uid in retiring},
                    edge_updates={key: None for key in incident_keys},
                    next_generation=self.generation + 1,
                )
            for key in tuple(incident_keys):
                edge = self.edges.get(key)
                if edge is None:
                    continue
                owner = edge.source.shard(self.partition_count)
                self._deindex_edge(edge)
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
                self._outgoing_edge_keys.pop(uid, None)
                self._incoming_edge_keys.pop(uid, None)
                self.versions.bump(node_ref(uid))
                self.retired_tombstones[uid] = RetiredTombstone(
                    uid,
                    node.level,
                    node.memory_type,
                    node.structural_key,
                    retired_generation,
                    replacement_uid,
                    reason,
                )
            self.generation = retired_generation
            self._cached_read_view = None
            return tuple(sorted(accepted))

    def read_view(self) -> ReadView:
        with self._publication_lock:
            if self._cached_read_view is None:
                self._cached_read_view = ReadView.build(
                    self.generation,
                    self.nodes,
                    self.payloads,
                    self.edges,
                    dict(self.versions._versions),
                )
            return self._cached_read_view

    def training_view(self, *, max_nodes: int = 800, max_edges: int = 4000) -> ReadView:
        """Build a bounded HGT cut without materializing/freezing the full canonical graph."""
        maximum_nodes = max(8, int(max_nodes))
        maximum_edges = max(1, int(max_edges))
        with self._publication_lock:
            behavior_candidates = []
            m0_training_uids = tuple(self._training_m0_reservoir)
            if not m0_training_uids:
                m0_training_uids = tuple(self._uids_by_level[MemoryLevel.M0])
            for uid in m0_training_uids:
                node = self.nodes.get(uid)
                if node is None:
                    continue
                payload = self.payloads.get(uid, {})
                if payload.get("action_id") is None or payload.get("environment_instance_id") is None or payload.get("episode_id") is None:
                    continue
                behavior_candidates.append((uid, node, abs(int(payload.get("primary_valence", 0)))))
            seeds = heapq.nlargest(
                max(1, maximum_nodes // 2),
                behavior_candidates,
                key=lambda row: (row[2], int(row[1].created_watermark), row[0]),
            )
            selected: dict[MemoryUid, CanonicalNode] = {uid: node for uid, node, _ in seeds}
            # Reserve representation capacity for every available Hydra level.
            # HGT is heterogeneous; a behavior-heavy M0 reservoir must not crowd
            # M2-M7 abstractions out of the bounded training cut.
            remaining_budget = max(0, maximum_nodes - len(selected))
            nonempty_levels = [level for level in MemoryLevel if self._uids_by_level[level]]
            per_level = max(1, remaining_budget // max(1, len(nonempty_levels)))
            for level in nonempty_levels:
                level_rows = (
                    (uid, self.nodes[uid])
                    for uid in self._uids_by_level[level]
                    if uid not in selected
                    and uid in self.nodes
                    and _cognitively_visible(self.payloads.get(uid, {}))
                )
                for uid, node in heapq.nlargest(
                    per_level,
                    level_rows,
                    key=lambda row: (int(row[1].created_watermark), row[0]),
                ):
                    if len(selected) >= maximum_nodes:
                        break
                    selected[uid] = node
            historical_uids = {
                uid
                for uid, _, _ in seeds
                if not _cognitively_visible(self.payloads.get(uid, {}))
            }
            frontier = deque(uid for uid, _, _ in seeds)
            visited: set[MemoryUid] = set()
            while frontier and len(selected) < maximum_nodes:
                uid = frontier.popleft()
                if uid in visited:
                    continue
                visited.add(uid)
                keys = set(self._outgoing_edge_keys.get(uid, ())) | set(self._incoming_edge_keys.get(uid, ()))
                for key in sorted(keys):
                    edge = self.edges.get(key)
                    if edge is None:
                        continue
                    other = edge.target if edge.source == uid else edge.source
                    if other in selected:
                        continue
                    node = self.nodes.get(other)
                    if node is None or not _cognitively_visible(self.payloads.get(other, {})):
                        continue
                    selected[other] = node
                    frontier.append(other)
                    if len(selected) >= maximum_nodes:
                        break
            if len(selected) < maximum_nodes:
                remaining = (
                    (uid, node)
                    for uid, node in self.nodes.items()
                    if uid not in selected and _cognitively_visible(self.payloads.get(uid, {}))
                )
                for uid, node in heapq.nlargest(
                    maximum_nodes - len(selected),
                    remaining,
                    key=lambda row: (int(row[1].created_watermark), row[0]),
                ):
                    selected[uid] = node
            selected_uids = set(selected)
            candidate_edge_keys: set[tuple[MemoryUid, str, MemoryUid]] = set()
            for uid in selected_uids:
                candidate_edge_keys.update(self._outgoing_edge_keys.get(uid, ()))
            eligible_edges = (
                self.edges[key]
                for key in candidate_edge_keys
                if key in self.edges and self.edges[key].source in selected_uids and self.edges[key].target in selected_uids
            )
            chosen_edges = heapq.nlargest(
                maximum_edges,
                eligible_edges,
                key=lambda edge: max(
                    int(self.nodes[edge.source].created_watermark),
                    int(self.nodes[edge.target].created_watermark),
                ),
            )
            subset_edges = {edge.key: edge for edge in chosen_edges}
            subset_payloads = {uid: self.payloads.get(uid, {}) for uid in selected_uids}
            return ReadView.build(
                self.generation,
                dict(selected),
                subset_payloads,
                subset_edges,
                {},
                include_hidden_uids=historical_uids,
            )

    def memory_count(self, level: MemoryLevel | None = None) -> int:
        with self._publication_lock:
            return len(self.nodes) if level is None else len(self._uids_by_level[level])

    def uids_at_level(self, level: MemoryLevel) -> tuple[MemoryUid, ...]:
        with self._publication_lock:
            return tuple(self._uids_by_level[level])

    def _partition_state(self, partition: int):
        nodes = [
            {
                "hi": uid.hi,
                "lo": uid.lo,
                "level": int(self.nodes[uid].level),
                "memory_type": int(self.nodes[uid].memory_type),
                "structural_key": list(self.nodes[uid].structural_key),
                "created_watermark": self.nodes[uid].created_watermark,
                "payload": self.payloads.get(uid, {}),
            }
            for uid in sorted(self._node_uids_by_partition[partition])
        ]
        edges = [
            {
                "source_hi": self.edges[key].source.hi,
                "source_lo": self.edges[key].source.lo,
                "relation": self.edges[key].relation.value,
                "target_hi": self.edges[key].target.hi,
                "target_lo": self.edges[key].target.lo,
                "evidence": [[uid.hi, uid.lo] for uid in self.edges[key].evidence_uids],
                "authority": self.edges[key].authority.value,
                "object_version": self.edges[key].object_version,
            }
            for key in sorted(self._edge_keys_by_partition[partition])
            if key in self.edges
        ]
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
            tombstones = [
                {
                    "hi": uid.hi,
                    "lo": uid.lo,
                    "level": int(row.level),
                    "memory_type": int(row.memory_type),
                    "structural_key": list(row.structural_key),
                    "retired_generation": row.retired_generation,
                    "replacement": [row.replacement_uid.hi, row.replacement_uid.lo],
                    "reason": row.reason,
                }
                for uid, row in sorted(self.retired_tombstones.items())
            ]
            return {
                "schema_version": self.SCHEMA_VERSION,
                "partition_count": self.partition_count,
                "node_capacity_per_partition": self.node_capacity_per_partition,
                "edge_capacity_per_partition": self.edge_capacity_per_partition,
                "applied_proposal_capacity": self.applied_proposal_capacity,
                "generation": self.generation,
                "nodes": nodes,
                "edges": edges,
                "retired_tombstones": tombstones,
                "versions": self.versions.state_dict(),
                "applied_proposals": list(self._applied_order),
            }

    def install_sharded_rows(self, shard_rows: Any) -> None:
        """Install decoded partition rows incrementally, preserving graph indexes."""
        recent_m0: list[tuple[int, MemoryUid]] = []
        reservoir_limit = self._training_m0_reservoir.maxlen or 65_536
        for nodes, edges in shard_rows:
            for raw in nodes:
                uid = MemoryUid(int(raw["hi"]), int(raw["lo"]))
                node = CanonicalNode(uid, MemoryLevel(int(raw["level"])), MemoryType(int(raw["memory_type"])), tuple(int(v) for v in raw["structural_key"]), int(raw["created_watermark"]))
                self.nodes[uid] = node
                payload = dict(raw.get("payload", {}))
                self.payloads[uid] = payload
                owner = uid.shard(self.partition_count)
                self._node_counts_by_partition[owner] += 1
                self._node_uids_by_partition[owner].add(uid)
                self._uids_by_level[node.level].add(uid)
                if node.level is MemoryLevel.M0 and payload.get("action_id") is not None:
                    recent_m0.append((node.created_watermark, uid))
            for raw in edges:
                edge = RelationEdge(MemoryUid(int(raw["source_hi"]), int(raw["source_lo"])), RelationType(str(raw["relation"])), MemoryUid(int(raw["target_hi"]), int(raw["target_lo"])), tuple(MemoryUid(int(v[0]), int(v[1])) for v in raw.get("evidence", [])), EdgeAuthority(str(raw["authority"])), int(raw.get("object_version", 0)))
                self.edges[edge.key] = edge
                owner = edge.source.shard(self.partition_count)
                self._edge_counts_by_partition[owner] += 1
                self._edge_keys_by_partition[owner].add(edge.key)
                self._index_edge(edge)
        if recent_m0:
            merged = list((self.nodes[uid].created_watermark, uid) for uid in self._training_m0_reservoir if uid in self.nodes)
            merged.extend(recent_m0)
            self._training_m0_reservoir.clear()
            for _, uid in sorted(merged, key=lambda row: row[0])[-reservoir_limit:]:
                self._training_m0_reservoir.append(uid)

    @classmethod
    def from_sharded_state(cls, header: dict[str, object], shard_rows: Any) -> "CanonicalGraph":
        """Restore graph incrementally from partition shards."""
        state = dict(header)
        state["nodes"] = ()
        state["edges"] = ()
        result = cls.from_state_dict(state)
        result.install_sharded_rows(shard_rows)
        # Header-owned metadata must survive direct-shard restoration exactly.
        if result.generation != int(header.get("generation", 0)):
            raise RuntimeError("direct graph restore generation mismatch")
        if len(result.retired_tombstones) != len(header.get("retired_tombstones", [])):
            raise RuntimeError("direct graph restore tombstone mismatch")
        if len(result._applied_order) != min(len(header.get("applied_proposals", [])), result.applied_proposal_capacity):
            raise RuntimeError("direct graph restore applied-proposal mismatch")
        return result

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
            node = CanonicalNode(
                uid,
                MemoryLevel(int(raw["level"])),
                MemoryType(int(raw["memory_type"])),
                tuple(int(v) for v in raw["structural_key"]),
                int(raw["created_watermark"]),
            )
            result.nodes[uid] = node
            result.payloads[uid] = dict(raw.get("payload", {}))
            owner = uid.shard(result.partition_count)
            result._node_counts_by_partition[owner] += 1
            result._node_uids_by_partition[owner].add(uid)
            result._uids_by_level[node.level].add(uid)
        recent_m0 = heapq.nlargest(
            result._training_m0_reservoir.maxlen or 65_536,
            (
                (int(result.nodes[uid].created_watermark), uid)
                for uid in result._uids_by_level[MemoryLevel.M0]
                if uid in result.nodes and result.payloads.get(uid, {}).get("action_id") is not None
            ),
            key=lambda row: (row[0], row[1]),
        )
        for _, uid in reversed(recent_m0):
            result._training_m0_reservoir.append(uid)

        for raw in state.get("edges", []):
            edge = RelationEdge(
                MemoryUid(int(raw["source_hi"]), int(raw["source_lo"])),
                RelationType(str(raw["relation"])),
                MemoryUid(int(raw["target_hi"]), int(raw["target_lo"])),
                tuple(MemoryUid(int(v[0]), int(v[1])) for v in raw.get("evidence", [])),
                EdgeAuthority(str(raw["authority"])),
                int(raw.get("object_version", 0)),
            )
            result.edges[edge.key] = edge
            owner = edge.source.shard(result.partition_count)
            result._edge_counts_by_partition[owner] += 1
            result._edge_keys_by_partition[owner].add(edge.key)
            result._index_edge(edge)
        for raw in state.get("retired_tombstones", []):
            uid = MemoryUid(int(raw["hi"]), int(raw["lo"]))
            replacement = raw.get("replacement", [0, 0])
            result.retired_tombstones[uid] = RetiredTombstone(
                uid,
                MemoryLevel(int(raw["level"])),
                MemoryType(int(raw["memory_type"])),
                tuple(int(v) for v in raw.get("structural_key", [])),
                int(raw.get("retired_generation", result.generation)),
                MemoryUid(int(replacement[0]), int(replacement[1])),
                str(raw.get("reason", "retention_compaction")),
            )
        result.versions = VersionTable.from_state_dict(list(state.get("versions", [])))
        applied_order = [int(value) for value in state.get("applied_proposals", [])]
        result._applied_order = deque(applied_order[-result.applied_proposal_capacity :])
        result.applied_proposals = set(result._applied_order)
        return result
