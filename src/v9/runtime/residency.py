from __future__ import annotations

from collections import deque
from dataclasses import replace
import gc
import heapq
import math
import time
from types import SimpleNamespace
from threading import Event, Lock, Thread
from typing import Any, Iterable

from v9.memory.identity import MemoryUid, stable_u64
from v9.memory.model import MemoryLevel, MemoryType
from v9.memory.provenance import DerivationProvenance
from v9.memory.relations import RelationEdge, RelationType
from v9.mutation.transactions import MutationOutcome

from .memory_governor import MemoryGovernorState, RuntimeMemoryGovernor
from .publication import CanonicalGraph, edge_ref, node_ref
from .read_view import ReadView, _cognitively_visible


_COMPACTION_DELETE_BATCH_LIMIT = 2_048
_COMPACTION_SCAN_BATCH_LIMIT = 8_192
_PREPARED_COMPACTION_BATCH_LIMIT = 4


def _is_deletable_low_level(node: Any) -> bool:
    return (
        node.level is MemoryLevel.M0 and node.memory_type is MemoryType.EPISODE
    ) or (
        node.level is MemoryLevel.M1 and node.memory_type is MemoryType.GROUNDED_CONTINGENCY
    )


def _replacement_is_more_abstract(target: Any, replacement: Any) -> bool:
    if target.level is MemoryLevel.M0:
        return replacement.level >= MemoryLevel.M1
    if target.level is MemoryLevel.M1 and target.memory_type is MemoryType.GROUNDED_CONTINGENCY:
        return replacement.level > MemoryLevel.M1 or replacement.memory_type is MemoryType.NORMALIZED_RELATION
    return False


def _has_consolidation_path(graph: CanonicalGraph, source: MemoryUid, target: MemoryUid, *, max_hops: int = 8) -> bool:
    if source == target:
        return False
    frontier = [source]
    visited = {source}
    for _ in range(max(1, int(max_hops))):
        next_frontier: list[MemoryUid] = []
        for current in frontier:
            for child in graph._provenance_targets_by_source.get(current, ()):
                if child == target:
                    return True
                if child not in visited:
                    visited.add(child)
                    next_frontier.append(child)
        if not next_frontier:
            break
        frontier = next_frontier
    return False


def _prune_payload_uid_rows(payload: dict[str, Any], deleted: set[MemoryUid]) -> bool:
    changed = False
    deleted_pairs = {(uid.hi, uid.lo) for uid in deleted}
    for key in ("parents", "evidence_refs"):
        rows = payload.get(key)
        if not isinstance(rows, (list, tuple)):
            continue
        kept = [
            row
            for row in rows
            if not (
                isinstance(row, (list, tuple))
                and len(row) == 2
                and (int(row[0]), int(row[1])) in deleted_pairs
            )
        ]
        if len(kept) != len(rows):
            payload[key] = kept
            changed = True
    return changed


def _bounded_ancestor_sources(graph: CanonicalGraph, seeds: Iterable[MemoryUid], *, max_nodes: int = 65_536) -> set[MemoryUid]:
    result: set[MemoryUid] = set()
    frontier = deque(seeds)
    while frontier and len(result) < max(1, int(max_nodes)):
        uid = frontier.popleft()
        for source in graph._provenance_sources_by_target.get(uid, ()):
            if source in result:
                continue
            result.add(source)
            frontier.append(source)
            if len(result) >= max_nodes:
                break
    return result


def _install_graph_contract() -> None:
    if getattr(CanonicalGraph, "_v979_residency_installed", False):
        return

    original_init = CanonicalGraph.__init__
    original_publish_locked = CanonicalGraph._publish_locked
    original_state_dict = CanonicalGraph.state_dict
    original_from_state_dict = CanonicalGraph.from_state_dict

    def graph_init(self: CanonicalGraph, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self.low_level_nodes_inserted_total = 0
        self.low_level_nodes_reused_total = 0
        self.low_level_nodes_deleted_total = 0
        self.low_level_edges_deleted_total = 0
        self.low_level_delete_batches_total = 0
        self._resident_m0_count = 0
        self._resident_m1_grounded_count = 0
        self._low_level_candidate_feed: deque[tuple[MemoryUid, MemoryLevel]] = deque()
        self._low_level_candidate_deleted_feed: deque[tuple[MemoryUid, MemoryLevel]] = deque()

    def publish_locked(self: CanonicalGraph, proposal: Any):
        candidates: list[tuple[MemoryUid, MemoryLevel, MemoryType]] = []
        reused = 0
        duplicate = proposal.proposal_uid in self.applied_proposals
        if not duplicate:
            for write in proposal.writes:
                node = getattr(write, "node", None)
                if node is None or not _is_deletable_low_level(node):
                    continue
                if node.uid in self.nodes:
                    reused += 1
                    continue
                candidates.append((node.uid, node.level, node.memory_type))
        result = original_publish_locked(self, proposal)
        if result.outcome is MutationOutcome.ACCEPTED:
            self.low_level_nodes_reused_total += int(reused)
            for uid, level, memory_type in candidates:
                node = self.nodes.get(uid)
                if node is None or node.level is not level or node.memory_type is not memory_type:
                    continue
                if level is MemoryLevel.M0:
                    self._resident_m0_count += 1
                else:
                    self._resident_m1_grounded_count += 1
                self.low_level_nodes_inserted_total += 1
                self._low_level_candidate_feed.append((uid, level))
        return result

    def delete_low_level_nodes_batch(
        self: CanonicalGraph,
        plans: tuple[tuple[MemoryUid, MemoryUid, str], ...],
    ) -> tuple[MemoryUid, ...]:
        if not plans:
            return ()
        all_partitions = tuple(range(self.partition_count))
        with self._coordinator.locked(all_partitions), self._publication_lock:
            requested = {uid for uid, _, _ in plans}
            accepted: dict[MemoryUid, tuple[Any, MemoryUid, str]] = {}
            for uid, replacement_uid, reason in plans:
                node = self.nodes.get(uid)
                replacement = self.nodes.get(replacement_uid)
                if node is None or replacement is None or replacement_uid in requested:
                    continue
                if not _is_deletable_low_level(node) or not _replacement_is_more_abstract(node, replacement):
                    continue
                if not _has_consolidation_path(self, replacement_uid, uid):
                    continue
                accepted[uid] = (node, replacement_uid, str(reason))
            if not accepted:
                return ()

            deleting = set(accepted)
            replacement_uids = {replacement_uid for _, replacement_uid, _ in accepted.values()}
            affected_sources = _bounded_ancestor_sources(self, replacement_uids)
            affected_sources.update(replacement_uids)
            for uid in deleting:
                affected_sources.update(self._provenance_sources_by_target.get(uid, ()))

            incident_keys: set[tuple[MemoryUid, str, MemoryUid]] = set()
            for uid in deleting:
                incident_keys.update(self._outgoing_edge_keys.get(uid, ()))
                incident_keys.update(self._incoming_edge_keys.get(uid, ()))

            if self._durable_commit is not None:
                node_updates: dict[MemoryUid, tuple[Any, dict[str, Any]]] = {}
                edge_updates: dict[tuple[MemoryUid, str, MemoryUid], RelationEdge | None] = {
                    key: None for key in incident_keys
                }
                for source_uid in affected_sources:
                    if source_uid in deleting:
                        continue
                    node = self.nodes.get(source_uid)
                    payload = self.payloads.get(source_uid)
                    if node is not None and payload is not None:
                        pruned = dict(payload)
                        _prune_payload_uid_rows(pruned, deleting)
                        if pruned != payload:
                            node_updates[source_uid] = (node, pruned)
                    for key in tuple(self._outgoing_edge_keys.get(source_uid, ())):
                        edge = self.edges.get(key)
                        if edge is None or key in incident_keys or not edge.evidence_uids:
                            continue
                        filtered = tuple(uid for uid in edge.evidence_uids if uid not in deleting)
                        if filtered != edge.evidence_uids:
                            edge_updates[key] = replace(edge, evidence_uids=filtered)
                self._durable_commit(
                    proposal=SimpleNamespace(
                        proposal_uid=stable_u64(
                            self.generation,
                            *(uid.hex() for uid in sorted(deleting)),
                            person=b"v9-resident-delete-wal",
                        )
                    ),
                    node_updates=node_updates,
                    node_deletes={uid: accepted[uid][0] for uid in deleting},
                    edge_updates=edge_updates,
                    next_generation=self.generation + 1,
                )

            removed_edge_refs = []
            removed_edges = 0
            for key in tuple(incident_keys):
                edge = self.edges.get(key)
                if edge is None:
                    continue
                owner = edge.source.shard(self.partition_count)
                self._deindex_edge(edge)
                self.edges.pop(key, None)
                self._edge_keys_by_partition[owner].discard(key)
                self._edge_counts_by_partition[owner] -= 1
                removed_edge_refs.append(edge_ref(edge))
                removed_edges += 1

            for source_uid in affected_sources:
                payload = self.payloads.get(source_uid)
                if payload is not None:
                    _prune_payload_uid_rows(payload, deleting)
                for key in tuple(self._outgoing_edge_keys.get(source_uid, ())):
                    edge = self.edges.get(key)
                    if edge is None or not edge.evidence_uids:
                        continue
                    filtered = tuple(uid for uid in edge.evidence_uids if uid not in deleting)
                    if filtered != edge.evidence_uids:
                        self.edges[key] = replace(edge, evidence_uids=filtered)

            removed_node_refs = []
            for uid, (node, _replacement_uid, _reason) in accepted.items():
                owner = uid.shard(self.partition_count)
                self.nodes.pop(uid, None)
                self.payloads.pop(uid, None)
                self._node_uids_by_partition[owner].discard(uid)
                self._uids_by_level[node.level].discard(uid)
                self._node_counts_by_partition[owner] -= 1
                self._outgoing_edge_keys.pop(uid, None)
                self._incoming_edge_keys.pop(uid, None)
                self.retired_tombstones.pop(uid, None)
                removed_node_refs.append(node_ref(uid))
                if node.level is MemoryLevel.M0:
                    self._resident_m0_count = max(0, int(self._resident_m0_count) - 1)
                else:
                    self._resident_m1_grounded_count = max(0, int(self._resident_m1_grounded_count) - 1)
                self._low_level_candidate_deleted_feed.append((uid, node.level))

            self.versions.remove_many(removed_edge_refs)
            self.versions.remove_many(removed_node_refs)
            if deleting:
                maxlen = self._training_m0_reservoir.maxlen
                self._training_m0_reservoir = deque(
                    (uid for uid in self._training_m0_reservoir if uid not in deleting and uid in self.nodes),
                    maxlen=maxlen,
                )
            self.low_level_nodes_deleted_total += len(accepted)
            self.low_level_edges_deleted_total += removed_edges
            self.low_level_delete_batches_total += 1
            self.generation += 1
            self._cached_read_view = None
            return tuple(sorted(accepted))

    def retire_nodes_batch(self: CanonicalGraph, plans: tuple[tuple[MemoryUid, MemoryUid, str], ...]) -> tuple[MemoryUid, ...]:
        return self.delete_low_level_nodes_batch(plans)

    def bounded_view(
        self: CanonicalGraph,
        *,
        max_nodes: int = 800,
        max_edges: int = 4000,
        seed_uids: Iterable[MemoryUid] = (),
        per_level_quotas: dict[MemoryLevel, int] | None = None,
        selection_reason: str = "bounded_runtime_view",
    ) -> ReadView:
        maximum_nodes = max(1, int(max_nodes))
        maximum_edges = max(1, int(max_edges))
        seeds = tuple(dict.fromkeys(seed_uids))
        if not seeds and not per_level_quotas:
            return self.training_view(max_nodes=maximum_nodes, max_edges=maximum_edges)
        with self._publication_lock:
            selected: dict[MemoryUid, Any] = {}
            for uid in seeds:
                node = self.nodes.get(uid)
                if node is not None and _cognitively_visible(self.payloads.get(uid, {})):
                    selected[uid] = node
                    if len(selected) >= maximum_nodes:
                        break
            quotas = per_level_quotas or {}
            for level, quota in sorted(quotas.items(), key=lambda row: int(row[0])):
                if len(selected) >= maximum_nodes:
                    break
                rows = (
                    (uid, self.nodes[uid])
                    for uid in self._uids_by_level[level]
                    if uid in self.nodes and uid not in selected and _cognitively_visible(self.payloads.get(uid, {}))
                )
                for uid, node in heapq.nlargest(
                    min(maximum_nodes - len(selected), max(0, int(quota))),
                    rows,
                    key=lambda row: (int(row[1].created_watermark), row[0]),
                ):
                    selected[uid] = node
            frontier = deque(selected)
            while frontier and len(selected) < maximum_nodes:
                uid = frontier.popleft()
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
            edge_keys: set[tuple[MemoryUid, str, MemoryUid]] = set()
            for uid in selected_uids:
                edge_keys.update(self._outgoing_edge_keys.get(uid, ()))
            eligible = (
                self.edges[key]
                for key in edge_keys
                if key in self.edges
                and self.edges[key].source in selected_uids
                and self.edges[key].target in selected_uids
            )
            chosen = heapq.nlargest(
                maximum_edges,
                eligible,
                key=lambda edge: max(
                    int(self.nodes[edge.source].created_watermark),
                    int(self.nodes[edge.target].created_watermark),
                ),
            )
            payloads = {uid: self.payloads.get(uid, {}) for uid in selected_uids}
            return ReadView.build(
                self.generation,
                dict(selected),
                payloads,
                {edge.key: edge for edge in chosen},
                {},
            )

    def state_dict(self: CanonicalGraph) -> dict[str, object]:
        state = original_state_dict(self)
        # Deleted concrete evidence is absent from live cognition; snapshots retain
        # only aggregate deletion accounting rather than UID tombstones.
        state["retired_tombstones"] = []
        state["low_level_nodes_inserted_total"] = int(self.low_level_nodes_inserted_total)
        state["low_level_nodes_reused_total"] = int(self.low_level_nodes_reused_total)
        state["low_level_nodes_deleted_total"] = int(self.low_level_nodes_deleted_total)
        state["low_level_edges_deleted_total"] = int(self.low_level_edges_deleted_total)
        state["low_level_delete_batches_total"] = int(self.low_level_delete_batches_total)
        return state

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> CanonicalGraph:
        result = original_from_state_dict(state)
        result.low_level_nodes_inserted_total = int(state.get("low_level_nodes_inserted_total", 0))
        result.low_level_nodes_reused_total = int(state.get("low_level_nodes_reused_total", 0))
        result.low_level_nodes_deleted_total = int(state.get("low_level_nodes_deleted_total", 0))
        result.low_level_edges_deleted_total = int(state.get("low_level_edges_deleted_total", 0))
        result.low_level_delete_batches_total = int(state.get("low_level_delete_batches_total", 0))
        if not hasattr(result, "_low_level_candidate_feed"):
            result._low_level_candidate_feed = deque()
        if not hasattr(result, "_low_level_candidate_deleted_feed"):
            result._low_level_candidate_deleted_feed = deque()
        return result

    CanonicalGraph.__init__ = graph_init
    CanonicalGraph._publish_locked = publish_locked
    CanonicalGraph.delete_low_level_nodes_batch = delete_low_level_nodes_batch
    CanonicalGraph.retire_nodes_batch = retire_nodes_batch
    CanonicalGraph.bounded_view = bounded_view
    CanonicalGraph.state_dict = state_dict
    CanonicalGraph.from_state_dict = from_state_dict
    CanonicalGraph._v979_residency_installed = True


class ResidentMemoryManager:
    """Capacity-driven representative retention for concrete M0/M1 evidence."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.scientific = runtime.config.scientific
        self.governor = RuntimeMemoryGovernor(self.scientific)
        self.compaction_cycles = 0
        self.compaction_seconds = 0.0
        self.compaction_planning_seconds = 0.0
        self.compaction_delete_seconds = 0.0
        self.startup_compaction_seconds = 0.0
        self.last_deleted = 0
        self.last_edges_deleted = 0
        self.last_insert_check = 0
        self._last_snapshot = None
        self._candidate_lock = Lock()
        self._planning_lock = Lock()
        self._prepared_lock = Lock()
        self._candidate_queues = {
            MemoryLevel.M0: deque(),
            MemoryLevel.M1: deque(),
        }
        self._candidate_sets = {
            MemoryLevel.M0: set(),
            MemoryLevel.M1: set(),
        }
        self._replacement_cache: dict[MemoryUid, MemoryUid] = {}
        self._candidate_levels: dict[MemoryUid, MemoryLevel] = {}
        self._replacement_group_counts: dict[tuple[MemoryUid, MemoryLevel], int] = {}
        self._prepared_plans: deque[tuple[tuple[MemoryUid, MemoryUid, str], ...]] = deque()
        self._prepared_uids: set[MemoryUid] = set()
        self._planner_wakeup = Event()
        self._planner_stop = Event()
        self._planner_thread: Thread | None = None
        self._planner_retry_after = 0.0
        self._recount_graph()
        self._seed_candidate_queues()
        legacy_tombstones = len(runtime.graph.retired_tombstones)
        if legacy_tombstones:
            runtime.graph.low_level_nodes_deleted_total += legacy_tombstones
            runtime.graph.retired_tombstones.clear()
        self.last_insert_check = int(runtime.graph.low_level_nodes_inserted_total)

    def _candidate_type(self, level: MemoryLevel) -> MemoryType:
        return MemoryType.EPISODE if level is MemoryLevel.M0 else MemoryType.GROUNDED_CONTINGENCY

    def _enqueue_candidate_locked(self, level: MemoryLevel, uid: MemoryUid) -> None:
        if level not in self._candidate_queues or uid in self._candidate_sets[level]:
            return
        self._candidate_queues[level].append(uid)
        self._candidate_sets[level].add(uid)

    def _enqueue_candidate(self, level: MemoryLevel, uid: MemoryUid) -> None:
        with self._candidate_lock:
            self._enqueue_candidate_locked(level, uid)

    def _seed_candidate_queues(self) -> None:
        graph = self.runtime.graph
        with self._candidate_lock:
            for level in (MemoryLevel.M0, MemoryLevel.M1):
                expected = self._candidate_type(level)
                for uid in tuple(graph._uids_by_level[level]):
                    node = graph.nodes.get(uid)
                    if node is not None and node.memory_type is expected:
                        self._enqueue_candidate_locked(level, uid)

    def _forget_candidate_locked(self, level: MemoryLevel, uid: MemoryUid) -> None:
        self._candidate_sets.get(level, set()).discard(uid)
        replacement_uid = self._replacement_cache.pop(uid, None)
        self._candidate_levels.pop(uid, None)
        if replacement_uid is None:
            return
        key = (replacement_uid, level)
        remaining = max(0, int(self._replacement_group_counts.get(key, 0)) - 1)
        if remaining:
            self._replacement_group_counts[key] = remaining
        else:
            self._replacement_group_counts.pop(key, None)

    def _drain_candidate_feeds(self) -> None:
        graph = self.runtime.graph
        inserted: list[tuple[MemoryUid, MemoryLevel]] = []
        deleted: list[tuple[MemoryUid, MemoryLevel]] = []
        feed = getattr(graph, "_low_level_candidate_feed", None)
        deleted_feed = getattr(graph, "_low_level_candidate_deleted_feed", None)
        if feed is not None:
            while feed:
                inserted.append(feed.popleft())
        if deleted_feed is not None:
            while deleted_feed:
                deleted.append(deleted_feed.popleft())
        if not inserted and not deleted:
            return
        with self._candidate_lock:
            for uid, level in deleted:
                self._forget_candidate_locked(level, uid)
            for uid, level in inserted:
                node = graph.nodes.get(uid)
                if node is not None and node.level is level and node.memory_type is self._candidate_type(level):
                    self._enqueue_candidate_locked(level, uid)

    def start_background_compaction(self) -> None:
        if self._planner_thread is not None:
            return
        self._planner_thread = Thread(
            target=self._planner_loop,
            name="v9-resident-compaction-planner",
            daemon=True,
        )
        self._planner_thread.start()

    def stop_background_compaction(self, *, timeout: float = 5.0) -> None:
        self._planner_stop.set()
        self._planner_wakeup.set()
        thread = self._planner_thread
        if thread is not None:
            thread.join(timeout=max(0.0, float(timeout)))
        self._planner_thread = None

    def request_compaction(self, *, force: bool = False) -> bool:
        graph = self.runtime.graph
        inserts = int(graph.low_level_nodes_inserted_total)
        interval_due = (
            inserts - int(self.last_insert_check)
            >= int(self.scientific.resident_compaction_check_interval)
        )
        m0, m1g = self.counts()
        hard_limit = (
            m0 > int(self.scientific.resident_m0_limit)
            or m1g > int(self.scientific.resident_m1_grounded_limit)
        )
        if not (force or interval_due or hard_limit):
            return False
        self.last_insert_check = inserts
        if time.monotonic() >= float(self._planner_retry_after):
            self._planner_wakeup.set()
        return True

    def _enqueue_prepared_plans(
        self, plans: tuple[tuple[MemoryUid, MemoryUid, str], ...]
    ) -> int:
        if not plans:
            return 0
        with self._prepared_lock:
            if len(self._prepared_plans) >= _PREPARED_COMPACTION_BATCH_LIMIT:
                return 0
            selected = tuple(
                plan for plan in plans if plan[0] not in self._prepared_uids
            )
            if not selected:
                return 0
            self._prepared_plans.append(selected)
            self._prepared_uids.update(uid for uid, _, _ in selected)
            return len(selected)

    def _planner_loop(self) -> None:
        while not self._planner_stop.is_set():
            self._planner_wakeup.wait(0.5)
            self._planner_wakeup.clear()
            if self._planner_stop.is_set():
                break
            while not self._planner_stop.is_set() and self.backlog() > 0:
                with self._prepared_lock:
                    prepared_full = len(self._prepared_plans) >= _PREPARED_COMPACTION_BATCH_LIMIT
                if prepared_full:
                    break
                plans = self._plan_compaction_batch(force=True)
                if not plans:
                    self._planner_retry_after = time.monotonic() + 0.10
                    break
                self._enqueue_prepared_plans(plans)

    def service_prepared_compaction(self, *, max_batches: int = 1) -> int:
        total = 0
        for _ in range(max(0, int(max_batches))):
            with self._prepared_lock:
                if not self._prepared_plans:
                    break
                plans = self._prepared_plans.popleft()
                for uid, _, _ in plans:
                    self._prepared_uids.discard(uid)
            total += self._apply_compaction_plans(plans)
        if self.backlog() > 0:
            self._planner_wakeup.set()
        return total

    def sample_memory(self):
        return self.governor.sample(
            backlog=self.backlog(),
            tracked_shm_bytes=max(0, int(self.runtime.__dict__.get("_tracked_shm_bytes", 0))),
        )

    def _recount_graph(self) -> None:
        graph = self.runtime.graph
        graph._resident_m0_count = len(graph._uids_by_level[MemoryLevel.M0])
        graph._resident_m1_grounded_count = sum(
            1
            for uid in graph._uids_by_level[MemoryLevel.M1]
            if (node := graph.nodes.get(uid)) is not None
            and node.memory_type is MemoryType.GROUNDED_CONTINGENCY
        )

    def counts(self) -> tuple[int, int]:
        graph = self.runtime.graph
        return int(graph._resident_m0_count), int(graph._resident_m1_grounded_count)

    def targets(self) -> tuple[int, int]:
        ratio = float(self.scientific.resident_low_level_target_ratio)
        return (
            max(int(self.scientific.resident_m0_representative_floor), int(int(self.scientific.resident_m0_limit) * ratio)),
            max(int(self.scientific.resident_m1_grounded_representative_floor), int(int(self.scientific.resident_m1_grounded_limit) * ratio)),
        )

    def backlog(self) -> int:
        m0, m1g = self.counts()
        target_m0, target_m1g = self.targets()
        return max(0, m0 - target_m0) + max(0, m1g - target_m1g)

    def _replacement_for(self, uid: MemoryUid) -> MemoryUid | None:
        graph = self.runtime.graph
        node = graph.nodes.get(uid)
        if node is None or not _is_deletable_low_level(node):
            return None

        # Follow the consolidation ancestry instead of stopping at the nearest
        # M1N. M2+ families may cover many otherwise-singleton M1N rows and are
        # therefore the correct replacement scope for redundant concrete
        # evidence.
        candidates: set[MemoryUid] = set()
        visited = {uid}
        frontier = deque([uid])
        for _ in range(6):
            next_frontier: deque[MemoryUid] = deque()
            while frontier:
                current = frontier.popleft()
                for source_uid in tuple(graph._provenance_sources_by_target.get(current, ())):
                    if source_uid in visited:
                        continue
                    visited.add(source_uid)
                    source = graph.nodes.get(source_uid)
                    if source is None:
                        continue
                    if source.memory_type is MemoryType.NORMALIZED_RELATION or source.level > MemoryLevel.M1:
                        candidates.add(source_uid)
                    next_frontier.append(source_uid)
            if not next_frontier:
                break
            frontier = next_frontier
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda candidate_uid: (
                int(graph.nodes[candidate_uid].level),
                int(graph.nodes[candidate_uid].memory_type is not MemoryType.NORMALIZED_RELATION),
                candidate_uid,
            ),
        )

    def _group_size(self, replacement_uid: MemoryUid, level: MemoryLevel, *, limit: int = 8_192) -> int:
        graph = self.runtime.graph
        maximum = max(1, int(limit))
        count = 0
        visited = {replacement_uid}
        frontier = deque([replacement_uid])
        while frontier and count < maximum:
            current = frontier.popleft()
            for child_uid in tuple(graph._provenance_targets_by_source.get(current, ())):
                if child_uid in visited:
                    continue
                visited.add(child_uid)
                child = graph.nodes.get(child_uid)
                if child is None:
                    continue
                matches = (
                    child.level is MemoryLevel.M0
                    and child.memory_type is MemoryType.EPISODE
                    if level is MemoryLevel.M0
                    else child.level is MemoryLevel.M1
                    and child.memory_type is MemoryType.GROUNDED_CONTINGENCY
                )
                if matches:
                    count += 1
                    if count >= maximum:
                        break
                else:
                    frontier.append(child_uid)
        return count

    def _score_candidates(self, rows: list[tuple[MemoryUid, Any, dict[str, Any], MemoryUid]]) -> list[tuple[float, int, MemoryUid, MemoryUid]]:
        context_counts: dict[tuple[int, int], int] = {}
        structure_counts: dict[tuple[int, ...], int] = {}
        training_m0 = set(self.runtime.graph.training_m0_reservoir_snapshot())
        for _uid, node, payload, _replacement in rows:
            context = int(payload.get("context_signature", payload.get("grounded_context_signature", 0)) or 0)
            action = int(payload.get("action_id", payload.get("executable_action_token", 0)) or 0)
            context_counts[(context, action)] = context_counts.get((context, action), 0) + 1
            structure_counts[node.structural_key] = structure_counts.get(node.structural_key, 0) + 1

        watermark = max(1, int(self.runtime.watermark))
        scored: list[tuple[float, int, MemoryUid, MemoryUid]] = []
        for uid, node, payload, replacement_uid in rows:
            age = max(0, watermark - int(node.created_watermark))
            recency = math.exp(-age / 8192.0)
            valence = min(1.0, abs(float(payload.get("primary_valence", 0) or 0)))
            boundary = float(
                bool(payload.get("task_success", False))
                or bool(payload.get("task_failure", False))
                or bool(payload.get("task_truncated", False))
            )
            prediction_error = min(1.0, abs(float(payload.get("prediction_error", 0.0) or 0.0)))
            future_options = min(1.0, abs(float(payload.get("future_option_delta", 0.0) or 0.0)) / 8.0)
            context = int(payload.get("context_signature", payload.get("grounded_context_signature", 0)) or 0)
            action = int(payload.get("action_id", payload.get("executable_action_token", 0)) or 0)
            rarity = 1.0 / max(1, context_counts.get((context, action), 1))
            structural_rarity = 1.0 / max(1, structure_counts.get(node.structural_key, 1))
            ancestor_count = len(self.runtime.graph._provenance_sources_by_target.get(replacement_uid, ()))
            descendant_support = min(1.0, ancestor_count / 4.0)
            hgt_reservoir = float(uid in training_m0)
            score = (
                0.25 * recency
                + 0.15 * valence
                + 0.15 * boundary
                + 0.10 * prediction_error
                + 0.10 * future_options
                + 0.08 * rarity
                + 0.05 * structural_rarity
                + 0.07 * descendant_support
                + 0.05 * hgt_reservoir
            )
            scored.append((score, int(node.created_watermark), uid, replacement_uid))
        scored.sort(key=lambda row: (row[0], row[1], row[2]))
        return scored

    def _recover_candidate_queue(self, level: MemoryLevel, limit: int) -> int:
        graph = self.runtime.graph
        maximum = max(1, int(limit))
        expected = self._candidate_type(level)
        recovered: list[MemoryUid] = []
        # Recovery is a bounded fallback only when the incremental feed/queue is
        # empty while resident counts say compaction work still exists.
        with graph._publication_lock:
            for uid in graph._uids_by_level[level]:
                node = graph.nodes.get(uid)
                if node is not None and node.memory_type is expected:
                    recovered.append(uid)
                    if len(recovered) >= maximum:
                        break
        for uid in recovered:
            self._enqueue_candidate(level, uid)
        if recovered:
            telemetry = self.runtime.telemetry
            telemetry["compaction_candidate_recovery_scans"] = int(
                telemetry.get("compaction_candidate_recovery_scans", 0)
            ) + 1
            telemetry["compaction_candidates_recovered"] = int(
                telemetry.get("compaction_candidates_recovered", 0)
            ) + len(recovered)
        return len(recovered)

    def _candidate_rows(self, level: MemoryLevel, limit: int) -> list[tuple[MemoryUid, Any, dict[str, Any], MemoryUid]]:
        graph = self.runtime.graph
        self._drain_candidate_feeds()
        maximum = max(1, int(limit))
        with self._candidate_lock:
            queue_empty = not self._candidate_queues[level]
        if queue_empty:
            current_m0, current_m1g = self.counts()
            target_m0, target_m1g = self.targets()
            current = current_m0 if level is MemoryLevel.M0 else current_m1g
            target = target_m0 if level is MemoryLevel.M0 else target_m1g
            if current > target:
                self._recover_candidate_queue(level, maximum)

        selected_uids: list[MemoryUid] = []
        with self._candidate_lock:
            queue = self._candidate_queues[level]
            attempts = min(maximum, len(queue))
            for _ in range(attempts):
                uid = queue.popleft()
                if uid not in self._candidate_sets[level]:
                    continue
                self._candidate_sets[level].discard(uid)
                selected_uids.append(uid)

        rows: list[tuple[MemoryUid, Any, dict[str, Any], MemoryUid]] = []
        unresolved: list[MemoryUid] = []
        expected = self._candidate_type(level)
        for uid in selected_uids:
            node = graph.nodes.get(uid)
            if node is None or node.level is not level or node.memory_type is not expected:
                with self._candidate_lock:
                    self._forget_candidate_locked(level, uid)
                continue
            with self._candidate_lock:
                cached = self._replacement_cache.get(uid)
            replacement_uid = cached
            if replacement_uid is None or replacement_uid not in graph.nodes:
                if cached is not None:
                    with self._candidate_lock:
                        self._forget_candidate_locked(level, uid)
                replacement_uid = self._replacement_for(uid)
                if replacement_uid is None:
                    unresolved.append(uid)
                    continue
                with self._candidate_lock:
                    self._replacement_cache[uid] = replacement_uid
                    self._candidate_levels[uid] = level
                    key = (replacement_uid, level)
                    self._replacement_group_counts[key] = (
                        int(self._replacement_group_counts.get(key, 0)) + 1
                    )
            rows.append((uid, node, graph.payloads.get(uid, {}), replacement_uid))

        for uid in unresolved:
            self._enqueue_candidate(level, uid)
        return rows

    def _effective_group_floor(self, level: MemoryLevel, group_size: int) -> int:
        size = max(0, int(group_size))
        if size <= 1:
            return size
        m0, m1g = self.counts()
        target_m0, target_m1g = self.targets()
        current = m0 if level is MemoryLevel.M0 else m1g
        target = target_m0 if level is MemoryLevel.M0 else target_m1g
        configured = (
            int(self.scientific.resident_m0_representative_floor)
            if level is MemoryLevel.M0
            else int(self.scientific.resident_m1_grounded_representative_floor)
        )
        if current <= target:
            return min(configured, size)
        # Capacity pressure must be able to shrink recurrent evidence groups.
        # Scale the representative set with the level-wide target while always
        # retaining at least one concrete witness for a non-singleton group.
        target_fraction = min(1.0, max(0.0, float(target) / max(1.0, float(current))))
        capacity_floor = max(1, int(math.floor(size * target_fraction)))
        return min(configured, capacity_floor, size - 1)

    def _plans_for_level(self, level: MemoryLevel, required: int, scan_budget: int) -> list[tuple[MemoryUid, MemoryUid, str]]:
        if required <= 0:
            return []
        rows = self._candidate_rows(level, scan_budget)
        scored = self._score_candidates(rows)
        planned_by_group: dict[MemoryUid, int] = {}
        group_floors: dict[MemoryUid, int] = {}
        group_sizes: dict[MemoryUid, int] = {}
        plans: list[tuple[MemoryUid, MemoryUid, str]] = []
        selected: set[MemoryUid] = set()
        with self._candidate_lock:
            observed_group_sizes = dict(self._replacement_group_counts)
        for _score, _created, uid, replacement_uid in scored:
            if len(plans) >= required:
                break
            if replacement_uid not in group_sizes:
                observed = max(
                    0, int(observed_group_sizes.get((replacement_uid, level), 0))
                )
                # Incremental accounting can be incomplete after restore, bulk
                # publication, or a missed feed. Verify small/unknown groups by
                # bounded provenance traversal; this never scans the whole level.
                if observed <= 1:
                    observed = max(
                        observed,
                        self._group_size(
                            replacement_uid,
                            level,
                            limit=max(64, min(int(scan_budget), int(required) + 16)),
                        ),
                    )
                group_sizes[replacement_uid] = max(1, observed)
                group_floors[replacement_uid] = self._effective_group_floor(
                    level, group_sizes[replacement_uid]
                )
            already = planned_by_group.get(replacement_uid, 0)
            if group_sizes[replacement_uid] - already <= group_floors[replacement_uid]:
                continue
            plans.append((uid, replacement_uid, "resident_capacity_compaction"))
            selected.add(uid)
            planned_by_group[replacement_uid] = already + 1

        for uid, _node, _payload, _replacement_uid in rows:
            if uid not in selected:
                self._enqueue_candidate(level, uid)
        return plans

    def _plan_compaction_batch(
        self, *, force: bool = False
    ) -> tuple[tuple[MemoryUid, MemoryUid, str], ...]:
        graph = self.runtime.graph
        m0, m1g = self.counts()
        target_m0, target_m1g = self.targets()
        excess_m0 = max(0, m0 - target_m0)
        excess_m1g = max(0, m1g - target_m1g)
        if not force and excess_m0 + excess_m1g <= 0:
            return ()
        maximum_delete = min(
            int(self.scientific.resident_max_delete_batch),
            _COMPACTION_DELETE_BATCH_LIMIT,
        )
        maximum_scan = min(
            int(self.scientific.resident_max_scan_batch),
            _COMPACTION_SCAN_BATCH_LIMIT,
        )
        budget_m0 = min(excess_m0, maximum_delete)
        budget_m1 = min(excess_m1g, max(0, maximum_delete - budget_m0))
        if budget_m1 == 0 and excess_m1g > 0 and budget_m0 < maximum_delete:
            budget_m1 = min(excess_m1g, maximum_delete - budget_m0)
        scan_m0 = min(maximum_scan, max(2048, budget_m0 * 2)) if budget_m0 else 0
        scan_m1 = min(maximum_scan, max(2048, budget_m1 * 2)) if budget_m1 else 0

        started = time.perf_counter()
        with self._planning_lock:
            plans = self._plans_for_level(MemoryLevel.M0, budget_m0, scan_m0)
            remaining = max(0, maximum_delete - len(plans))
            plans.extend(
                self._plans_for_level(
                    MemoryLevel.M1, min(excess_m1g, remaining), scan_m1
                )
            )
        elapsed = time.perf_counter() - started
        self.compaction_planning_seconds += elapsed
        self.compaction_seconds += elapsed
        return tuple(plans)

    def _apply_compaction_plans(
        self, plans: tuple[tuple[MemoryUid, MemoryUid, str], ...]
    ) -> int:
        if not plans:
            return 0
        graph = self.runtime.graph
        planned_uids = tuple(uid for uid, _replacement, _reason in plans)
        affected_signatures: set[int] = set()
        for ancestor_uid in _bounded_ancestor_sources(
            graph, planned_uids, max_nodes=max(4096, len(planned_uids) * 16)
        ):
            node = graph.nodes.get(ancestor_uid)
            if node is None or node.memory_type is not MemoryType.NORMALIZED_RELATION:
                continue
            payload = graph.payloads.get(ancestor_uid, {})
            signature = payload.get("structural_signature")
            if signature is not None:
                affected_signatures.add(int(signature))
        before_edges = int(graph.low_level_edges_deleted_total)
        started = time.perf_counter()
        deleted = graph.delete_low_level_nodes_batch(plans)
        if deleted:
            self.runtime.on_low_level_deleted(
                deleted, affected_signatures=affected_signatures
            )
        elapsed = time.perf_counter() - started
        self.compaction_delete_seconds += elapsed
        self.compaction_seconds += elapsed
        if deleted:
            self.compaction_cycles += 1
            self.last_deleted = len(deleted)
            self.last_edges_deleted = int(graph.low_level_edges_deleted_total) - before_edges
        deleted_set = set(deleted)
        with self._candidate_lock:
            for uid in deleted:
                replacement_uid = self._replacement_cache.pop(uid, None)
                level = self._candidate_levels.pop(uid, None)
                if replacement_uid is None or level is None:
                    continue
                key = (replacement_uid, level)
                remaining = max(0, int(self._replacement_group_counts.get(key, 0)) - 1)
                if remaining:
                    self._replacement_group_counts[key] = remaining
                else:
                    self._replacement_group_counts.pop(key, None)
        for uid, _replacement_uid, _reason in plans:
            if uid not in deleted_set:
                node = graph.nodes.get(uid)
                if node is not None and _is_deletable_low_level(node):
                    self._enqueue_candidate(node.level, uid)
        self._drain_candidate_feeds()
        self._publish_telemetry()
        return len(deleted)

    def compact_once(self, *, force: bool = False) -> int:
        plans = self._plan_compaction_batch(force=force)
        return self._apply_compaction_plans(plans)

    def maybe_compact(self, *, force: bool = False) -> int:
        graph = self.runtime.graph
        inserts = int(graph.low_level_nodes_inserted_total)
        interval_due = (
            inserts - self.last_insert_check
            >= int(self.scientific.resident_compaction_check_interval)
        )
        m0, m1g = self.counts()
        hard_limit = (
            m0 > int(self.scientific.resident_m0_limit)
            or m1g > int(self.scientific.resident_m1_grounded_limit)
        )
        snapshot = self.sample_memory()
        self._last_snapshot = snapshot
        pressure = snapshot.state in {
            MemoryGovernorState.COMPACTING,
            MemoryGovernorState.HARD_PRESSURE_DRAIN,
        }
        if not (force or interval_due or hard_limit or pressure):
            self._publish_telemetry()
            return 0
        self.last_insert_check = inserts
        total = self.service_prepared_compaction(max_batches=1)
        passes = 2 if snapshot.state is MemoryGovernorState.HARD_PRESSURE_DRAIN else 1
        for _ in range(passes):
            deleted = self.compact_once(force=True)
            total += deleted
            if deleted <= 0 or self.backlog() <= 0:
                break
        if snapshot.state is MemoryGovernorState.HARD_PRESSURE_DRAIN:
            gc.collect()
        self._publish_telemetry()
        return total

    def startup_compact(self) -> dict[str, int | float]:
        started = time.perf_counter()
        pre_m0, pre_m1g = self.counts()
        before_deleted = int(self.runtime.graph.low_level_nodes_deleted_total)
        before_edges = int(self.runtime.graph.low_level_edges_deleted_total)
        before_memory = self.sample_memory()
        passes = 0
        while self.backlog() > 0 and passes < 128:
            passes += 1
            if self.compact_once(force=True) <= 0:
                break
        gc.collect()
        post_m0, post_m1g = self.counts()
        after_memory = self.sample_memory()
        self._last_snapshot = after_memory
        self.startup_compaction_seconds = time.perf_counter() - started
        result: dict[str, int | float] = {
            "pre_migration_M0": pre_m0,
            "pre_migration_M1G": pre_m1g,
            "post_migration_M0": post_m0,
            "post_migration_M1G": post_m1g,
            "nodes_deleted": int(self.runtime.graph.low_level_nodes_deleted_total) - before_deleted,
            "edges_deleted": int(self.runtime.graph.low_level_edges_deleted_total) - before_edges,
            "migration_seconds": self.startup_compaction_seconds,
            "RSS_before": before_memory.rss_bytes,
            "RSS_after": after_memory.rss_bytes,
        }
        self._publish_telemetry()
        return result

    def _publish_telemetry(self) -> None:
        graph = self.runtime.graph
        snapshot = self._last_snapshot or self.sample_memory()
        m0, m1g = self.counts()
        target_m0, target_m1g = self.targets()
        gauges = getattr(self.runtime.unified_telemetry, "gauges", None)
        if isinstance(gauges, dict):
            gauges.update({
                "M0_resident": m0,
                "M1_grounded_resident": m1g,
                "resident_M0_limit": int(self.scientific.resident_m0_limit),
                "resident_M1_grounded_limit": int(self.scientific.resident_m1_grounded_limit),
                "resident_M0_target": target_m0,
                "resident_M1_grounded_target": target_m1g,
                "compaction_backlog": self.backlog(),
                "low_level_nodes_inserted": int(graph.low_level_nodes_inserted_total),
                "low_level_nodes_reused": int(graph.low_level_nodes_reused_total),
                "low_level_dedup_rate": float(graph.low_level_nodes_reused_total) / max(
                    1, int(graph.low_level_nodes_reused_total) + int(graph.low_level_nodes_inserted_total)
                ),
                "low_level_nodes_deleted": int(graph.low_level_nodes_deleted_total),
                "low_level_edges_deleted": int(graph.low_level_edges_deleted_total),
                "low_level_delete_batches": int(graph.low_level_delete_batches_total),
                "compaction_cycles": int(self.compaction_cycles),
                "compaction_seconds": float(self.compaction_seconds),
                "compaction_planning_seconds": float(self.compaction_planning_seconds),
                "compaction_delete_seconds": float(self.compaction_delete_seconds),
                "compaction_prepared_batches": len(self._prepared_plans),
                "compaction_candidate_queue_depth": sum(len(queue) for queue in self._candidate_queues.values()),
                "startup_compaction_seconds": float(self.startup_compaction_seconds),
                "process_rss_bytes": int(snapshot.rss_bytes),
                "process_uss_bytes": int(snapshot.uss_bytes),
                "process_swap_bytes": int(snapshot.swap_bytes),
                "memory_governor_state": snapshot.state.value,
                "concrete_admission_retained_events": int(
                    self.runtime.telemetry.get("concrete_admission_retained_events", 0)
                ),
                "concrete_admission_skipped_events": int(
                    self.runtime.telemetry.get("concrete_admission_skipped_events", 0)
                ),
                "concrete_nodes_avoided": int(
                    self.runtime.telemetry.get("concrete_nodes_avoided", 0)
                ),
                "concrete_admission_retention_rate": float(
                    self.runtime.telemetry.get("concrete_admission_retained_events", 0)
                )
                / max(
                    1,
                    int(self.runtime.telemetry.get("concrete_admission_retained_events", 0))
                    + int(self.runtime.telemetry.get("concrete_admission_skipped_events", 0)),
                ),
            })
            gauges.update(self.governor.state_dict())

    def metrics(self) -> dict[str, int | float | str]:
        self._publish_telemetry()
        snapshot = self._last_snapshot or self.sample_memory()
        m0, m1g = self.counts()
        return {
            "M0_resident": m0,
            "M1_grounded_resident": m1g,
            "resident_M0_limit": int(self.scientific.resident_m0_limit),
            "resident_M1_grounded_limit": int(self.scientific.resident_m1_grounded_limit),
            "compaction_backlog": self.backlog(),
            "low_level_nodes_inserted": int(self.runtime.graph.low_level_nodes_inserted_total),
            "low_level_nodes_reused": int(self.runtime.graph.low_level_nodes_reused_total),
            "low_level_dedup_rate": float(self.runtime.graph.low_level_nodes_reused_total) / max(
                1, int(self.runtime.graph.low_level_nodes_reused_total) + int(self.runtime.graph.low_level_nodes_inserted_total)
            ),
            "low_level_nodes_deleted": int(self.runtime.graph.low_level_nodes_deleted_total),
            "low_level_edges_deleted": int(self.runtime.graph.low_level_edges_deleted_total),
            "compaction_cycles": int(self.compaction_cycles),
            "compaction_seconds": float(self.compaction_seconds),
            "compaction_planning_seconds": float(self.compaction_planning_seconds),
            "compaction_delete_seconds": float(self.compaction_delete_seconds),
            "compaction_prepared_batches": len(self._prepared_plans),
            "compaction_candidate_queue_depth": sum(len(queue) for queue in self._candidate_queues.values()),
            "startup_compaction_seconds": float(self.startup_compaction_seconds),
            "process_rss_bytes": int(snapshot.rss_bytes),
            "process_uss_bytes": int(snapshot.uss_bytes),
            "process_swap_bytes": int(snapshot.swap_bytes),
            "memory_governor_state": snapshot.state.value,
            "concrete_admission_retained_events": int(
                self.runtime.telemetry.get("concrete_admission_retained_events", 0)
            ),
            "concrete_admission_skipped_events": int(
                self.runtime.telemetry.get("concrete_admission_skipped_events", 0)
            ),
            "concrete_nodes_avoided": int(
                self.runtime.telemetry.get("concrete_nodes_avoided", 0)
            ),
            "concrete_admission_retention_rate": float(
                self.runtime.telemetry.get("concrete_admission_retained_events", 0)
            )
            / max(
                1,
                int(self.runtime.telemetry.get("concrete_admission_retained_events", 0))
                + int(self.runtime.telemetry.get("concrete_admission_skipped_events", 0)),
            ),
        }


def _install_runtime_contract(runtime_cls: type) -> None:
    if getattr(runtime_cls, "_v979_residency_installed", False):
        return

    original_init = runtime_cls.__init__
    original_flush = runtime_cls.flush_deferred_memory_updates
    original_apply_batch = runtime_cls.apply_prepared_ingestion_batch
    original_dashboard = runtime_cls.dashboard_metrics
    original_full_metrics = runtime_cls.full_metrics
    original_close = runtime_cls.close

    def runtime_init(self: Any, config: Any) -> None:
        original_init(self, config)
        self._resident_memory = ResidentMemoryManager(self)
        migration = self._resident_memory.startup_compact()
        self._resident_memory.start_background_compaction()
        if int(migration["nodes_deleted"]) > 0:
            print(
                f"{time.strftime('[%H:%M]')} startup compaction "
                f"M0={migration['pre_migration_M0']}->{migration['post_migration_M0']} "
                f"M1G={migration['pre_migration_M1G']}->{migration['post_migration_M1G']} "
                f"deleted={migration['nodes_deleted']} seconds={migration['migration_seconds']:.2f}",
                flush=True,
            )

    def on_low_level_deleted(
        self: Any,
        deleted_uids: Iterable[MemoryUid],
        *,
        affected_signatures: Iterable[int] = (),
    ) -> None:
        deleted = set(deleted_uids)
        if not deleted:
            return
        registry = self.lifecycle
        for uid in deleted:
            registry.remove(uid)
            self._replay_pool.pop(uid, None)
            self._deferred_base_nodes.pop(uid, None)
            self._transfer_trials.pop(uid, None)

        environment_index = getattr(self, "_memory_uids_by_environment", None)
        if environment_index is not None:
            for environment_id in tuple(environment_index):
                environment_index[environment_id].difference_update(deleted)
                if not environment_index[environment_id]:
                    del environment_index[environment_id]
        self._latest_interaction_grounding = {
            key: row
            for key, row in self._latest_interaction_grounding.items()
            if row.uid not in deleted
        }

        deleted_lo = {int(uid.lo) for uid in deleted}
        grounding_states = getattr(getattr(self, "grounding", None), "states", None)
        if isinstance(grounding_states, dict) and deleted_lo:
            for key in tuple(grounding_states):
                if len(key) >= 2 and (
                    int(key[0]) in deleted_lo or int(key[1]) in deleted_lo
                ):
                    grounding_states.pop(key, None)

        # Rebuild only normalized occurrence samples whose concrete provenance
        # intersects this deletion batch. Aggregate support remains authoritative.
        family_index = getattr(self, "_m1n_family_occurrences", None)
        for signature in {int(value) for value in affected_signatures}:
            rows = self._m1n_occurrences.get(signature, ())
            if not rows:
                continue
            template = rows[0]
            surviving = tuple(
                uid
                for uid in tuple(
                    self.graph._provenance_targets_by_source.get(template.uid, ())
                )
                if uid in self.graph.nodes
                and self.graph.nodes[uid].memory_type is MemoryType.GROUNDED_CONTINGENCY
            )
            updated = None
            if surviving:
                limit = max(2, int(self.config.scientific.m1n_facts_per_channel))
                parents = tuple(sorted(surviving)[:limit])
                evidence: list[MemoryUid] = []
                for parent in parents:
                    evidence.extend(
                        uid
                        for uid in tuple(
                            self.graph._provenance_targets_by_source.get(parent, ())
                        )
                        if uid in self.graph.nodes
                        and self.graph.nodes[uid].level is MemoryLevel.M0
                    )
                provenance = DerivationProvenance(
                    parents, tuple(sorted(set(evidence)))
                )
                updated = replace(template, provenance=provenance)
                self._m1n_occurrences[signature] = [updated]
            else:
                self._m1n_occurrences[signature] = []

            if isinstance(family_index, dict):
                family = int(
                    getattr(template, "family_signature", 0)
                    or template.structural_signature
                )
                bucket = [
                    row
                    for row in family_index.get(family, ())
                    if row.uid != template.uid
                ]
                if updated is not None:
                    bucket.append(updated)
                if bucket:
                    family_index[family] = bucket
                else:
                    family_index.pop(family, None)

    def apply_prepared_ingestion_batch(self: Any, rows: Iterable[Any]):
        prepared_rows = tuple(rows)
        manager = getattr(self, "_resident_memory", None)
        if manager is not None:
            snapshot = manager.sample_memory()
            manager._last_snapshot = snapshot
            self.graph._memory_pressure_state = snapshot.state
            if snapshot.state is MemoryGovernorState.HARD_PRESSURE_DRAIN:
                manager.request_compaction(force=True)
                manager.service_prepared_compaction(max_batches=1)
        result = original_apply_batch(self, prepared_rows)
        if manager is not None:
            manager.request_compaction()
            manager.service_prepared_compaction(max_batches=1)
        # Add boundary outcome evidence to the deferred M0 payload so retention
        # scoring can preserve rare successes/failures and truncation boundaries.
        for prepared in prepared_rows:
            transition = getattr(prepared, "transition", None)
            m0 = getattr(prepared, "m0", None)
            if transition is None or m0 is None:
                continue
            deferred = self._deferred_base_nodes.get(m0.uid)
            if deferred is None:
                continue
            node, payload, evidence = deferred
            payload = dict(payload)
            payload["task_success"] = bool(getattr(transition, "task_success", False))
            payload["task_failure"] = bool(getattr(transition, "task_failure", False))
            payload["task_truncated"] = bool(getattr(transition, "task_truncated", False))
            payload["level_index"] = int(getattr(transition, "level_index", 0))
            payload["levels_completed"] = int(getattr(transition, "levels_completed", 0))
            self._deferred_base_nodes[m0.uid] = (node, payload, evidence)
        return result

    def flush_deferred_memory_updates(self: Any) -> None:
        original_flush(self)
        manager = getattr(self, "_resident_memory", None)
        if manager is not None:
            manager.request_compaction(force=manager.backlog() > 0)
            manager.service_prepared_compaction(max_batches=1)

    def dashboard_metrics(self: Any) -> dict[str, Any]:
        result = original_dashboard(self)
        manager = getattr(self, "_resident_memory", None)
        if manager is not None:
            result.update(manager.metrics())
        return result

    def full_metrics(self: Any) -> dict[str, Any]:
        result = original_full_metrics(self)
        manager = getattr(self, "_resident_memory", None)
        if manager is not None:
            result.update(manager.metrics())
        return result

    def close(self: Any, *, normal: bool = True, timeout: float = 300.0):
        manager = getattr(self, "_resident_memory", None)
        if manager is not None:
            manager.stop_background_compaction(timeout=min(5.0, max(0.0, float(timeout))))
        return original_close(self, normal=normal, timeout=timeout)

    runtime_cls.__init__ = runtime_init
    runtime_cls.on_low_level_deleted = on_low_level_deleted
    runtime_cls.apply_prepared_ingestion_batch = apply_prepared_ingestion_batch
    runtime_cls.flush_deferred_memory_updates = flush_deferred_memory_updates
    runtime_cls.dashboard_metrics = dashboard_metrics
    runtime_cls.full_metrics = full_metrics
    runtime_cls.close = close
    runtime_cls._v979_residency_installed = True


def install_bounded_residency(runtime_cls: type) -> None:
    """Install the v9.7.9 bounded-residency contract once at package import."""
    _install_graph_contract()
    _install_runtime_contract(runtime_cls)
