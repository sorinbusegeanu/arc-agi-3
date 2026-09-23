from __future__ import annotations

from collections import OrderedDict, deque
import heapq
from itertools import islice
from typing import Any, Iterable

from v9.memory.identity import MemoryUid
from v9.memory.model import MemoryLevel, MemoryType
from v9.mutation.transactions import MutationOutcome

from .publication import CanonicalGraph
from .read_view import ReadView, _cognitively_visible
from .residency import ResidentMemoryManager


_DEFAULT_RECENT_CAPACITY = 65_536
_DEFAULT_EDGE_CAPACITY = 256


def _is_compaction_level(node: Any, level: MemoryLevel) -> bool:
    if level is MemoryLevel.M0:
        return node.level is MemoryLevel.M0 and node.memory_type is MemoryType.EPISODE
    return node.level is MemoryLevel.M1 and node.memory_type is MemoryType.GROUNDED_CONTINGENCY


def _ensure_indexes(graph: CanonicalGraph) -> None:
    if hasattr(graph, "_bounded_recent_by_level"):
        return
    graph._bounded_recent_capacity = _DEFAULT_RECENT_CAPACITY
    graph._bounded_edge_capacity = _DEFAULT_EDGE_CAPACITY
    graph._bounded_recent_by_level = {
        level: deque(maxlen=_DEFAULT_RECENT_CAPACITY)
        for level in MemoryLevel
    }
    graph._bounded_edges_by_uid: dict[MemoryUid, OrderedDict[tuple[MemoryUid, str, MemoryUid], None]] = {}
    graph._bounded_compaction_queue = {
        MemoryLevel.M0: deque(),
        MemoryLevel.M1: deque(),
    }
    graph._bounded_view_nodes = 0
    graph._bounded_view_edges = 0
    graph._bounded_view_node_scan = 0
    graph._bounded_view_edge_scan = 0
    graph._bounded_index_rebuilds = 0


def _remember_edge(graph: CanonicalGraph, uid: MemoryUid, key: tuple[MemoryUid, str, MemoryUid]) -> None:
    rows = graph._bounded_edges_by_uid.setdefault(uid, OrderedDict())
    rows.pop(key, None)
    rows[key] = None
    while len(rows) > int(graph._bounded_edge_capacity):
        rows.popitem(last=False)


def _remember_node(graph: CanonicalGraph, node: Any) -> None:
    recent = graph._bounded_recent_by_level[node.level]
    recent.append(node.uid)
    if node.level is MemoryLevel.M0 and node.memory_type is MemoryType.EPISODE:
        graph._bounded_compaction_queue[MemoryLevel.M0].append(node.uid)
    elif node.level is MemoryLevel.M1 and node.memory_type is MemoryType.GROUNDED_CONTINGENCY:
        graph._bounded_compaction_queue[MemoryLevel.M1].append(node.uid)


def _rebuild_bounded_indexes(
    graph: CanonicalGraph,
    *,
    recent_capacity: int = _DEFAULT_RECENT_CAPACITY,
    edge_capacity: int = _DEFAULT_EDGE_CAPACITY,
) -> None:
    """One-time restore/startup rebuild; steady-state consumers never full-scan."""
    _ensure_indexes(graph)
    recent_capacity = max(256, int(recent_capacity))
    edge_capacity = max(16, int(edge_capacity))
    graph._bounded_recent_capacity = recent_capacity
    graph._bounded_edge_capacity = edge_capacity
    graph._bounded_recent_by_level = {}
    for level in MemoryLevel:
        newest = heapq.nlargest(
            recent_capacity,
            (
                (int(graph.nodes[uid].created_watermark), uid)
                for uid in graph._uids_by_level[level]
                if uid in graph.nodes
            ),
        )
        newest.sort()
        graph._bounded_recent_by_level[level] = deque(
            (uid for _watermark, uid in newest),
            maxlen=recent_capacity,
        )

    # Concrete resident levels are already capacity-controlled. Keep one
    # rotating UID reference per resident item so every compaction pass has a
    # bounded scan cost independent of historical graph size.
    graph._bounded_compaction_queue = {
        MemoryLevel.M0: deque(
            uid
            for uid in graph._uids_by_level[MemoryLevel.M0]
            if uid in graph.nodes and _is_compaction_level(graph.nodes[uid], MemoryLevel.M0)
        ),
        MemoryLevel.M1: deque(
            uid
            for uid in graph._uids_by_level[MemoryLevel.M1]
            if uid in graph.nodes and _is_compaction_level(graph.nodes[uid], MemoryLevel.M1)
        ),
    }

    graph._bounded_edges_by_uid = {}
    for key, edge in graph.edges.items():
        _remember_edge(graph, edge.source, key)
        _remember_edge(graph, edge.target, key)
    graph._bounded_index_rebuilds += 1


def _recent_rows(
    graph: CanonicalGraph,
    level: MemoryLevel,
    *,
    scan_limit: int,
) -> Iterable[tuple[MemoryUid, Any]]:
    rows = graph._bounded_recent_by_level[level]
    scanned = 0
    for uid in reversed(rows):
        if scanned >= scan_limit:
            break
        scanned += 1
        node = graph.nodes.get(uid)
        if node is None or node.level is not level:
            continue
        if not _cognitively_visible(graph.payloads.get(uid, {})):
            continue
        yield uid, node


def _indexed_view(
    graph: CanonicalGraph,
    *,
    max_nodes: int,
    max_edges: int,
    seed_uids: Iterable[MemoryUid] = (),
    per_level_quotas: dict[MemoryLevel, int] | None = None,
    training: bool = False,
) -> ReadView:
    _ensure_indexes(graph)
    maximum_nodes = max(1, int(max_nodes))
    maximum_edges = max(1, int(max_edges))
    node_scan_budget = max(maximum_nodes * 8, maximum_nodes + 64)
    edge_scan_budget = max(maximum_edges * 4, maximum_nodes * 8, 256)

    with graph._publication_lock:
        selected: dict[MemoryUid, Any] = {}
        historical_uids: set[MemoryUid] = set()
        node_scanned = 0

        for uid in dict.fromkeys(seed_uids):
            if len(selected) >= maximum_nodes or node_scanned >= node_scan_budget:
                break
            node_scanned += 1
            node = graph.nodes.get(uid)
            if node is not None and _cognitively_visible(graph.payloads.get(uid, {})):
                selected[uid] = node

        if training and len(selected) < maximum_nodes:
            behavior_scan = min(
                node_scan_budget - node_scanned,
                max(maximum_nodes * 4, maximum_nodes),
            )
            candidates: list[tuple[int, int, MemoryUid, Any]] = []
            for uid in islice(reversed(graph._training_m0_reservoir), max(0, behavior_scan)):
                node_scanned += 1
                node = graph.nodes.get(uid)
                if node is None:
                    continue
                payload = graph.payloads.get(uid, {})
                if payload.get("action_id") is None or payload.get("environment_instance_id") is None or payload.get("episode_id") is None:
                    continue
                candidates.append((abs(int(payload.get("primary_valence", 0))), int(node.created_watermark), uid, node))
            behavior_budget = max(1, maximum_nodes // 2)
            for _valence, _watermark, uid, node in heapq.nlargest(behavior_budget, candidates):
                if len(selected) >= maximum_nodes:
                    break
                selected.setdefault(uid, node)

        quotas = dict(per_level_quotas or {})
        if not quotas:
            remaining = max(0, maximum_nodes - len(selected))
            per_level = max(1, remaining // max(1, len(tuple(MemoryLevel))))
            quotas = {level: per_level for level in MemoryLevel}

        for level in MemoryLevel:
            if len(selected) >= maximum_nodes or node_scanned >= node_scan_budget:
                break
            quota = max(0, int(quotas.get(level, 0)))
            if quota <= 0:
                continue
            level_scan = min(
                node_scan_budget - node_scanned,
                max(quota * 4, quota),
            )
            added = 0
            for uid, node in _recent_rows(graph, level, scan_limit=level_scan):
                node_scanned += 1
                if uid in selected:
                    continue
                selected[uid] = node
                added += 1
                if added >= quota or len(selected) >= maximum_nodes:
                    break

        # Expand only through bounded per-node adjacency. No global edge or node
        # collection is materialized here.
        frontier = deque(selected)
        edge_scanned = 0
        while frontier and len(selected) < maximum_nodes and edge_scanned < edge_scan_budget:
            uid = frontier.popleft()
            adjacency = graph._bounded_edges_by_uid.get(uid)
            if not adjacency:
                continue
            for key in reversed(adjacency):
                if edge_scanned >= edge_scan_budget or len(selected) >= maximum_nodes:
                    break
                edge_scanned += 1
                edge = graph.edges.get(key)
                if edge is None:
                    continue
                other = edge.target if edge.source == uid else edge.source
                if other in selected:
                    continue
                node = graph.nodes.get(other)
                if node is None or not _cognitively_visible(graph.payloads.get(other, {})):
                    continue
                selected[other] = node
                frontier.append(other)

        # Fill any remaining node budget round-robin from the bounded recent
        # indexes; never fall back to graph.nodes or full level-set iteration.
        if len(selected) < maximum_nodes and node_scanned < node_scan_budget:
            level_iters = {
                level: iter(reversed(graph._bounded_recent_by_level[level]))
                for level in MemoryLevel
            }
            active = list(MemoryLevel)
            while active and len(selected) < maximum_nodes and node_scanned < node_scan_budget:
                next_active: list[MemoryLevel] = []
                for level in active:
                    iterator = level_iters[level]
                    try:
                        uid = next(iterator)
                    except StopIteration:
                        continue
                    next_active.append(level)
                    node_scanned += 1
                    if uid in selected:
                        continue
                    node = graph.nodes.get(uid)
                    if node is None or not _cognitively_visible(graph.payloads.get(uid, {})):
                        continue
                    selected[uid] = node
                    if len(selected) >= maximum_nodes or node_scanned >= node_scan_budget:
                        break
                active = next_active

        selected_uids = set(selected)
        edge_candidates: set[tuple[MemoryUid, str, MemoryUid]] = set()
        for uid in selected_uids:
            if edge_scanned >= edge_scan_budget:
                break
            adjacency = graph._bounded_edges_by_uid.get(uid)
            if not adjacency:
                continue
            for key in reversed(adjacency):
                if edge_scanned >= edge_scan_budget:
                    break
                edge_scanned += 1
                edge = graph.edges.get(key)
                if edge is None or edge.source not in selected_uids or edge.target not in selected_uids:
                    continue
                edge_candidates.add(key)
                if len(edge_candidates) >= maximum_edges:
                    break
            if len(edge_candidates) >= maximum_edges:
                break

        subset_edges = {
            key: graph.edges[key]
            for key in edge_candidates
            if key in graph.edges
        }
        subset_payloads = {uid: graph.payloads.get(uid, {}) for uid in selected_uids}
        graph._bounded_view_nodes = len(selected)
        graph._bounded_view_edges = len(subset_edges)
        graph._bounded_view_node_scan = node_scanned
        graph._bounded_view_edge_scan = edge_scanned
        return ReadView.build(
            graph.generation,
            dict(selected),
            subset_payloads,
            subset_edges,
            {},
            include_hidden_uids=historical_uids,
        )


def install_bounded_indexes(runtime_cls: type) -> None:
    """Install steady-state bounded graph indexes and consumers once."""
    if getattr(CanonicalGraph, "_v979_bounded_indexes_installed", False):
        return

    original_graph_init = CanonicalGraph.__init__
    original_publish_locked = CanonicalGraph._publish_locked

    def graph_init(self: CanonicalGraph, *args: Any, **kwargs: Any) -> None:
        original_graph_init(self, *args, **kwargs)
        _ensure_indexes(self)

    def publish_locked(self: CanonicalGraph, proposal: Any):
        _ensure_indexes(self)
        new_nodes = [
            write.node
            for write in proposal.writes
            if getattr(write, "node", None) is not None and write.node.uid not in self.nodes
        ]
        edge_writes = [
            write.edge
            for write in proposal.writes
            if getattr(write, "edge", None) is not None
        ]
        result = original_publish_locked(self, proposal)
        if result.outcome is MutationOutcome.ACCEPTED:
            for node in new_nodes:
                live = self.nodes.get(node.uid)
                if live is not None:
                    _remember_node(self, live)
            for edge in edge_writes:
                live = self.edges.get(edge.key)
                if live is not None:
                    _remember_edge(self, live.source, live.key)
                    _remember_edge(self, live.target, live.key)
        return result

    def rebuild_bounded_indexes(
        self: CanonicalGraph,
        *,
        recent_capacity: int = _DEFAULT_RECENT_CAPACITY,
        edge_capacity: int = _DEFAULT_EDGE_CAPACITY,
    ) -> None:
        _rebuild_bounded_indexes(
            self,
            recent_capacity=recent_capacity,
            edge_capacity=edge_capacity,
        )

    def training_view(self: CanonicalGraph, *, max_nodes: int = 800, max_edges: int = 4000) -> ReadView:
        return _indexed_view(
            self,
            max_nodes=max(8, int(max_nodes)),
            max_edges=max_edges,
            training=True,
        )

    def bounded_view(
        self: CanonicalGraph,
        *,
        max_nodes: int = 800,
        max_edges: int = 4000,
        seed_uids: Iterable[MemoryUid] = (),
        per_level_quotas: dict[MemoryLevel, int] | None = None,
        selection_reason: str = "bounded_runtime_view",
    ) -> ReadView:
        del selection_reason
        return _indexed_view(
            self,
            max_nodes=max_nodes,
            max_edges=max_edges,
            seed_uids=seed_uids,
            per_level_quotas=per_level_quotas,
            training=False,
        )

    original_manager_init = ResidentMemoryManager.__init__
    original_manager_metrics = ResidentMemoryManager.metrics
    original_publish_telemetry = ResidentMemoryManager._publish_telemetry

    def manager_init(self: ResidentMemoryManager, runtime: Any) -> None:
        original_manager_init(self, runtime)
        scientific = runtime.config.scientific
        recent_capacity = max(
            8192,
            min(
                int(scientific.resident_max_scan_batch),
                max(int(scientific.hgt_max_subgraph_nodes) * 4, int(scientific.hgt_max_total_nodes)),
            ),
        )
        edge_capacity = max(
            64,
            min(1024, int(scientific.hgt_max_subgraph_edges) // max(1, int(scientific.hgt_max_subgraph_nodes) // 4)),
        )
        runtime.graph.rebuild_bounded_indexes(
            recent_capacity=recent_capacity,
            edge_capacity=edge_capacity,
        )

    def candidate_rows(self: ResidentMemoryManager, level: MemoryLevel, limit: int):
        graph = self.runtime.graph
        _ensure_indexes(graph)
        queue = graph._bounded_compaction_queue[level]
        scan_limit = min(max(1, int(limit)), len(queue))
        rows = []
        for _ in range(scan_limit):
            uid = queue.popleft()
            node = graph.nodes.get(uid)
            if node is None or not _is_compaction_level(node, level):
                continue
            queue.append(uid)
            replacement_uid = self._replacement_for(uid)
            if replacement_uid is None:
                continue
            rows.append((uid, node, graph.payloads.get(uid, {}), replacement_uid))
        return rows

    def publish_telemetry(self: ResidentMemoryManager) -> None:
        original_publish_telemetry(self)
        graph = self.runtime.graph
        gauges = getattr(self.runtime.unified_telemetry, "gauges", None)
        if isinstance(gauges, dict):
            gauges.update({
                "bounded_view_nodes": int(getattr(graph, "_bounded_view_nodes", 0)),
                "bounded_view_edges": int(getattr(graph, "_bounded_view_edges", 0)),
                "bounded_view_node_scan": int(getattr(graph, "_bounded_view_node_scan", 0)),
                "bounded_view_edge_scan": int(getattr(graph, "_bounded_view_edge_scan", 0)),
                "bounded_index_rebuilds": int(getattr(graph, "_bounded_index_rebuilds", 0)),
            })

    def manager_metrics(self: ResidentMemoryManager):
        result = original_manager_metrics(self)
        graph = self.runtime.graph
        result.update({
            "bounded_view_nodes": int(getattr(graph, "_bounded_view_nodes", 0)),
            "bounded_view_edges": int(getattr(graph, "_bounded_view_edges", 0)),
            "bounded_view_node_scan": int(getattr(graph, "_bounded_view_node_scan", 0)),
            "bounded_view_edge_scan": int(getattr(graph, "_bounded_view_edge_scan", 0)),
            "bounded_index_rebuilds": int(getattr(graph, "_bounded_index_rebuilds", 0)),
        })
        return result

    CanonicalGraph.__init__ = graph_init
    CanonicalGraph._publish_locked = publish_locked
    CanonicalGraph.rebuild_bounded_indexes = rebuild_bounded_indexes
    CanonicalGraph.training_view = training_view
    CanonicalGraph.bounded_view = bounded_view
    ResidentMemoryManager.__init__ = manager_init
    ResidentMemoryManager._candidate_rows = candidate_rows
    ResidentMemoryManager._publish_telemetry = publish_telemetry
    ResidentMemoryManager.metrics = manager_metrics
    CanonicalGraph._v979_bounded_indexes_installed = True
