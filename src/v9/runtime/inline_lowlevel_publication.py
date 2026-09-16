from __future__ import annotations

import time
from typing import Any

from v9.memory.identity import MemoryUid
from v9.memory.model import MemoryLevel, MemoryType
from v9.memory.relations import RelationEdge, RelationType
from v9.runtime.bounded_indexes import _ensure_indexes, _remember_edge, _remember_node
from v9.runtime.publication import edge_ref, node_ref


_INLINE_ROW_CHUNK = 2048


def _merge_payload(current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    merged.update(incoming)
    if "parents" in incoming:
        merged["parents"] = sorted(
            {tuple(map(int, row)) for row in current.get("parents", [])}
            | {tuple(map(int, row)) for row in incoming.get("parents", [])}
        )
    return merged


def _row_partitions(graph: Any, rows: tuple[tuple[Any, dict[str, Any], tuple[Any, ...]], ...]) -> set[int]:
    partitions: set[int] = set()
    for node, payload, _evidence in rows:
        partitions.add(node.uid.shard(graph.partition_count))
        for raw_parent in payload.get("parents", []):
            if not isinstance(raw_parent, (list, tuple)) or len(raw_parent) != 2:
                continue
            parent = MemoryUid(int(raw_parent[0]), int(raw_parent[1]))
            partitions.add(parent.shard(graph.partition_count))
    return partitions


def _publication_chunks(rows: tuple[Any, ...]) -> int:
    if not rows:
        return 0
    return (len(rows) + _INLINE_ROW_CHUNK - 1) // _INLINE_ROW_CHUNK


def _logical_cross_partition_transactions(graph: Any, groups: tuple[tuple[Any, ...], ...]) -> int:
    total = 0
    for group in groups:
        for offset in range(0, len(group), _INLINE_ROW_CHUNK):
            chunk = tuple(group[offset : offset + _INLINE_ROW_CHUNK])
            total += int(len(_row_partitions(graph, chunk)) > 1)
    return total


def _publish_chunk(runtime: Any, rows: tuple[tuple[Any, dict[str, Any], tuple[Any, ...]], ...]) -> int:
    if not rows:
        return 0
    graph = runtime.graph
    _ensure_indexes(graph)

    staged_nodes: dict[MemoryUid, tuple[Any, dict[str, Any], tuple[Any, ...]]] = {}
    staged_edges: dict[tuple[MemoryUid, str, MemoryUid], RelationEdge] = {}
    partitions: set[int] = set()

    for node, raw_payload, raw_evidence in rows:
        evidence = tuple(raw_evidence)
        payload = dict(raw_payload)
        payload.setdefault("evidence_refs", [[uid.hi, uid.lo] for uid in sorted(set(evidence))])
        current = staged_nodes.get(node.uid)
        if current is None:
            live_node = graph.nodes.get(node.uid)
            if live_node is not None and (
                live_node.level != node.level
                or live_node.memory_type != node.memory_type
                or live_node.structural_key != node.structural_key
            ):
                raise RuntimeError(f"inline low-level publication identity collision: {node.uid.hex()}")
            live_payload = graph.payloads.get(node.uid, {})
            staged_nodes[node.uid] = (node, _merge_payload(live_payload, payload) if live_node is not None else payload, evidence)
        else:
            staged_nodes[node.uid] = (node, _merge_payload(current[1], payload), evidence)
        partitions.add(node.uid.shard(graph.partition_count))
        for raw_parent in payload.get("parents", []):
            if not isinstance(raw_parent, (list, tuple)) or len(raw_parent) != 2:
                continue
            parent = MemoryUid(int(raw_parent[0]), int(raw_parent[1]))
            edge = RelationEdge(node.uid, RelationType.PROVENANCE, parent, evidence)
            staged_edges[edge.key] = edge
            partitions.add(parent.shard(graph.partition_count))

    with graph._coordinator.locked(tuple(sorted(partitions))), graph._publication_lock:
        node_deltas = [0] * graph.partition_count
        edge_deltas = [0] * graph.partition_count
        for uid, (node, _payload, _evidence) in staged_nodes.items():
            live = graph.nodes.get(uid)
            if live is not None and (
                live.level != node.level
                or live.memory_type != node.memory_type
                or live.structural_key != node.structural_key
            ):
                raise RuntimeError(f"inline low-level publication identity collision: {uid.hex()}")
            if live is None:
                node_deltas[uid.shard(graph.partition_count)] += 1
        for key, edge in staged_edges.items():
            if key not in graph.edges:
                edge_deltas[edge.source.shard(graph.partition_count)] += 1

        for partition, delta in enumerate(node_deltas):
            capacity = graph.node_capacity_per_partition
            if capacity is not None and graph._node_counts_by_partition[partition] + delta > capacity:
                raise RuntimeError("inline low-level publication exceeded node capacity")
        for partition, delta in enumerate(edge_deltas):
            capacity = graph.edge_capacity_per_partition
            if capacity is not None and graph._edge_counts_by_partition[partition] + delta > capacity:
                raise RuntimeError("inline low-level publication exceeded edge capacity")

        inserted_low_level = 0
        new_nodes: list[Any] = []
        for uid, (node, payload, _evidence) in staged_nodes.items():
            is_new = uid not in graph.nodes
            if is_new:
                owner = uid.shard(graph.partition_count)
                graph._node_counts_by_partition[owner] += 1
                graph._node_uids_by_partition[owner].add(uid)
                graph._uids_by_level[node.level].add(uid)
                graph.retired_tombstones.pop(uid, None)
                if node.level is MemoryLevel.M0 and payload.get("action_id") is not None:
                    graph._training_m0_reservoir.append(uid)
                if (
                    node.level is MemoryLevel.M0 and node.memory_type is MemoryType.EPISODE
                ) or (
                    node.level is MemoryLevel.M1 and node.memory_type is MemoryType.GROUNDED_CONTINGENCY
                ):
                    graph.low_level_nodes_inserted_total = int(getattr(graph, "low_level_nodes_inserted_total", 0)) + 1
                    if node.level is MemoryLevel.M0:
                        graph._resident_m0_count = int(getattr(graph, "_resident_m0_count", 0)) + 1
                    else:
                        graph._resident_m1_grounded_count = int(getattr(graph, "_resident_m1_grounded_count", 0)) + 1
                    inserted_low_level += 1
                new_nodes.append(node)
            graph.nodes[uid] = node
            graph.payloads[uid] = payload
            graph.versions.bump(node_ref(uid))

        new_edges: list[RelationEdge] = []
        for key, edge in staged_edges.items():
            previous = graph.edges.get(key)
            if previous is None:
                owner = edge.source.shard(graph.partition_count)
                graph._edge_counts_by_partition[owner] += 1
                graph._edge_keys_by_partition[owner].add(key)
                graph._index_edge(edge)
                new_edges.append(edge)
            graph.edges[key] = edge
            graph.versions.bump(edge_ref(edge))

        for node in new_nodes:
            _remember_node(graph, node)
        for edge in new_edges:
            _remember_edge(graph, edge.source, edge.key)
            _remember_edge(graph, edge.target, edge.key)

        graph.generation += 1
        graph._cached_read_view = None

    runtime.telemetry["proposals"] += 1
    runtime.telemetry["accepted"] += 1
    runtime.telemetry["cross_partition_transactions"] += int(len(partitions) > 1)
    runtime.__dict__.pop("_actor_policy_snapshot_cache", None)

    environment_index = getattr(runtime, "_memory_uids_by_environment", None)
    grounding_index = runtime.__dict__.setdefault("_grounding_action_payload_by_low", {})
    for uid, (node, payload, _evidence) in staged_nodes.items():
        live_payload = graph.payloads[uid]
        environment_id = live_payload.get("environment_instance_id")
        if environment_index is not None and environment_id is not None:
            environment_index.setdefault(int(environment_id), set()).add(uid)
        if node.level is MemoryLevel.M1 and live_payload.get("action_id", live_payload.get("executable_action_token")) is not None:
            grounding_index[int(uid.lo)] = (uid, live_payload)
        if runtime.config.enable_lifecycle:
            runtime.lifecycle.observe(
                uid,
                support_delta=1,
                relevant_opportunity=True,
                watermark=int(node.created_watermark),
            )
    runtime.set_telemetry_gauge("grounding_action_index_size", len(grounding_index))
    return inserted_low_level


def install_inline_lowlevel_publication(runtime_cls: type) -> None:
    if getattr(runtime_cls, "_inline_lowlevel_publication_installed", False):
        return

    def publication_generation_delta(self: Any, rows: tuple[Any, ...]) -> int:
        return _publication_chunks(tuple(rows))

    def publish_inline_groups(self: Any, groups: tuple[tuple[Any, ...], ...]) -> None:
        logical_groups = tuple(tuple(group) for group in groups if group)
        if not logical_groups:
            return
        rows = tuple(row for group in logical_groups for row in group)
        started = time.perf_counter()
        inserted = 0
        physical_batches = 0
        logical_batches = sum(_publication_chunks(group) for group in logical_groups)
        logical_cross = _logical_cross_partition_transactions(self.graph, logical_groups)
        with self._lock:
            cross_before = int(self.telemetry.get("cross_partition_transactions", 0))
            for offset in range(0, len(rows), _INLINE_ROW_CHUNK):
                chunk = tuple(rows[offset : offset + _INLINE_ROW_CHUNK])
                inserted += _publish_chunk(self, chunk)
                physical_batches += 1
            physical_cross = int(self.telemetry.get("cross_partition_transactions", 0)) - cross_before
            logical_extra = logical_batches - physical_batches
            if logical_extra:
                self.graph.generation += logical_extra
                self.graph._cached_read_view = None
                self.telemetry["proposals"] += logical_extra
                self.telemetry["accepted"] += logical_extra
            cross_adjustment = logical_cross - physical_cross
            if cross_adjustment:
                self.telemetry["cross_partition_transactions"] += cross_adjustment
        elapsed = time.perf_counter() - started
        self._inline_lowlevel_rows = int(getattr(self, "_inline_lowlevel_rows", 0)) + len(rows)
        self._inline_lowlevel_batches = int(getattr(self, "_inline_lowlevel_batches", 0)) + logical_batches
        self._inline_lowlevel_seconds = float(getattr(self, "_inline_lowlevel_seconds", 0.0)) + elapsed
        self.set_telemetry_gauge("inline_lowlevel_publication_rows", self._inline_lowlevel_rows)
        self.set_telemetry_gauge("inline_lowlevel_publication_batches", self._inline_lowlevel_batches)
        self.set_telemetry_gauge("inline_lowlevel_publication_inserted", int(getattr(self, "_inline_lowlevel_inserted", 0)) + inserted)
        self._inline_lowlevel_inserted = int(getattr(self, "_inline_lowlevel_inserted", 0)) + inserted
        self.set_telemetry_gauge("inline_lowlevel_publication_ms", 1000.0 * self._inline_lowlevel_seconds)
        self.set_telemetry_gauge("inline_lowlevel_publication_ms_per_row", 1000.0 * self._inline_lowlevel_seconds / max(1, self._inline_lowlevel_rows))
        self.set_telemetry_gauge("deferred_base_nodes", len(self._deferred_base_nodes))
        self.set_telemetry_gauge("dirty_m1n_supports", len(self._m1n_dirty))

    def publish_inline(self: Any, rows: tuple[tuple[Any, dict[str, Any], tuple[Any, ...]], ...]) -> None:
        publish_inline_groups(self, (tuple(rows),))

    runtime_cls._deferred_publication_generation_delta = publication_generation_delta
    runtime_cls._defer_base_groups = publish_inline_groups
    runtime_cls._defer_base_group = publish_inline
    runtime_cls._inline_lowlevel_publication_installed = True
