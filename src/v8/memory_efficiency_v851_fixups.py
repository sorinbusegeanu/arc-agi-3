from __future__ import annotations

"""Correctness fixups for the v8.51 memory-efficiency layer."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from v8.model import CognitiveState, MemoryLevel, MemoryUid, RelationType


_INSTALLED = False
_BASE_RUNTIME_INIT = None


def _spec_payload_bytes(root: Path, snapshot: Path, spec: dict[str, object]) -> int:
    if "chunks" in spec:
        return sum(max(0, int(item.get("bytes", 0))) for item in spec.get("chunks", []))
    return int((snapshot / str(spec["file"])).stat().st_size)


def _restore_latest_snapshot_v851_fix(root, descriptors):
    from v8 import arena as arena_module
    from v8 import memory_storage_v851 as storage
    from v8 import snapshot as snapshot_module

    root = Path(root)
    snapshot = snapshot_module.latest_complete_snapshot(root)
    if snapshot is None:
        return None
    manifest_payload = (snapshot / "manifest.json").read_bytes()
    expected_manifest = (snapshot / "COMPLETE").read_text(encoding="ascii").strip()
    if hashlib.sha256(manifest_payload).hexdigest() != expected_manifest:
        raise RuntimeError("snapshot manifest checksum mismatch")
    manifest = json.loads(manifest_payload)
    shard_entries = manifest.get("shards", [])
    descriptors = tuple(descriptors)
    if len(shard_entries) != len(descriptors):
        raise RuntimeError("snapshot shard count does not match runtime")
    opened = []
    try:
        for shard, descriptor in zip(shard_entries, descriptors, strict=True):
            for label, arena_cls, desc in (
                ("nodes", arena_module.SharedNodeArena, descriptor.nodes),
                ("edges", arena_module.SharedEdgeArena, descriptor.edges),
                ("actions", arena_module.SharedActionArena, descriptor.actions),
            ):
                spec = shard[label]
                arena = arena_cls.attach(desc)
                opened.append(arena)
                header = storage._payload_header(root, snapshot, spec)
                source_count, _seq = arena_module._HEADER.unpack(header)
                source_bytes = _spec_payload_bytes(root, snapshot, spec)
                current_expected = arena_module._HEADER.size + int(source_count) * arena.record.size
                if label == "actions" and int(source_count) != int(arena.capacity):
                    storage._restore_action_stream(root, snapshot, spec, arena)
                elif source_bytes == current_expected:
                    storage._copy_current_payload_to_arena(root, snapshot, spec, arena)
                else:
                    # Legacy layouts are only a migration path and may materialize
                    # one historical arena. All current v8.51 snapshots stream.
                    payload = b"".join(storage._payload_chunks(root, snapshot, spec))
                    if hashlib.sha256(payload).hexdigest() != str(spec["sha256"]):
                        raise RuntimeError(f"snapshot checksum mismatch for {label}")
                    if label == "nodes":
                        snapshot_module._load_nodes_compatible(arena, payload)
                    else:
                        arena.load_snapshot(payload)
        return int(manifest["snapshot_id"]), int(manifest["watermark"])
    finally:
        for arena in opened:
            arena.close()


def _compact_retired_memory_v851_fix(descriptors, *, archive_path):
    from v8 import compaction as compaction_module
    from v8 import memory_storage_v851 as storage
    from v8.arena import EdgeRecord, SharedEdgeArena, SharedNodeArena

    descriptors = tuple(descriptors)
    retired: set[MemoryUid] = set()
    low_retired: set[MemoryUid] = set()
    active: set[MemoryUid] = set()
    source_watermark: dict[MemoryUid, int] = {}
    opened = []
    node_arenas = []
    edge_arenas = []
    active_states = {
        int(CognitiveState.ACTIVE),
        int(CognitiveState.VALIDATED),
        int(CognitiveState.REACTIVATED),
    }
    try:
        for descriptor in descriptors:
            nodes = SharedNodeArena.attach(descriptor.nodes)
            edges = SharedEdgeArena.attach(descriptor.edges)
            opened.extend((nodes, edges))
            node_arenas.append(nodes)
            edge_arenas.append(edges)
            for row in nodes.records():
                if int(row.cognitive_state) in active_states:
                    active.add(row.uid)
                    source_watermark[row.uid] = int(row.updated_watermark)
                if (
                    int(row.cognitive_state) == int(CognitiveState.RETIRED)
                    and not storage._is_grounded_m1(row)
                ):
                    retired.add(row.uid)
                    if int(row.level) <= int(MemoryLevel.M1):
                        low_retired.add(row.uid)
        if not retired:
            return compaction_module.CompactionResult(
                0,
                0,
                sum(arena.count for arena in node_arenas),
                sum(arena.count for arena in edge_arenas),
            )

        games_by_target: dict[MemoryUid, dict[MemoryUid, EdgeRecord]] = defaultdict(dict)
        superseders: dict[MemoryUid, set[MemoryUid]] = defaultdict(set)
        for arena in edge_arenas:
            for edge in arena.records():
                relation = int(edge.relation_type)
                if (
                    relation == int(RelationType.GAME_PROVENANCE)
                    and edge.source_uid in low_retired
                    and int(edge.target_uid.hi) == 0
                ):
                    prior = games_by_target[edge.source_uid].get(edge.target_uid)
                    if prior is None or int(edge.support_count) > int(prior.support_count):
                        games_by_target[edge.source_uid][edge.target_uid] = edge
                elif (
                    relation == int(RelationType.SUPERSEDES)
                    and edge.target_uid in low_retired
                    and edge.source_uid in active
                ):
                    superseders[edge.target_uid].add(edge.source_uid)

        candidate_by_shard: dict[int, list[EdgeRecord]] = defaultdict(list)
        candidate_keys: set[tuple[MemoryUid, int, MemoryUid]] = set()
        for target_uid in low_retired:
            for source_uid in superseders.get(target_uid, ()):
                for game_uid, provenance in games_by_target.get(target_uid, {}).items():
                    edge = EdgeRecord(
                        source_uid,
                        int(RelationType.GAME_PROVENANCE),
                        game_uid,
                        max(1, int(provenance.support_count)),
                        max(
                            int(provenance.updated_watermark),
                            int(source_watermark.get(source_uid, 0)),
                        ),
                    )
                    key = (edge.source_uid, int(edge.relation_type), edge.target_uid)
                    if key in candidate_keys:
                        continue
                    candidate_keys.add(key)
                    candidate_by_shard[source_uid.shard(len(descriptors))].append(edge)

        existing: set[tuple[MemoryUid, int, MemoryUid]] = set()
        if candidate_keys:
            for arena in edge_arenas:
                for edge in arena.records():
                    key = (edge.source_uid, int(edge.relation_type), edge.target_uid)
                    if key in candidate_keys:
                        existing.add(key)

        archive = Path(archive_path)
        archive.parent.mkdir(parents=True, exist_ok=True)
        removed_edges = remaining_nodes = remaining_edges = 0
        with archive.open("a", encoding="utf-8") as handle:
            for arena in node_arenas:
                original_count = int(arena.count)
                write_index = 0
                arena.begin_write()
                try:
                    for index in range(original_count):
                        row = arena.read(index)
                        if row.uid in retired:
                            handle.write(
                                json.dumps(
                                    {"kind": "node", "record": compaction_module._node_payload(row)},
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                            continue
                        if write_index != index:
                            arena.write(write_index, row)
                        write_index += 1
                finally:
                    arena.end_write(count=write_index)
                remaining_nodes += write_index

            for shard_index, arena in enumerate(edge_arenas):
                original_count = int(arena.count)
                write_index = 0
                arena.begin_write()
                try:
                    for index in range(original_count):
                        edge = arena.read(index)
                        if edge.source_uid in retired or edge.target_uid in retired:
                            removed_edges += 1
                            handle.write(
                                json.dumps(
                                    {"kind": "edge", "record": compaction_module._edge_payload(edge)},
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                            continue
                        if write_index != index:
                            arena.write(write_index, edge)
                        write_index += 1
                    for edge in candidate_by_shard.get(shard_index, ()):
                        key = (edge.source_uid, int(edge.relation_type), edge.target_uid)
                        if key in existing:
                            continue
                        if write_index >= arena.capacity:
                            raise MemoryError(
                                "provenance-preserving compaction exceeds edge arena capacity"
                            )
                        arena.write(write_index, edge)
                        write_index += 1
                finally:
                    arena.end_write(count=write_index)
                remaining_edges += write_index

        return compaction_module.CompactionResult(
            len(retired), removed_edges, remaining_nodes, remaining_edges
        )
    finally:
        for arena in opened:
            arena.close()


def _runtime_init_v851_fix(self, config) -> None:
    _BASE_RUNTIME_INIT(self, config)
    ledger = getattr(getattr(self, "peers", None), "ledger", None)
    if ledger is not None and hasattr(ledger, "truncate_after"):
        ledger.truncate_after(int(self.watermark))


def install_memory_efficiency_v851_fixups() -> None:
    global _INSTALLED, _BASE_RUNTIME_INIT
    if _INSTALLED:
        return
    from v8 import compaction as compaction_module
    from v8 import runtime as runtime_module
    from v8 import snapshot as snapshot_module
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    snapshot_module.restore_latest_snapshot = _restore_latest_snapshot_v851_fix
    runtime_module.restore_latest_snapshot = _restore_latest_snapshot_v851_fix
    compaction_module.compact_retired_memory = _compact_retired_memory_v851_fix
    runtime_module.compact_retired_arenas = _compact_retired_memory_v851_fix

    _BASE_RUNTIME_INIT = V82ContinuousMemoryRuntime.__init__
    V82ContinuousMemoryRuntime.__init__ = _runtime_init_v851_fix
    _INSTALLED = True
