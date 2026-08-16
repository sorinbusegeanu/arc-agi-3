from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from v8.arena import EdgeRecord, NodeRecord, SharedEdgeArena, SharedNodeArena
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, RelationType
from v8.publication import ShardReadDescriptor


@dataclass(frozen=True, slots=True)
class CompactionResult:
    retired_nodes: int
    removed_edges: int
    remaining_nodes: int
    remaining_edges: int


def _uid(uid: MemoryUid) -> str:
    return uid.hex()


def _node_payload(row: NodeRecord) -> dict[str, object]:
    raw = asdict(row)
    raw["uid"] = _uid(row.uid)
    return raw


def _edge_payload(row: EdgeRecord) -> dict[str, object]:
    return {
        "source_uid": _uid(row.source_uid),
        "relation_type": int(row.relation_type),
        "target_uid": _uid(row.target_uid),
        "support_count": int(row.support_count),
        "updated_watermark": int(row.updated_watermark),
    }


def _is_grounded_m1(row: NodeRecord) -> bool:
    return bool(
        int(row.level) == int(MemoryLevel.M1)
        and int(row.memory_type) == int(MemoryType.CONTINGENCY)
        and len(row.key_parts) >= 4
    )


def _promote_low_level_provenance(
    *,
    descriptors: tuple[ShardReadDescriptor, ...],
    by_uid: dict[MemoryUid, NodeRecord],
    retired: set[MemoryUid],
    shard_edges: list[tuple[SharedEdgeArena, list[EdgeRecord]]],
) -> None:
    """Copy direct game provenance from retired M0/M1 to active superseders."""
    low_retired = {
        uid
        for uid in retired
        if uid in by_uid and int(by_uid[uid].level) <= int(MemoryLevel.M1)
    }
    if not low_retired:
        return

    active_states = {
        int(CognitiveState.ACTIVE),
        int(CognitiveState.VALIDATED),
        int(CognitiveState.REACTIVATED),
    }
    all_edges = tuple(edge for _arena, rows in shard_edges for edge in rows)
    games_by_target: dict[MemoryUid, dict[MemoryUid, EdgeRecord]] = {}
    superseders_by_target: dict[MemoryUid, set[MemoryUid]] = {}
    existing = {
        (edge.source_uid, int(edge.relation_type), edge.target_uid)
        for edge in all_edges
    }

    for edge in all_edges:
        relation = int(edge.relation_type)
        if (
            relation == int(RelationType.GAME_PROVENANCE)
            and edge.source_uid in low_retired
            and int(edge.target_uid.hi) == 0
        ):
            prior = games_by_target.setdefault(edge.source_uid, {}).get(edge.target_uid)
            if prior is None or int(edge.support_count) > int(prior.support_count):
                games_by_target[edge.source_uid][edge.target_uid] = edge
        elif relation == int(RelationType.SUPERSEDES) and edge.target_uid in low_retired:
            source = by_uid.get(edge.source_uid)
            if (
                source is not None
                and edge.source_uid not in retired
                and int(source.cognitive_state) in active_states
            ):
                superseders_by_target.setdefault(edge.target_uid, set()).add(edge.source_uid)

    shard_count = len(descriptors)
    for target_uid in sorted(low_retired):
        game_edges = games_by_target.get(target_uid, {})
        if not game_edges:
            continue
        for source_uid in sorted(superseders_by_target.get(target_uid, ())):
            source = by_uid[source_uid]
            shard_index = source_uid.shard(shard_count)
            _arena, rows = shard_edges[shard_index]
            for game_uid, provenance in sorted(game_edges.items()):
                key = (source_uid, int(RelationType.GAME_PROVENANCE), game_uid)
                if key in existing:
                    continue
                rows.append(
                    EdgeRecord(
                        source_uid,
                        int(RelationType.GAME_PROVENANCE),
                        game_uid,
                        max(1, int(provenance.support_count)),
                        max(int(provenance.updated_watermark), int(source.updated_watermark)),
                    )
                )
                existing.add(key)


def compact_retired_memory(
    descriptors: tuple[ShardReadDescriptor, ...],
    *,
    archive_path: str | Path,
) -> CompactionResult:
    """Physically reclaim safely RETIRED rows at a quiescent generation barrier.

    Removed records and incident edges are appended to a durable archive before RAM
    rows are rewritten densely.  M0 and normalized M1 may now retire; direct game
    provenance is first copied to their active semantic superseders.  Grounded M1 is
    a live ActionArena/M7 causal anchor and is fail-closed here even if an invalid
    caller marks it RETIRED, so the action index never needs an unsafe partial rebuild.
    Shard writers must be restarted afterwards so UID/edge indexes match compacted RAM.
    """
    opened: list[object] = []
    shard_nodes: list[tuple[SharedNodeArena, list[NodeRecord]]] = []
    shard_edges: list[tuple[SharedEdgeArena, list[EdgeRecord]]] = []
    retired: set[MemoryUid] = set()
    by_uid: dict[MemoryUid, NodeRecord] = {}
    try:
        for descriptor in descriptors:
            nodes = SharedNodeArena.attach(descriptor.nodes)
            edges = SharedEdgeArena.attach(descriptor.edges)
            opened.extend((nodes, edges))
            rows = list(nodes.records())
            edge_rows = list(edges.records())
            shard_nodes.append((nodes, rows))
            shard_edges.append((edges, edge_rows))
            for row in rows:
                by_uid[row.uid] = row
                if (
                    int(row.cognitive_state) == int(CognitiveState.RETIRED)
                    and not _is_grounded_m1(row)
                ):
                    retired.add(row.uid)

        if not retired:
            return CompactionResult(
                0,
                0,
                sum(len(rows) for _arena, rows in shard_nodes),
                sum(len(rows) for _arena, rows in shard_edges),
            )

        _promote_low_level_provenance(
            descriptors=descriptors,
            by_uid=by_uid,
            retired=retired,
            shard_edges=shard_edges,
        )

        archive = Path(archive_path)
        archive.parent.mkdir(parents=True, exist_ok=True)
        archived_lines: list[str] = []
        for _arena, rows in shard_nodes:
            for row in rows:
                if row.uid in retired:
                    archived_lines.append(
                        json.dumps({"kind": "node", "record": _node_payload(row)}, sort_keys=True)
                    )
        removed_edges = 0
        for _arena, rows in shard_edges:
            for edge in rows:
                if edge.source_uid in retired or edge.target_uid in retired:
                    removed_edges += 1
                    archived_lines.append(
                        json.dumps({"kind": "edge", "record": _edge_payload(edge)}, sort_keys=True)
                    )
        with archive.open("a", encoding="utf-8") as handle:
            for line in archived_lines:
                handle.write(line + "\n")

        remaining_nodes = 0
        for arena, rows in shard_nodes:
            keep = [row for row in rows if row.uid not in retired]
            arena.begin_write()
            try:
                for index, row in enumerate(keep):
                    arena.write(index, row)
            finally:
                arena.end_write(count=len(keep))
            remaining_nodes += len(keep)

        remaining_edges = 0
        for arena, rows in shard_edges:
            keep = [
                edge
                for edge in rows
                if edge.source_uid not in retired and edge.target_uid not in retired
            ]
            if len(keep) > arena.capacity:
                raise MemoryError("provenance-preserving compaction exceeds edge arena capacity")
            arena.begin_write()
            try:
                for index, edge in enumerate(keep):
                    arena.write(index, edge)
            finally:
                arena.end_write(count=len(keep))
            remaining_edges += len(keep)

        return CompactionResult(
            len(retired),
            removed_edges,
            remaining_nodes,
            remaining_edges,
        )
    finally:
        for arena in opened:
            arena.close()
