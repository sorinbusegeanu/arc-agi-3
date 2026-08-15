from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from v8.arena import EdgeRecord, NodeRecord, SharedEdgeArena, SharedNodeArena
from v8.model import CognitiveState, MemoryUid
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


def compact_retired_memory(
    descriptors: tuple[ShardReadDescriptor, ...],
    *,
    archive_path: str | Path,
) -> CompactionResult:
    """Physically reclaim RETIRED node/edge rows at a quiescent generation barrier.

    The removed records and all incident edges are appended to a durable archive before
    RAM rows are rewritten densely. M0/M1 are never retired by lifecycle, so the live
    action index needs no rebuild here. Shard writer processes must be restarted after
    this function so their local UID/edge indexes are reconstructed from compacted RAM.
    """
    opened: list[object] = []
    shard_nodes: list[tuple[SharedNodeArena, list[NodeRecord]]] = []
    shard_edges: list[tuple[SharedEdgeArena, list[EdgeRecord]]] = []
    retired: set[MemoryUid] = set()
    try:
        for descriptor in descriptors:
            nodes = SharedNodeArena.attach(descriptor.nodes)
            edges = SharedEdgeArena.attach(descriptor.edges)
            opened.extend((nodes, edges))
            rows = list(nodes.records())
            edge_rows = list(edges.records())
            shard_nodes.append((nodes, rows))
            shard_edges.append((edges, edge_rows))
            retired.update(
                row.uid
                for row in rows
                if int(row.cognitive_state) == int(CognitiveState.RETIRED)
            )

        if not retired:
            return CompactionResult(
                0,
                0,
                sum(len(rows) for _arena, rows in shard_nodes),
                sum(len(rows) for _arena, rows in shard_edges),
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
