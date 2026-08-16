from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b

from v8.arena import EdgeRecord, NodeRecord


@dataclass(frozen=True, slots=True)
class DevelopmentalGenerationCut:
    generation: int
    watermark: int
    shard_vector: tuple[tuple[int, int], ...]
    nodes: tuple[NodeRecord, ...]
    edges: tuple[EdgeRecord, ...]
    graph_digest: str


def capture_developmental_cut(read_view, *, generation: int, watermark: int) -> DevelopmentalGenerationCut:
    """Pin immutable per-shard records and expose one stable vector scientific cut.

    Actors continue using the normal bounded-staleness LiveReadView.  This function
    is for peer/scientific operations that need a declared multi-shard source cut.
    """
    nodes: list[NodeRecord] = []
    edges: list[EdgeRecord] = []
    vector: list[tuple[int, int]] = []
    for index, (node_arena, edge_arena) in enumerate(zip(read_view._nodes, read_view._edges, strict=True)):
        node_rows, node_version = read_view._stable_records_with_version(node_arena)
        edge_rows, edge_version = read_view._stable_records_with_version(edge_arena)
        nodes.extend(node_rows)
        edges.extend(edge_rows)
        vector.append((int(node_version), int(edge_version)))

    node_uids = {row.uid for row in nodes}
    # Pseudo GAME_PROVENANCE targets deliberately do not correspond to live nodes.
    filtered_edges = tuple(
        edge
        for edge in edges
        if edge.source_uid in node_uids
        and (edge.target_uid in node_uids or int(edge.target_uid.hi) == 0)
    )
    ordered_nodes = tuple(sorted(nodes, key=lambda row: row.uid))
    ordered_edges = tuple(
        sorted(
            filtered_edges,
            key=lambda edge: (edge.source_uid, int(edge.relation_type), edge.target_uid),
        )
    )
    digest = blake2b(digest_size=16, person=b"v8.2-dev-cut")
    for row in ordered_nodes:
        digest.update(row.uid.hi.to_bytes(8, "little"))
        digest.update(row.uid.lo.to_bytes(8, "little"))
        digest.update(int(row.updated_watermark).to_bytes(8, "little"))
        digest.update(int(row.cognitive_state).to_bytes(2, "little", signed=True))
        digest.update(int(row.validation_state).to_bytes(2, "little", signed=True))
    for edge in ordered_edges:
        digest.update(edge.source_uid.hi.to_bytes(8, "little"))
        digest.update(edge.source_uid.lo.to_bytes(8, "little"))
        digest.update(int(edge.relation_type).to_bytes(2, "little"))
        digest.update(edge.target_uid.hi.to_bytes(8, "little"))
        digest.update(edge.target_uid.lo.to_bytes(8, "little"))
        digest.update(int(edge.updated_watermark).to_bytes(8, "little"))
    return DevelopmentalGenerationCut(
        int(generation),
        int(watermark),
        tuple(vector),
        ordered_nodes,
        ordered_edges,
        digest.hexdigest(),
    )
