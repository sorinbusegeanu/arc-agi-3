from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from v3_1.utils.ids import make_handle


NODE_KINDS = {
    "poi",
    "trigger",
    "panel",
    "gate",
    "exit",
    "symbol_state",
    "region",
    "effect_region",
}

EDGE_KINDS = {
    "changes",
    "displays",
    "matches",
    "controls_access",
    "opens",
    "requires",
    "causes_remote_change",
    "enables_exit",
    "contradicts",
}


@dataclass(frozen=True)
class MechanicNode:
    node_id: str
    node_kind: str
    evidence_tier: str
    confidence: float
    source_episode_ids: tuple[str, ...] = ()
    source_round_ids: tuple[int, ...] = ()
    support_count: int = 0
    contradiction_count: int = 0
    first_seen_round: int = 0
    last_seen_round: int = 0
    semantic_key: str | None = None
    object_ref: str | None = None
    pattern_id: str | None = None
    source_entity_id: str | None = None
    identity_confidence: float = 0.0
    identity_status: str = "unknown"
    identity_history: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MechanicEdge:
    edge_id: str
    src_node_id: str
    edge_kind: str
    dst_node_id: str
    condition_key: str | None
    evidence_tier: str
    confidence: float
    source_episode_ids: tuple[str, ...] = ()
    source_round_ids: tuple[int, ...] = ()
    support_count: int = 0
    contradiction_count: int = 0
    first_seen_round: int = 0
    last_seen_round: int = 0
    observed_support_count: int = 0
    hypothesized_support_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MechanicGraphSnapshot:
    snapshot_handle: str
    mechanic_graph_version: str
    created_round_id: int
    created_pass_id: int
    material_change: bool
    state: dict[str, Any]
    indexes: dict[str, Any] = field(default_factory=dict)


def empty_mechanic_graph_state() -> dict[str, Any]:
    return {
        "nodes_by_id": {},
        "edges_by_id": {},
        "adjacency_out": {},
        "adjacency_in": {},
        "object_to_node_indexes": {},
        "pattern_to_node_indexes": {},
        "observed_edge_ids": [],
        "hypothesized_edge_ids": [],
    }


def build_mechanic_graph_indexes(nodes_by_id: dict[str, dict], edges_by_id: dict[str, dict]) -> dict[str, Any]:
    adjacency_out: dict[str, list[str]] = {}
    adjacency_in: dict[str, list[str]] = {}
    object_to_node_indexes: dict[str, list[str]] = {}
    pattern_to_node_indexes: dict[str, list[str]] = {}
    for node_id, node in nodes_by_id.items():
        object_ref = str(node.get("object_ref") or "")
        pattern_id = str(node.get("pattern_id") or "")
        if object_ref:
            object_to_node_indexes.setdefault(object_ref, []).append(str(node_id))
        if pattern_id:
            pattern_to_node_indexes.setdefault(pattern_id, []).append(str(node_id))
    for edge_id, edge in edges_by_id.items():
        adjacency_out.setdefault(str(edge.get("src_node_id") or ""), []).append(str(edge_id))
        adjacency_in.setdefault(str(edge.get("dst_node_id") or ""), []).append(str(edge_id))
    return {
        "adjacency_out": {key: sorted(value) for key, value in adjacency_out.items()},
        "adjacency_in": {key: sorted(value) for key, value in adjacency_in.items()},
        "object_to_node_indexes": {key: sorted(set(value)) for key, value in object_to_node_indexes.items()},
        "pattern_to_node_indexes": {key: sorted(set(value)) for key, value in pattern_to_node_indexes.items()},
        "observed_edge_ids": sorted([edge_id for edge_id, edge in edges_by_id.items() if str(edge.get("evidence_tier") or "") == "observed"]),
        "hypothesized_edge_ids": sorted([edge_id for edge_id, edge in edges_by_id.items() if str(edge.get("evidence_tier") or "") != "observed"]),
    }


@dataclass
class MechanicGraphState:
    session_id: str
    game_id: str
    revision: int = 0
    state: dict[str, Any] = field(default_factory=empty_mechanic_graph_state)

    def snapshot(self, *, round_id: int, pass_id: int, material_change: bool) -> MechanicGraphSnapshot:
        state = dict(self.state)
        state.setdefault("nodes_by_id", {})
        state.setdefault("edges_by_id", {})
        indexes = build_mechanic_graph_indexes(dict(state["nodes_by_id"]), dict(state["edges_by_id"]))
        state.update(indexes)
        payload = {
            "version": f"mg:{self.session_id}:{round_id}:{self.revision}",
            "revision": self.revision,
            "state": state,
        }
        return MechanicGraphSnapshot(
            snapshot_handle=make_handle("snapshot:mechanic_graph", payload),
            mechanic_graph_version=str(payload["version"]),
            created_round_id=int(round_id),
            created_pass_id=int(pass_id),
            material_change=bool(material_change),
            state=state,
            indexes={
                "node_count": len(dict(state.get("nodes_by_id", {}))),
                "edge_count": len(dict(state.get("edges_by_id", {}))),
                "object_to_node_count": len(dict(state.get("object_to_node_indexes", {}))),
                "pattern_to_node_count": len(dict(state.get("pattern_to_node_indexes", {}))),
            },
        )
