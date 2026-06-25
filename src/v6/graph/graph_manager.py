from __future__ import annotations

import json

import networkx as nx

from v6.contingency.contingency_learner import Contingency
from v6.delta.delta_extractor import Delta
from v6.memory.interaction_store import Interaction
from v6.transformation.transformation_clusterer import TransformationFamily

EDGE_FOLLOWS = "follows"
EDGE_GENERATED = "generated"
EDGE_MEMBER_OF = "member_of"
EDGE_PREDICTS = "predicts"
EDGE_ENABLES = "enables"
EDGE_BLOCKS = "blocks"
EDGE_RESTRICTS = "restricts"
EDGE_EXPANDS = "expands"
EDGE_TERMINATES = "terminates"
EDGE_REVERSIBLE_WITH = "reversible_with"
EDGE_EXPLAINS = "explains"
EDGE_SIMILAR_ROLE_TO = "similar_role_to"
EDGE_DEPENDS_ON = "depends_on"
EDGE_CONTRADICTS = "contradicts"

KNOWN_EDGE_TYPES = (
    EDGE_FOLLOWS,
    EDGE_GENERATED,
    EDGE_MEMBER_OF,
    EDGE_PREDICTS,
    EDGE_ENABLES,
    EDGE_BLOCKS,
    EDGE_RESTRICTS,
    EDGE_EXPANDS,
    EDGE_TERMINATES,
    EDGE_REVERSIBLE_WITH,
    EDGE_EXPLAINS,
    EDGE_SIMILAR_ROLE_TO,
    EDGE_DEPENDS_ON,
    EDGE_CONTRADICTS,
)


class GraphManager:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self._last_interaction_node: str | None = None

    def add_interaction(self, interaction: Interaction) -> None:
        node = _interaction_node(interaction.id)
        self.graph.add_node(node, node_type="Interaction", id=int(interaction.id))
        if self._last_interaction_node is not None:
            self.add_typed_edge(self._last_interaction_node, node, EDGE_FOLLOWS)
        self._last_interaction_node = node

    def add_delta(self, interaction_id: int, delta: Delta) -> None:
        node = _delta_node(delta.id)
        self.graph.add_node(node, node_type="Delta", id=int(delta.id), changed_cells=int(delta.changed_cells))
        self.add_typed_edge(_interaction_node(interaction_id), node, EDGE_GENERATED)

    def replace_families(self, families: dict[int, TransformationFamily], delta_to_family: dict[int, int]) -> None:
        for family in families.values():
            family_node = _family_node(family.id)
            self.graph.add_node(
                family_node,
                node_type="TransformationFamily",
                id=int(family.id),
                support_count=int(family.support_count),
            )
        for delta_id, family_id in delta_to_family.items():
            self.add_typed_edge(_delta_node(delta_id), _family_node(family_id), EDGE_MEMBER_OF)

    def add_contingency(self, contingency: Contingency) -> None:
        node = _contingency_node(contingency.id)
        self.graph.add_node(
            node,
            node_type="Contingency",
            id=int(contingency.id),
            context_level=int(contingency.context_level),
            support_count=int(contingency.support_count),
            confidence=float(contingency.confidence),
        )
        self.add_typed_edge(node, _family_node(contingency.transformation_family), EDGE_PREDICTS)

    def add_typed_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        *,
        weight: float = 1.0,
        evidence: dict | None = None,
    ) -> None:
        self._ensure_unknown_node(source_id)
        self._ensure_unknown_node(target_id)
        merged_evidence = dict(evidence or {})
        for _key, attrs in self.graph.get_edge_data(source_id, target_id, default={}).items():
            if str(attrs.get("type", attrs.get("edge_type", ""))) != str(edge_type):
                continue
            existing_evidence = attrs.get("evidence") or {}
            if isinstance(existing_evidence, dict):
                merged_evidence = {**existing_evidence, **merged_evidence}
            attrs["type"] = str(edge_type)
            attrs["edge_type"] = str(edge_type)
            attrs["weight"] = max(float(attrs.get("weight", 0.0)), float(weight))
            attrs["evidence"] = merged_evidence
            return
        self.graph.add_edge(
            source_id,
            target_id,
            type=str(edge_type),
            edge_type=str(edge_type),
            weight=float(weight),
            evidence=merged_evidence,
        )

    def add_enables(self, source_id: str, target_id: str, *, weight: float = 1.0, evidence: dict | None = None) -> None:
        self.add_typed_edge(source_id, target_id, EDGE_ENABLES, weight=weight, evidence=evidence)

    def add_blocks(self, source_id: str, target_id: str, *, weight: float = 1.0, evidence: dict | None = None) -> None:
        self.add_typed_edge(source_id, target_id, EDGE_BLOCKS, weight=weight, evidence=evidence)

    def add_restricts(self, source_id: str, target_id: str, *, weight: float = 1.0, evidence: dict | None = None) -> None:
        self.add_typed_edge(source_id, target_id, EDGE_RESTRICTS, weight=weight, evidence=evidence)

    def add_expands(self, source_id: str, target_id: str, *, weight: float = 1.0, evidence: dict | None = None) -> None:
        self.add_typed_edge(source_id, target_id, EDGE_EXPANDS, weight=weight, evidence=evidence)

    def add_terminates(self, source_id: str, target_id: str, *, weight: float = 1.0, evidence: dict | None = None) -> None:
        self.add_typed_edge(source_id, target_id, EDGE_TERMINATES, weight=weight, evidence=evidence)

    def add_reversible_with(self, source_id: str, target_id: str, *, weight: float = 1.0, evidence: dict | None = None) -> None:
        self.add_typed_edge(source_id, target_id, EDGE_REVERSIBLE_WITH, weight=weight, evidence=evidence)
        self.add_typed_edge(target_id, source_id, EDGE_REVERSIBLE_WITH, weight=weight, evidence=evidence)

    def add_explains(self, source_id: str, target_id: str, *, weight: float = 1.0, evidence: dict | None = None) -> None:
        self.add_typed_edge(source_id, target_id, EDGE_EXPLAINS, weight=weight, evidence=evidence)

    def add_similar_role_to(self, source_id: str, target_id: str, *, weight: float = 1.0, evidence: dict | None = None) -> None:
        self.add_typed_edge(source_id, target_id, EDGE_SIMILAR_ROLE_TO, weight=weight, evidence=evidence)
        self.add_typed_edge(target_id, source_id, EDGE_SIMILAR_ROLE_TO, weight=weight, evidence=evidence)

    def add_depends_on(self, source_id: str, target_id: str, *, weight: float = 1.0, evidence: dict | None = None) -> None:
        self.add_typed_edge(source_id, target_id, EDGE_DEPENDS_ON, weight=weight, evidence=evidence)

    def add_contradicts(self, source_id: str, target_id: str, *, weight: float = 1.0, evidence: dict | None = None) -> None:
        self.add_typed_edge(source_id, target_id, EDGE_CONTRADICTS, weight=weight, evidence=evidence)

    def edge_type_counts(self) -> dict[str, int]:
        counts = {edge_type: 0 for edge_type in KNOWN_EDGE_TYPES}
        for _source, _target, attrs in self.graph.edges(data=True):
            edge_type = str(attrs.get("type", attrs.get("edge_type", "")))
            if edge_type in counts:
                counts[edge_type] += 1
        return counts

    def count_edges_of_type(self, edge_type: str) -> int:
        return int(self.edge_type_counts().get(str(edge_type), 0))

    def import_node(self, node_id: str, node_type: str, canonical_key: str | None, attrs: dict | None = None) -> None:
        merged = dict(attrs or {})
        merged.setdefault("node_type", str(node_type))
        if canonical_key is not None:
            merged.setdefault("canonical_key", str(canonical_key))
        self.graph.add_node(str(node_id), **merged)

    def import_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_type: str,
        weight: float = 1.0,
        evidence: dict | None = None,
        support_count: int | None = None,
    ) -> None:
        self.add_typed_edge(
            str(source_node_id),
            str(target_node_id),
            str(edge_type),
            weight=float(weight),
            evidence=evidence,
        )
        edge_data = self.graph.get_edge_data(str(source_node_id), str(target_node_id), default={})
        for attrs in edge_data.values():
            if str(attrs.get("type", attrs.get("edge_type", ""))) != str(edge_type):
                continue
            if support_count is not None:
                attrs["support_count"] = max(int(attrs.get("support_count", 0) or 0), int(support_count))
            break

    def export_compact_rows(self) -> dict[str, list[dict[str, object]]]:
        nodes: list[dict[str, object]] = []
        for node_id, attrs in self.graph.nodes(data=True):
            serializable = {key: _to_jsonable(value) for key, value in attrs.items()}
            nodes.append(
                {
                    "node_id": str(node_id),
                    "node_type": str(attrs.get("node_type", "Unknown")),
                    "canonical_key": None if attrs.get("canonical_key") is None else str(attrs.get("canonical_key")),
                    "support_count": attrs.get("support_count"),
                    "attrs_json": json.dumps(serializable, sort_keys=True),
                }
            )
        edges: list[dict[str, object]] = []
        for source, target, attrs in self.graph.edges(data=True):
            evidence = attrs.get("evidence") if isinstance(attrs.get("evidence"), dict) else {}
            edges.append(
                {
                    "source_node_id": str(source),
                    "target_node_id": str(target),
                    "edge_type": str(attrs.get("type", attrs.get("edge_type", ""))),
                    "weight": float(attrs.get("weight", 1.0) or 1.0),
                    "support_count": attrs.get("support_count"),
                    "evidence_json": json.dumps(_to_jsonable(evidence), sort_keys=True),
                }
            )
        return {"nodes": nodes, "edges": edges}

    def _ensure_unknown_node(self, node_id: str) -> None:
        if self.graph.has_node(node_id):
            return
        self.graph.add_node(node_id, node_type="Unknown", id=str(node_id))


def _interaction_node(identifier: int) -> str:
    return f"interaction:{int(identifier)}"


def _delta_node(identifier: int) -> str:
    return f"delta:{int(identifier)}"


def _family_node(identifier: int) -> str:
    return f"family:{int(identifier)}"


def _contingency_node(identifier: int) -> str:
    return f"contingency:{int(identifier)}"


def _to_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
