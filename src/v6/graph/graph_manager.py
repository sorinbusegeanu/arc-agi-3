from __future__ import annotations

import json
from collections import Counter
from typing import Any

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
    def __init__(
        self,
        *,
        max_total_edges: int = 1_000_000,
        max_edges_per_source: int = 128,
        min_edge_confidence: float = 0.0,
    ) -> None:
        self.graph = nx.MultiDiGraph()
        self._last_interaction_node: str | None = None
        self.max_total_edges = max(1, int(max_total_edges))
        self.max_edges_per_source = max(1, int(max_edges_per_source))
        self.min_edge_confidence = float(min_edge_confidence)
        self.rejected_edge_count = 0
        self.rejected_edge_reason_counts: Counter[str] = Counter()

    def add_interaction(self, interaction: Interaction) -> None:
        node = _interaction_node(interaction.id)
        self.graph.add_node(
            node,
            node_type="Interaction",
            id=int(interaction.id),
        )
        if self._last_interaction_node is not None:
            self.add_typed_edge(
                self._last_interaction_node,
                node,
                EDGE_FOLLOWS,
                edge_source="runtime_sequence",
            )
        self._last_interaction_node = node

    def add_delta(self, interaction_id: int, delta: Delta) -> None:
        node = _delta_node(delta.id)
        self.graph.add_node(
            node,
            node_type="Delta",
            id=int(delta.id),
            changed_cells=int(delta.changed_cells),
        )
        self.add_typed_edge(
            _interaction_node(interaction_id),
            node,
            EDGE_GENERATED,
            edge_source="delta_extractor",
        )

    def replace_families(
        self,
        families: dict[int, TransformationFamily],
        delta_to_family: dict[int, int],
    ) -> None:
        for family in families.values():
            family_node = _family_node(family.id)
            self.graph.add_node(
                family_node,
                node_type="TransformationFamily",
                id=int(family.id),
                support_count=int(family.support_count),
            )
        for delta_id, family_id in delta_to_family.items():
            self.add_typed_edge(
                _delta_node(delta_id),
                _family_node(family_id),
                EDGE_MEMBER_OF,
                edge_source="transformation_clusterer",
            )

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
        self.add_typed_edge(
            node,
            _family_node(contingency.transformation_family),
            EDGE_PREDICTS,
            weight=float(contingency.confidence),
            edge_confidence=float(contingency.confidence),
            evidence_count=int(contingency.support_count),
            specificity_score=1.0 / (1.0 + int(contingency.context_level)),
            edge_source="contingency_learner",
        )

    def add_typed_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        *,
        weight: float = 1.0,
        evidence: dict | None = None,
        edge_status: str = "accepted",
        edge_confidence: float | None = None,
        edge_source: str | None = None,
        evidence_count: int = 1,
        specificity_score: float | None = None,
        last_validated_epoch: int | None = None,
    ) -> bool:
        confidence = (
            float(weight)
            if edge_confidence is None
            else float(edge_confidence)
        )
        if confidence < self.min_edge_confidence:
            self._reject("confidence_below_threshold")
            return False
        if self.graph.number_of_edges() >= self.max_total_edges:
            self._reject("total_edge_cap")
            return False

        existing_for_source = sum(
            1
            for _source, _target, attrs in self.graph.out_edges(
                source_id, data=True
            )
            if str(attrs.get("edge_status", "accepted")) != "rejected"
        )
        has_same_edge = any(
            str(attrs.get("edge_type", attrs.get("type", "")))
            == str(edge_type)
            for attrs in self.graph.get_edge_data(
                source_id, target_id, default={}
            ).values()
        )
        if (
            not has_same_edge
            and existing_for_source >= self.max_edges_per_source
        ):
            self._reject("per_source_edge_cap")
            return False

        self._ensure_unknown_node(source_id)
        self._ensure_unknown_node(target_id)
        merged_evidence = dict(evidence or {})
        for attrs in self.graph.get_edge_data(
            source_id, target_id, default={}
        ).values():
            if str(
                attrs.get("type", attrs.get("edge_type", ""))
            ) != str(edge_type):
                continue
            existing_evidence = attrs.get("evidence") or {}
            if isinstance(existing_evidence, dict):
                merged_evidence = {
                    **existing_evidence,
                    **merged_evidence,
                }
            attrs.update(
                {
                    "type": str(edge_type),
                    "edge_type": str(edge_type),
                    "weight": max(
                        float(attrs.get("weight", 0.0)),
                        float(weight),
                    ),
                    "evidence": merged_evidence,
                    "support_count": int(
                        attrs.get("support_count", 0) or 0
                    )
                    + max(0, int(evidence_count)),
                    "edge_status": str(edge_status),
                    "edge_confidence": max(
                        float(attrs.get("edge_confidence", 0.0) or 0.0),
                        confidence,
                    ),
                    "edge_source": (
                        edge_source
                        or attrs.get("edge_source")
                        or "runtime"
                    ),
                    "specificity_score": (
                        specificity_score
                        if specificity_score is not None
                        else attrs.get("specificity_score")
                    ),
                    "last_validated_epoch": (
                        last_validated_epoch
                        if last_validated_epoch is not None
                        else attrs.get("last_validated_epoch")
                    ),
                }
            )
            return True

        self.graph.add_edge(
            source_id,
            target_id,
            type=str(edge_type),
            edge_type=str(edge_type),
            weight=float(weight),
            evidence=merged_evidence,
            support_count=max(0, int(evidence_count)),
            edge_status=str(edge_status),
            edge_confidence=confidence,
            edge_source=str(edge_source or "runtime"),
            specificity_score=specificity_score,
            last_validated_epoch=last_validated_epoch,
        )
        return True

    def validate_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        *,
        accepted: bool,
        confidence: float,
        specificity_score: float | None = None,
        epoch: int | None = None,
    ) -> bool:
        changed = False
        for attrs in self.graph.get_edge_data(
            source_id, target_id, default={}
        ).values():
            if str(
                attrs.get("edge_type", attrs.get("type", ""))
            ) != str(edge_type):
                continue
            attrs["edge_status"] = (
                "accepted" if accepted else "rejected"
            )
            attrs["edge_confidence"] = float(confidence)
            attrs["specificity_score"] = specificity_score
            attrs["last_validated_epoch"] = epoch
            attrs["edge_source"] = "validation"
            changed = True
        return changed

    def add_enables(self, source_id: str, target_id: str, **kwargs: Any) -> None:
        self.add_typed_edge(source_id, target_id, EDGE_ENABLES, **kwargs)

    def add_blocks(self, source_id: str, target_id: str, **kwargs: Any) -> None:
        self.add_typed_edge(source_id, target_id, EDGE_BLOCKS, **kwargs)

    def add_restricts(self, source_id: str, target_id: str, **kwargs: Any) -> None:
        self.add_typed_edge(source_id, target_id, EDGE_RESTRICTS, **kwargs)

    def add_expands(self, source_id: str, target_id: str, **kwargs: Any) -> None:
        self.add_typed_edge(source_id, target_id, EDGE_EXPANDS, **kwargs)

    def add_terminates(self, source_id: str, target_id: str, **kwargs: Any) -> None:
        self.add_typed_edge(source_id, target_id, EDGE_TERMINATES, **kwargs)

    def add_reversible_with(
        self,
        source_id: str,
        target_id: str,
        **kwargs: Any,
    ) -> None:
        self.add_typed_edge(
            source_id, target_id, EDGE_REVERSIBLE_WITH, **kwargs
        )
        self.add_typed_edge(
            target_id, source_id, EDGE_REVERSIBLE_WITH, **kwargs
        )

    def add_explains(self, source_id: str, target_id: str, **kwargs: Any) -> None:
        self.add_typed_edge(source_id, target_id, EDGE_EXPLAINS, **kwargs)

    def add_similar_role_to(
        self,
        source_id: str,
        target_id: str,
        **kwargs: Any,
    ) -> None:
        self.add_typed_edge(
            source_id, target_id, EDGE_SIMILAR_ROLE_TO, **kwargs
        )
        self.add_typed_edge(
            target_id, source_id, EDGE_SIMILAR_ROLE_TO, **kwargs
        )

    def add_depends_on(self, source_id: str, target_id: str, **kwargs: Any) -> None:
        self.add_typed_edge(source_id, target_id, EDGE_DEPENDS_ON, **kwargs)

    def add_contradicts(self, source_id: str, target_id: str, **kwargs: Any) -> None:
        self.add_typed_edge(source_id, target_id, EDGE_CONTRADICTS, **kwargs)

    def edge_type_counts(self) -> dict[str, int]:
        counts = {edge_type: 0 for edge_type in KNOWN_EDGE_TYPES}
        for _source, _target, attrs in self.graph.edges(data=True):
            edge_type = str(
                attrs.get("type", attrs.get("edge_type", ""))
            )
            if edge_type in counts:
                counts[edge_type] += 1
        return counts

    def edge_status_counts(self) -> dict[str, int]:
        counter = Counter(
            str(attrs.get("edge_status", "accepted"))
            for _source, _target, attrs in self.graph.edges(data=True)
        )
        return dict(sorted(counter.items()))

    def count_edges_of_type(self, edge_type: str) -> int:
        return int(self.edge_type_counts().get(str(edge_type), 0))

    def import_node(
        self,
        node_id: str,
        node_type: str,
        canonical_key: str | None,
        attrs: dict | None = None,
    ) -> None:
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
        edge_status: str = "accepted",
        edge_confidence: float | None = None,
        edge_source: str | None = None,
        specificity_score: float | None = None,
        last_validated_epoch: int | None = None,
    ) -> None:
        self.add_typed_edge(
            str(source_node_id),
            str(target_node_id),
            str(edge_type),
            weight=float(weight),
            evidence=evidence,
            evidence_count=int(support_count or 1),
            edge_status=edge_status,
            edge_confidence=edge_confidence,
            edge_source=edge_source or "compact_restore",
            specificity_score=specificity_score,
            last_validated_epoch=last_validated_epoch,
        )

    def export_compact_rows(self) -> dict[str, list[dict[str, object]]]:
        nodes: list[dict[str, object]] = []
        for node_id, attrs in self.graph.nodes(data=True):
            serializable = {
                key: _to_jsonable(value)
                for key, value in attrs.items()
            }
            nodes.append(
                {
                    "node_id": str(node_id),
                    "node_type": str(
                        attrs.get("node_type", "Unknown")
                    ),
                    "canonical_key": (
                        None
                        if attrs.get("canonical_key") is None
                        else str(attrs.get("canonical_key"))
                    ),
                    "support_count": attrs.get("support_count"),
                    "attrs_json": json.dumps(
                        serializable, sort_keys=True
                    ),
                }
            )
        edges: list[dict[str, object]] = []
        for source, target, attrs in self.graph.edges(data=True):
            evidence = (
                attrs.get("evidence")
                if isinstance(attrs.get("evidence"), dict)
                else {}
            )
            edges.append(
                {
                    "source_node_id": str(source),
                    "target_node_id": str(target),
                    "edge_type": str(
                        attrs.get(
                            "type",
                            attrs.get("edge_type", ""),
                        )
                    ),
                    "weight": float(attrs.get("weight", 1.0) or 1.0),
                    "support_count": int(
                        attrs.get("support_count", 0) or 0
                    ),
                    "evidence_json": json.dumps(
                        _to_jsonable(evidence),
                        sort_keys=True,
                    ),
                    "edge_status": str(
                        attrs.get("edge_status", "accepted")
                    ),
                    "edge_confidence": attrs.get("edge_confidence"),
                    "edge_source": attrs.get("edge_source"),
                    "specificity_score": attrs.get(
                        "specificity_score"
                    ),
                    "last_validated_epoch": attrs.get(
                        "last_validated_epoch"
                    ),
                }
            )
        return {"nodes": nodes, "edges": edges}

    def diagnostics(self) -> dict[str, Any]:
        return {
            "edge_count": self.graph.number_of_edges(),
            "node_count": self.graph.number_of_nodes(),
            "edge_type_counts": self.edge_type_counts(),
            "edge_status_counts": self.edge_status_counts(),
            "rejected_edge_count": self.rejected_edge_count,
            "rejected_edge_reason_counts": dict(
                sorted(self.rejected_edge_reason_counts.items())
            ),
            "max_total_edges": self.max_total_edges,
            "max_edges_per_source": self.max_edges_per_source,
            "min_edge_confidence": self.min_edge_confidence,
        }

    def _reject(self, reason: str) -> None:
        self.rejected_edge_count += 1
        self.rejected_edge_reason_counts[str(reason)] += 1

    def _ensure_unknown_node(self, node_id: str) -> None:
        if self.graph.has_node(node_id):
            return
        self.graph.add_node(
            node_id,
            node_type="Unknown",
            id=str(node_id),
        )


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
        return {
            str(key): _to_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
