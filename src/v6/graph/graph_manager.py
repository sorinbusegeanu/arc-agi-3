from __future__ import annotations

import networkx as nx

from v6.contingency.contingency_learner import Contingency
from v6.delta.delta_extractor import Delta
from v6.memory.interaction_store import Interaction
from v6.transformation.transformation_clusterer import TransformationFamily


class GraphManager:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self._last_interaction_node: str | None = None

    def add_interaction(self, interaction: Interaction) -> None:
        node = _interaction_node(interaction.id)
        self.graph.add_node(node, node_type="Interaction", id=int(interaction.id))
        if self._last_interaction_node is not None:
            self.graph.add_edge(self._last_interaction_node, node, edge_type="follows")
        self._last_interaction_node = node

    def add_delta(self, interaction_id: int, delta: Delta) -> None:
        node = _delta_node(delta.id)
        self.graph.add_node(node, node_type="Delta", id=int(delta.id), changed_cells=int(delta.changed_cells))
        self.graph.add_edge(_interaction_node(interaction_id), node, edge_type="generated")

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
            self.graph.add_edge(_delta_node(delta_id), _family_node(family_id), edge_type="member_of")

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
        self.graph.add_edge(node, _family_node(contingency.transformation_family), edge_type="predicts")


def _interaction_node(identifier: int) -> str:
    return f"interaction:{int(identifier)}"


def _delta_node(identifier: int) -> str:
    return f"delta:{int(identifier)}"


def _family_node(identifier: int) -> str:
    return f"family:{int(identifier)}"


def _contingency_node(identifier: int) -> str:
    return f"contingency:{int(identifier)}"
