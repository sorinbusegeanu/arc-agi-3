from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha1
from typing import Any

from v6.memory.substrate import (
    MemoryEdge,
    MemoryNode,
    MemoryPromotion,
    MemorySubstrate,
    contingency_node_id,
    concept_node_id,
    family_node_id,
    interaction_node_id,
    role_node_id,
    strategy_node_id,
    world_model_node_id,
)


@dataclass(frozen=True)
class MemoryPromotionConfig:
    min_contingency_support: int = 3
    min_contingency_confidence: float = 0.6
    min_family_support: int = 3
    min_carrier_support: int = 3
    min_carrier_prediction_lift: float = 0.05
    min_carrier_compression_gain: float = 0.01
    min_role_support: int = 3
    min_role_transfer_score: float = 0.05
    min_concept_transfer_tests: int = 2
    min_concept_transfer_success_rate: float = 0.5


class MemoryPromotionEngine:
    def __init__(self, memory: MemorySubstrate, config: MemoryPromotionConfig | None = None) -> None:
        self.memory = memory
        self.config = config or MemoryPromotionConfig()

    def run_all(self, step: int | None = None) -> dict[str, Any]:
        summary = {
            "m0_to_m1": self.promote_m0_to_m1(step=step),
            "m1_to_m2": self.promote_m1_to_m2(step=step),
            "m2_to_m3_carrier": self.promote_m2_to_m3_carrier(step=step),
            "m3_carrier_to_role": self.promote_m3_carrier_to_role(step=step),
            "role_to_concept": self.promote_role_to_concept(step=step),
            "concept_to_world_model_fragment": self.promote_concept_to_world_model_fragment(step=step),
            "strategy_memories": self.promote_strategy_memories(step=step),
        }
        return summary

    def promote_m0_to_m1(self, *, step: int | None = None) -> dict[str, Any]:
        promoted = self._promote_m0_evidence_to_m1(step=step)
        promoted += self.validate_existing_m1_contingencies(step=step)["count"]
        return {"count": promoted}

    def validate_existing_m1_contingencies(self, *, step: int | None = None) -> dict[str, Any]:
        promoted = 0
        for node in self.memory.query_nodes(memory_level="M1", node_type="ContingencyMemory"):
            attrs = dict(node.get("attrs", {}))
            if int(attrs.get("support_count", 0) or 0) < self.config.min_contingency_support:
                continue
            if float(attrs.get("confidence", 0.0) or 0.0) < self.config.min_contingency_confidence:
                continue
            attrs["promotion_status"] = "promoted"
            self.memory.update_node_support_and_attrs(node["node_id"], attrs, step=step)
            self._record(
                source_node_id=node["node_id"],
                target_node_id=node["node_id"],
                promotion_type="M1_VALIDATION",
                evidence_count=int(attrs.get("support_count", 0) or 0),
                promotion_score=float(attrs.get("confidence", 0.0) or 0.0),
            )
            promoted += 1
        return {"count": promoted}

    def _promote_m0_evidence_to_m1(self, *, step: int | None = None) -> int:
        aggregates: dict[tuple[str, str, str], dict[str, Any]] = {}
        totals: dict[tuple[str, str], int] = defaultdict(int)
        for interaction in self.memory.query_nodes(memory_level="M0", node_type="InteractionMemory"):
            interaction_id = str(interaction["node_id"])
            action_edges = self.memory.edges_from(interaction_id, "takes_action")
            family_edges = [
                edge
                for edge in self.memory.edges_from(interaction_id, "supports")
                if str(edge["target_node_id"]).startswith("M2:family:")
            ]
            if not action_edges or not family_edges:
                continue
            attrs = dict(interaction.get("attrs", {}))
            context_signature = str(attrs.get("context_signature") or "**no_context**")
            action = str(action_edges[0]["target_node_id"]).split(":")[-1]
            for family_edge in family_edges:
                family = str(family_edge["target_node_id"]).split(":")[-1]
                key = (context_signature, action, family)
                row = aggregates.setdefault(
                    key,
                    {
                        "context_signature": context_signature,
                        "action": int(action),
                        "family": family,
                        "source_ids": [],
                    },
                )
                row["source_ids"].append(interaction_id)
                totals[(context_signature, action)] += 1
        promoted = 0
        for (context_signature, action, family), item in sorted(aggregates.items()):
            support_count = len(item["source_ids"])
            if support_count < self.config.min_contingency_support:
                continue
            total_for_context_action = max(1, totals[(context_signature, str(action))])
            confidence = float(support_count) / float(total_for_context_action)
            if confidence < self.config.min_contingency_confidence:
                continue
            contingency_id = contingency_node_id(0, context_signature, action, family)
            self.memory.upsert_node(
                MemoryNode(
                    node_id=contingency_id,
                    memory_level="M1",
                    node_type="ContingencyMemory",
                    canonical_key=f"0|{context_signature}|{action}|{family}",
                    attrs={
                        "context_level": 0,
                        "context_signature": context_signature,
                        "action": int(action),
                        "transformation_family": int(family),
                        "support_count": support_count,
                        "confidence": confidence,
                        "promotion_status": "promoted",
                    },
                ),
                step=step,
                support_increment=support_count,
            )
            self.memory.upsert_edge(MemoryEdge(contingency_id, family_node_id(family), "predicts"))
            for source_node_id in item["source_ids"]:
                self.memory.upsert_edge(MemoryEdge(source_node_id, contingency_id, "supports"))
                self.memory.upsert_edge(MemoryEdge(contingency_id, source_node_id, "derived_from"))
            self._record(
                source_node_id=str(item["source_ids"][0]),
                target_node_id=contingency_id,
                promotion_type="M0_M1",
                evidence_count=support_count,
                promotion_score=confidence,
            )
            promoted += 1
        return promoted

    def promote_m1_to_m2(self, *, step: int | None = None) -> dict[str, Any]:
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for node in self.memory.query_nodes(memory_level="M1", node_type="ContingencyMemory"):
            family = node.get("attrs", {}).get("transformation_family")
            if family is None:
                continue
            by_family[str(family)].append(node)
        promoted = 0
        for family, nodes in sorted(by_family.items()):
            support = sum(int(node.get("attrs", {}).get("support_count", 0) or 0) for node in nodes)
            if support < self.config.min_family_support:
                continue
            node_id = family_node_id(family)
            self.memory.upsert_node(
                MemoryNode(
                    node_id=node_id,
                    memory_level="M2",
                    node_type="TransformationFamilyMemory",
                    canonical_key=str(family),
                    attrs={"family_id": str(family), "support_count": support, "promotion_status": "promoted"},
                ),
                step=step,
            )
            for contingency in nodes:
                self.memory.upsert_edge(MemoryEdge(contingency["node_id"], node_id, "promoted_from"))
                self._record(
                    source_node_id=contingency["node_id"],
                    target_node_id=node_id,
                    promotion_type="M1_M2",
                    evidence_count=int(contingency.get("support_count", 0) or 0),
                    promotion_score=float(support),
                )
                promoted += 1
        return {"count": promoted}

    def promote_m2_to_m3_carrier(self, *, step: int | None = None) -> dict[str, Any]:
        promoted = 0
        for node in self.memory.query_nodes(memory_level="M3", node_type="CarrierMemory"):
            attrs = dict(node.get("attrs", {}))
            if str(attrs.get("carrier_source", "")) == "context_action_fallback":
                continue
            if int(attrs.get("support_count", 0) or 0) < self.config.min_carrier_support:
                continue
            if float(attrs.get("prediction_lift", 0.0) or 0.0) < self.config.min_carrier_prediction_lift:
                continue
            if float(attrs.get("compression_gain", 0.0) or 0.0) < self.config.min_carrier_compression_gain:
                continue
            attrs["promotion_status"] = "promoted"
            self.memory.update_node_support_and_attrs(node["node_id"], attrs, step=step)
            self._record(
                source_node_id=node["node_id"],
                target_node_id=node["node_id"],
                promotion_type="M2_M3_CARRIER",
                evidence_count=int(attrs.get("support_count", 0) or 0),
                promotion_score=float(attrs.get("prediction_lift", 0.0) or 0.0) + float(attrs.get("compression_gain", 0.0) or 0.0),
            )
            promoted += 1
        return {"count": promoted}

    def promote_m3_carrier_to_role(self, *, step: int | None = None) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        grouped_payloads: dict[str, dict[str, Any]] = {}
        carrier_nodes = [
            node for node in self.memory.query_nodes(memory_level="M3", node_type="CarrierMemory")
            if str(node.get("attrs", {}).get("promotion_status", "")) == "promoted"
        ]
        for carrier in carrier_nodes:
            outgoing = self.memory.edges_from(carrier["node_id"])
            incoming = self.memory.edges_to(carrier["node_id"])
            family_targets = sorted(
                edge["target_node_id"]
                for edge in outgoing
                if str(edge["edge_type"]) == "associated_with_family"
            )
            carried_interaction_ids = [
                str(edge["target_node_id"])
                for edge in outgoing
                if str(edge["edge_type"]) == "carried_by"
            ]
            interaction_edge_types: set[str] = set()
            outcome_targets: set[str] = set()
            expands_count = 0
            restricts_count = 0
            preserves_count = 0
            for interaction_id in carried_interaction_ids:
                for edge in self.memory.edges_from(interaction_id):
                    interaction_edge_types.add(str(edge["edge_type"]))
                    if str(edge["edge_type"]) == "expands_future_options":
                        expands_count += 1
                    elif str(edge["edge_type"]) == "restricts_future_options":
                        restricts_count += 1
                    elif str(edge["edge_type"]) == "preserves_future_options":
                        preserves_count += 1
                    elif str(edge["edge_type"]) == "has_outcome":
                        outcome_targets.add(str(edge["target_node_id"]))
            if expands_count > restricts_count:
                future_option_effect = "positive"
            elif restricts_count > expands_count:
                future_option_effect = "negative"
            else:
                future_option_effect = "neutral"
            signature_payload = {
                "outgoing": sorted({edge["edge_type"] for edge in outgoing}),
                "incoming": sorted({edge["edge_type"] for edge in incoming}),
                "carried_interaction_edge_types": sorted(interaction_edge_types),
                "families": family_targets,
                "outcomes": sorted(outcome_targets),
                "future_option_effect": future_option_effect,
            }
            role_signature = sha1(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
            grouped[role_signature].append(carrier)
            grouped_payloads[role_signature] = {
                "future_option_effect": future_option_effect,
                "expands_future_options_count": expands_count,
                "restricts_future_options_count": restricts_count,
                "preserves_future_options_count": preserves_count,
                "carried_interaction_count": len(carried_interaction_ids),
            }
        promoted = 0
        for role_signature, carriers in sorted(grouped.items()):
            if len(carriers) < self.config.min_role_support:
                continue
            node_id = role_node_id(role_signature)
            role_payload = dict(grouped_payloads.get(role_signature, {}))
            self.memory.upsert_node(
                MemoryNode(
                    node_id=node_id,
                    memory_level="M3",
                    node_type="FunctionalRoleMemory",
                    canonical_key=role_signature,
                    attrs={
                        "role_signature": role_signature,
                        "carrier_count": len(carriers),
                        "transfer_score": min(1.0, len(carriers) / 3.0),
                        **role_payload,
                    },
                ),
                step=step,
            )
            for carrier in carriers:
                self.memory.upsert_edge(MemoryEdge(carrier["node_id"], node_id, "plays_role"))
                self.memory.upsert_edge(MemoryEdge(node_id, carrier["node_id"], "abstracts_from"))
                self._record(
                    source_node_id=carrier["node_id"],
                    target_node_id=node_id,
                    promotion_type="M3_CARRIER_ROLE",
                    evidence_count=len(carriers),
                    promotion_score=min(1.0, len(carriers) / 3.0),
                )
                promoted += 1
        return {"count": promoted}

    def promote_role_to_concept(self, *, step: int | None = None) -> dict[str, Any]:
        promoted = 0
        for role in self.memory.query_nodes(memory_level="M3", node_type="FunctionalRoleMemory"):
            attrs = dict(role.get("attrs", {}))
            carrier_count = int(attrs.get("carrier_count", 0) or 0)
            transfer_tests = carrier_count
            success_rate = min(1.0, carrier_count / max(1.0, float(self.config.min_concept_transfer_tests)))
            if transfer_tests < self.config.min_concept_transfer_tests:
                continue
            if success_rate < self.config.min_concept_transfer_success_rate:
                continue
            concept_signature = sha1(str(attrs.get("role_signature", role["canonical_key"])).encode("utf-8")).hexdigest()[:20]
            node_id = concept_node_id(concept_signature)
            self.memory.upsert_node(
                MemoryNode(
                    node_id=node_id,
                    memory_level="M4",
                    node_type="ConceptMemory",
                    canonical_key=concept_signature,
                    attrs={
                        "source_roles": [role["node_id"]],
                        "transfer_tests": transfer_tests,
                        "transfer_success_count": transfer_tests,
                        "transfer_failure_count": 0,
                        "explanatory_reach": carrier_count,
                        "applicability_context": "cross_context",
                    },
                ),
                step=step,
            )
            self.memory.upsert_edge(MemoryEdge(role["node_id"], node_id, "transfers_to"))
            self.memory.upsert_edge(MemoryEdge(node_id, role["node_id"], "derived_from"))
            self._record(
                source_node_id=role["node_id"],
                target_node_id=node_id,
                promotion_type="M3_ROLE_M4_CONCEPT",
                evidence_count=transfer_tests,
                promotion_score=success_rate,
            )
            promoted += 1
        return {"count": promoted}

    def promote_concept_to_world_model_fragment(self, *, step: int | None = None) -> dict[str, Any]:
        promoted = 0
        for concept in self.memory.query_nodes(memory_level="M4", node_type="ConceptMemory"):
            attrs = dict(concept.get("attrs", {}))
            explanatory_reach = float(attrs.get("explanatory_reach", 0.0) or 0.0)
            if explanatory_reach <= 0.0:
                continue
            signature = sha1(str(concept["node_id"]).encode("utf-8")).hexdigest()[:20]
            node_id = world_model_node_id(signature)
            self.memory.upsert_node(
                MemoryNode(
                    node_id=node_id,
                    memory_level="M5",
                    node_type="WorldModelFragment",
                    canonical_key=signature,
                    attrs={
                        "concept_ids": [concept["node_id"]],
                        "predicted_outcomes": [],
                        "supported_contexts": [],
                        "contradiction_count": 0,
                        "explanatory_reach": explanatory_reach,
                    },
                ),
                step=step,
            )
            self.memory.upsert_edge(MemoryEdge(concept["node_id"], node_id, "explains"))
            self.memory.upsert_edge(MemoryEdge(node_id, concept["node_id"], "depends_on"))
            self._record(
                source_node_id=concept["node_id"],
                target_node_id=node_id,
                promotion_type="M4_M5_WORLD_MODEL",
                evidence_count=max(1, int(explanatory_reach)),
                promotion_score=explanatory_reach,
            )
            promoted += 1
        return {"count": promoted}

    def promote_strategy_memories(self, *, step: int | None = None) -> dict[str, Any]:
        promoted = 0
        for node in self.memory.query_nodes(memory_level="M6", node_type="EfficientStrategyMemory"):
            attrs = dict(node.get("attrs", {}))
            if attrs.get("best_known_cost") is None or attrs.get("current_cost") is None:
                continue
            score = float(attrs.get("normalized_solve_efficiency", 0.0) or 0.0)
            self._record(
                source_node_id=node["node_id"],
                target_node_id=node["node_id"],
                promotion_type="M5_M6_STRATEGY",
                evidence_count=1,
                promotion_score=score,
            )
            promoted += 1
        return {"count": promoted}

    def _record(
        self,
        *,
        source_node_id: str,
        target_node_id: str,
        promotion_type: str,
        evidence_count: int,
        promotion_score: float,
    ) -> None:
        if promotion_type == "M0_M1":
            if not str(source_node_id).startswith("M0:interaction:"):
                return
            if not str(target_node_id).startswith("M1:contingency:"):
                return
        promotion_id = f"promotion:{promotion_type}:{sha1(f'{source_node_id}|{target_node_id}|{promotion_type}'.encode('utf-8')).hexdigest()[:20]}"
        self.memory.record_promotion(
            MemoryPromotion(
                promotion_id=promotion_id,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                promotion_type=promotion_type,
                evidence_count=int(evidence_count),
                promotion_score=float(promotion_score),
                status="recorded",
            )
        )
