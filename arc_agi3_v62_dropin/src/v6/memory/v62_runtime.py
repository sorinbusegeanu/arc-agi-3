from __future__ import annotations

import json
import math
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import replace
from hashlib import sha1
from typing import Any, Iterable

from v6.future_options import FutureOptionDelta, FutureOptionEstimator, FutureOptionSet
from v6.memory.controller import MemoryController
from v6.memory.migrations.v62 import migrate_connection as migrate_v62
from v6.memory.promotion_engine import MemoryPromotionEngine
from v6.memory.query_engine import MemoryActionScore, MemoryPrediction, MemoryQueryEngine
from v6.memory.substrate import MemoryEdge, MemoryNode, MemoryScore, MemorySubstrate, concept_node_id, world_model_node_id
from v6.memory.trajectory_efficiency import compact_state_hash

POLICY_VERSION = "v62_evidence_policy_v1"
SCORE_VERSION = "v62_hierarchical_isf_v1"


def _clamp01(value: float | int | None) -> float:
    if value is None:
        return 0.0
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is None:
        return set()
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()}


class LearnedFutureOptionEstimator:
    """Memory-derived reachability with the old estimator only as cold-start fallback."""

    def __init__(self, connection: sqlite3.Connection, *, fallback: FutureOptionEstimator | None = None) -> None:
        self.connection = connection
        self.fallback = fallback or FutureOptionEstimator()

    def estimate_option_set(
        self,
        env_or_state: Any,
        *,
        depth: int = 1,
        available_actions: list[int] | tuple[int, ...] | None = None,
    ) -> FutureOptionSet:
        depth = max(1, int(depth))
        actions = tuple(sorted(int(item) for item in (available_actions or ())))
        state_hash = self._state_hash(env_or_state)
        adjacency = self._learned_adjacency()
        if state_hash not in adjacency:
            return self.fallback.estimate_option_set(
                env_or_state, depth=depth, available_actions=list(actions)
            )

        reachable: set[str] = set()
        frontier: deque[tuple[str, int]] = deque([(state_hash, 0)])
        visited_depth: dict[str, int] = {state_hash: 0}
        while frontier:
            state, distance = frontier.popleft()
            if distance >= depth:
                continue
            for action, target, terminal in adjacency.get(state, ()):  # learned transitions only
                transition_signature = f"{state}|a{action}|{target}"
                reachable.add(transition_signature)
                if terminal:
                    continue
                next_distance = distance + 1
                old = visited_depth.get(target)
                if old is None or next_distance < old:
                    visited_depth[target] = next_distance
                    frontier.append((target, next_distance))

        if not reachable:
            return self.fallback.estimate_option_set(
                env_or_state, depth=depth, available_actions=list(actions)
            )
        option_set_id = "fos:v62:" + sha1(
            _json({"state": state_hash, "depth": depth, "reachable": sorted(reachable)}).encode("utf-8")
        ).hexdigest()[:20]
        first_actions = {item[0] for item in adjacency.get(state_hash, ())}
        return FutureOptionSet(
            option_set_id=option_set_id,
            state_signature=state_hash,
            available_actions=tuple(sorted(first_actions or set(actions))),
            reachable_signatures=tuple(sorted(reachable)),
            estimated_branching_factor=len(first_actions or set(actions)),
            depth=depth,
        )

    def compare(self, before: FutureOptionSet, after: FutureOptionSet, interaction_id: int) -> FutureOptionDelta:
        before_set = set(before.reachable_signatures)
        after_set = set(after.reachable_signatures)
        added = tuple(sorted(after_set - before_set))
        removed = tuple(sorted(before_set - after_set))
        preserved = tuple(sorted(before_set & after_set))
        return FutureOptionDelta(
            interaction_id=int(interaction_id),
            before_option_set_id=before.option_set_id,
            after_option_set_id=after.option_set_id,
            added_options=added,
            removed_options=removed,
            preserved_options=preserved,
            delta_score=float(len(added) - len(removed)),
        )

    def _state_hash(self, value: Any) -> str:
        try:
            return str(compact_state_hash(value))
        except Exception:
            return "state:" + sha1(_json(value).encode("utf-8")).hexdigest()[:24]

    def _learned_adjacency(self) -> dict[str, list[tuple[int, str, bool]]]:
        columns = _table_columns(self.connection, "interactions")
        required = {"state_hash_before", "state_hash_after", "action"}
        if not required.issubset(columns):
            return {}
        outcome_expr = "outcome_state" if "outcome_state" in columns else "NULL"
        rows = self.connection.execute(
            f"""
            SELECT state_hash_before, action, state_hash_after, {outcome_expr}
            FROM interactions
            WHERE state_hash_before IS NOT NULL
              AND state_hash_after IS NOT NULL
            ORDER BY id ASC
            """
        ).fetchall()
        output: dict[str, list[tuple[int, str, bool]]] = defaultdict(list)
        seen: set[tuple[str, int, str, bool]] = set()
        for before, action, after, outcome in rows:
            item = (str(before), int(action), str(after), str(outcome or "") in {"WIN", "GAME_OVER"})
            if item in seen:
                continue
            seen.add(item)
            output[item[0]].append((item[1], item[2], item[3]))
        return output


class HierarchicalSignificanceEngine:
    LEVELS = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")
    SOURCE_EDGE_TYPES = {
        "supports",
        "promoted_from",
        "derived_from",
        "abstracts_from",
        "plays_role",
        "transfers_to",
        "explains",
        "depends_on",
        "associated_with_family",
    }

    def __init__(self, memory: MemorySubstrate) -> None:
        self.memory = memory

    def development_stage(self) -> str:
        accepted_counts: dict[str, int] = {}
        for level in self.LEVELS:
            count = 0
            for node in self.memory.query_nodes(memory_level=level):
                status = str(node.get("attrs", {}).get("promotion_status", ""))
                if level == "M0" or status in {"accepted", "promoted"}:
                    count += 1
            accepted_counts[level] = count
        if accepted_counts["M4"] > 0:
            return "concept_transfer"
        if any(node.get("node_type") == "FunctionalRoleMemory" and str(node.get("attrs", {}).get("promotion_status")) == "accepted" for node in self.memory.query_nodes(memory_level="M3")):
            return "role_discovery"
        if accepted_counts["M2"] > 0:
            return "graph_expansion"
        if accepted_counts["M1"] > 0:
            return "environmental_influence"
        if accepted_counts["M0"] >= 10:
            return "movement_freedom"
        return "survival"

    def current_isf_weights(self) -> dict[str, float]:
        stage = self.development_stage()
        weights = {
            "survival": (0.40, 0.30, 0.20, 0.05, 0.05),
            "movement_freedom": (0.25, 0.30, 0.30, 0.075, 0.075),
            "environmental_influence": (0.15, 0.20, 0.30, 0.20, 0.15),
            "graph_expansion": (0.10, 0.20, 0.25, 0.25, 0.20),
            "role_discovery": (0.10, 0.15, 0.20, 0.30, 0.25),
            "concept_transfer": (0.10, 0.10, 0.15, 0.30, 0.35),
        }[stage]
        keys = ("survival_impact", "prediction_error", "learning_value", "transfer_potential", "explanatory_potential")
        return dict(zip(keys, weights))

    def rescore_all(self, *, step: int | None = None) -> dict[str, Any]:
        stage = self.development_stage()
        scored = 0
        for level in self.LEVELS[1:]:
            for node in self.memory.query_nodes(memory_level=level):
                if str(node.get("attrs", {}).get("promotion_status", "")) == "rejected":
                    continue
                source_scores = self._source_scores(str(node["node_id"]))
                attrs = dict(node.get("attrs") or {})
                source_isf = sum(source_scores) / len(source_scores) if source_scores else 0.0
                prediction = _clamp01(attrs.get("prediction_lift"))
                transfer = _clamp01(attrs.get("transfer_score") or self._success_rate(attrs))
                explanatory = _clamp01(self._normalize_reach(attrs.get("explanatory_reach")))
                compression = _clamp01(attrs.get("compression_gain"))
                future = _clamp01(abs(float(attrs.get("future_option_delta", 0.0) or 0.0)))
                if attrs.get("future_option_effect") in {"positive", "negative"}:
                    future = max(future, 0.5)
                total = _clamp01(
                    0.35 * source_isf
                    + 0.15 * prediction
                    + 0.15 * transfer
                    + 0.15 * explanatory
                    + 0.10 * compression
                    + 0.10 * future
                )
                self.memory.upsert_score(
                    MemoryScore(
                        node_id=str(node["node_id"]),
                        isf_total=total,
                        prediction_lift=prediction or None,
                        transfer_score=transfer or None,
                        explanatory_reach=explanatory or None,
                        compression_gain=compression or None,
                        future_option_delta=(future if future else None),
                        replay_priority=total,
                    ),
                    step=step,
                )
                self.memory.connection.execute(
                    """
                    UPDATE memory_scores
                    SET hierarchical_score=?, developmental_stage=?, source_score_count=?, score_version=?
                    WHERE node_id=?
                    """,
                    (total, stage, len(source_scores), SCORE_VERSION, str(node["node_id"])),
                )
                attrs.update({
                    "hierarchical_isf": total,
                    "developmental_stage": stage,
                    "hierarchical_score_version": SCORE_VERSION,
                    "source_score_count": len(source_scores),
                })
                self.memory.update_node_support_and_attrs(str(node["node_id"]), attrs, support_increment=0, step=step)
                scored += 1
        self.memory.connection.execute(
            """
            INSERT INTO memory_development_state(key, value_json, updated_step, updated_at)
            VALUES ('current', ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value_json=excluded.value_json,
                updated_step=excluded.updated_step,
                updated_at=excluded.updated_at
            """,
            (_json({"stage": stage, "weights": self.current_isf_weights(), "score_version": SCORE_VERSION}), step, time.time()),
        )
        self.memory.connection.commit()
        return {"scored": scored, "developmental_stage": stage}

    def _source_scores(self, node_id: str) -> list[float]:
        source_ids: set[str] = set()
        for edge in self.memory.edges_to(node_id):
            if str(edge.get("edge_type")) in self.SOURCE_EDGE_TYPES:
                source_ids.add(str(edge["source_node_id"]))
        for edge in self.memory.edges_from(node_id):
            if str(edge.get("edge_type")) in {"derived_from", "abstracts_from", "depends_on"}:
                source_ids.add(str(edge["target_node_id"]))
        scores: list[float] = []
        for source_id in source_ids:
            row = self.memory.connection.execute("SELECT isf_total FROM memory_scores WHERE node_id=?", (source_id,)).fetchone()
            if row is not None and row[0] is not None:
                scores.append(_clamp01(row[0]))
        return scores

    @staticmethod
    def _normalize_reach(value: Any) -> float:
        try:
            reach = max(0.0, float(value or 0.0))
        except (TypeError, ValueError):
            return 0.0
        return 1.0 - math.exp(-reach / 5.0)

    @staticmethod
    def _success_rate(attrs: dict[str, Any]) -> float:
        successes = float(attrs.get("transfer_success_count", 0) or 0)
        tests = float(attrs.get("transfer_tests", 0) or 0)
        return successes / tests if tests > 0 else 0.0


class MultiRoleAbstractionEngine:
    """M4 concepts are multi-role transfer compressions; M5 models are multi-concept relational components."""

    def __init__(self, memory: MemorySubstrate) -> None:
        self.memory = memory

    def run(self, *, step: int | None = None) -> dict[str, int]:
        concepts = self.promote_multi_role_concepts(step=step)
        world_models = self.promote_world_models(step=step)
        return {"multi_role_concepts": concepts, "world_models": world_models}

    def promote_multi_role_concepts(self, *, step: int | None = None) -> int:
        roles = [
            node for node in self.memory.query_nodes(memory_level="M3", node_type="FunctionalRoleMemory")
            if str(node.get("attrs", {}).get("promotion_status", "")) == "accepted"
        ]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for role in roles:
            desc = self._role_descriptor(role)
            if desc["transfer_tests"] <= 0 or desc["transfer_success_count"] <= 0:
                continue
            grouping_key = str(desc["future_option_effect"] or "neutral")
            grouped[grouping_key].append({"node": role, "desc": desc})

        created = 0
        covered_roles: set[str] = set()
        for effect, items in sorted(grouped.items()):
            if len(items) < 2:
                continue
            tests = sum(int(item["desc"]["transfer_tests"]) for item in items)
            successes = sum(int(item["desc"]["transfer_success_count"]) for item in items)
            if tests < 2 or successes / max(1, tests) < 0.5:
                continue
            role_ids = sorted(str(item["node"]["node_id"]) for item in items)
            carrier_ids = sorted({carrier for item in items for carrier in item["desc"]["carriers"]})
            families = sorted({family for item in items for family in item["desc"]["families"]})
            contexts = sorted({context for item in items for context in item["desc"]["contexts"]})
            outcomes = sorted({outcome for item in items for outcome in item["desc"]["outcomes"]})
            signature = sha1(_json({"roles": role_ids, "effect": effect}).encode("utf-8")).hexdigest()[:20]
            node_id = concept_node_id("v62:" + signature)
            attrs = {
                "source_roles": role_ids,
                "source_carriers": carrier_ids,
                "source_families": families,
                "transfer_tests": tests,
                "transfer_success_count": successes,
                "transfer_failure_count": max(0, tests - successes),
                "transfer_score": successes / max(1, tests),
                "applicability_contexts": contexts,
                "predicted_outcomes": outcomes,
                "future_option_effect": effect,
                "explanatory_reach": len(carrier_ids) + len(families),
                "compression_gain": max(0.0, (len(role_ids) - 1) / len(role_ids)),
                "promotion_status": "candidate",
                "concept_version": "v62_multi_role",
            }
            self.memory.upsert_node(MemoryNode(node_id=node_id, memory_level="M4", node_type="ConceptMemory", canonical_key=signature, attrs=attrs), step=step)
            for role_id in role_ids:
                self.memory.upsert_edge(MemoryEdge(role_id, node_id, "transfers_to", edge_source="v62_multi_role"))
                self.memory.upsert_edge(MemoryEdge(node_id, role_id, "derived_from", edge_source="v62_multi_role"))
                covered_roles.add(role_id)
            created += 1

        if covered_roles:
            for concept in self.memory.query_nodes(memory_level="M4", node_type="ConceptMemory"):
                attrs = dict(concept.get("attrs") or {})
                source_roles = [str(item) for item in attrs.get("source_roles", [])]
                if len(source_roles) == 1 and source_roles[0] in covered_roles and attrs.get("concept_version") != "v62_multi_role":
                    attrs["promotion_status"] = "superseded"
                    attrs["superseded_reason"] = "replaced_by_multi_role_transfer_compression"
                    self.memory.update_node_support_and_attrs(str(concept["node_id"]), attrs, support_increment=0, step=step)
        return created

    def promote_world_models(self, *, step: int | None = None) -> int:
        concepts = [
            node for node in self.memory.query_nodes(memory_level="M4", node_type="ConceptMemory")
            if str(node.get("attrs", {}).get("promotion_status", "")) == "accepted"
        ]
        if len(concepts) < 2:
            return 0
        adjacency: dict[str, set[str]] = defaultdict(set)
        descriptors = {str(node["node_id"]): self._concept_descriptor(node) for node in concepts}
        ids = sorted(descriptors)
        for i, left in enumerate(ids):
            for right in ids[i + 1 :]:
                ldesc, rdesc = descriptors[left], descriptors[right]
                if (ldesc["contexts"] & rdesc["contexts"]) or (ldesc["families"] & rdesc["families"]):
                    adjacency[left].add(right)
                    adjacency[right].add(left)
        components: list[set[str]] = []
        unseen = set(ids)
        while unseen:
            root = unseen.pop()
            component = {root}
            queue = [root]
            while queue:
                current = queue.pop()
                for nxt in adjacency.get(current, set()):
                    if nxt in unseen:
                        unseen.remove(nxt)
                        component.add(nxt)
                        queue.append(nxt)
            if len(component) >= 2:
                components.append(component)
        created = 0
        for component in components:
            concept_ids = sorted(component)
            contexts = sorted({x for cid in component for x in descriptors[cid]["contexts"]})
            outcomes = sorted({x for cid in component for x in descriptors[cid]["outcomes"]})
            families = sorted({x for cid in component for x in descriptors[cid]["families"]})
            contradiction_count = sum(len(self.memory.edges_to(cid, "contradicts")) + len(self.memory.edges_from(cid, "contradicts")) for cid in component)
            signature = sha1(_json({"concepts": concept_ids, "contexts": contexts, "families": families}).encode("utf-8")).hexdigest()[:20]
            node_id = world_model_node_id("v62:" + signature)
            attrs = {
                "concept_ids": concept_ids,
                "predicted_outcomes": outcomes,
                "supported_contexts": contexts,
                "source_families": families,
                "contradiction_count": contradiction_count,
                "explanatory_reach": sum(float(descriptors[cid]["reach"]) for cid in component),
                "promotion_status": "candidate",
                "world_model_version": "v62_relational",
            }
            self.memory.upsert_node(MemoryNode(node_id=node_id, memory_level="M5", node_type="WorldModelFragment", canonical_key=signature, attrs=attrs), step=step)
            for cid in concept_ids:
                self.memory.upsert_edge(MemoryEdge(cid, node_id, "explains", edge_source="v62_relational"))
                self.memory.upsert_edge(MemoryEdge(node_id, cid, "depends_on", edge_source="v62_relational"))
            created += 1
        return created

    def _role_descriptor(self, role: dict[str, Any]) -> dict[str, Any]:
        role_id = str(role["node_id"])
        carriers = {str(edge["source_node_id"]) for edge in self.memory.edges_to(role_id, "plays_role")}
        families: set[str] = set()
        contexts: set[str] = set()
        outcomes: set[str] = set()
        for carrier in carriers:
            for edge in self.memory.edges_from(carrier):
                edge_type = str(edge["edge_type"])
                if edge_type == "associated_with_family":
                    families.add(str(edge["target_node_id"]))
                elif edge_type == "appears_in_context":
                    contexts.add(str(edge["target_node_id"]))
                elif edge_type == "carried_by":
                    interaction_id = str(edge["target_node_id"])
                    for interaction_edge in self.memory.edges_from(interaction_id):
                        if str(interaction_edge["edge_type"]) == "has_outcome":
                            outcomes.add(str(interaction_edge["target_node_id"]))
        tests, successes = self._role_transfer_counts(str(role.get("attrs", {}).get("role_signature", role.get("canonical_key", ""))))
        return {
            "carriers": carriers,
            "families": families,
            "contexts": contexts,
            "outcomes": outcomes,
            "future_option_effect": role.get("attrs", {}).get("future_option_effect"),
            "transfer_tests": tests,
            "transfer_success_count": successes,
        }

    def _role_transfer_counts(self, signature: str) -> tuple[int, int]:
        columns = _table_columns(self.memory.connection, "role_transfer_attempts")
        if not columns or "role_signature" not in columns:
            return 0, 0
        success_col = "reuse_success" if "reuse_success" in columns else ("success" if "success" in columns else None)
        if success_col is None:
            return 0, 0
        row = self.memory.connection.execute(
            f"SELECT COUNT(*), COALESCE(SUM(CASE WHEN COALESCE({success_col},0)=1 THEN 1 ELSE 0 END),0) FROM role_transfer_attempts WHERE role_signature=?",
            (signature,),
        ).fetchone()
        return int(row[0] or 0), int(row[1] or 0)

    @staticmethod
    def _concept_descriptor(concept: dict[str, Any]) -> dict[str, Any]:
        attrs = dict(concept.get("attrs") or {})
        return {
            "contexts": set(str(x) for x in attrs.get("applicability_contexts", attrs.get("supported_contexts", [])) or []),
            "families": set(str(x) for x in attrs.get("source_families", []) or []),
            "outcomes": set(str(x) for x in attrs.get("predicted_outcomes", []) or []),
            "reach": float(attrs.get("explanatory_reach", 0.0) or 0.0),
        }


class V62PromotionEngine:
    def __init__(self, memory: MemorySubstrate, base: MemoryPromotionEngine | None = None) -> None:
        self.memory = memory
        self.base = base or MemoryPromotionEngine(memory)
        self.abstractions = MultiRoleAbstractionEngine(memory)
        self.significance = HierarchicalSignificanceEngine(memory)

    def run_all(self, step: int | None = None) -> dict[str, Any]:
        base_summary = self.base.run_all(step=step)
        first_validation = self._validate_levels({"M1", "M2", "M3"}, step=step)
        abstraction_summary = self.abstractions.run(step=step)
        second_validation = self._validate_levels({"M4", "M5", "M6"}, step=step)
        score_summary = self.significance.rescore_all(step=step)
        return {
            **base_summary,
            "v62_validation_pre_abstraction": first_validation,
            "v62_abstractions": abstraction_summary,
            "v62_validation_post_abstraction": second_validation,
            "v62_hierarchical_scoring": score_summary,
        }

    def _validate_levels(self, levels: set[str], *, step: int | None) -> dict[str, int]:
        accepted = rejected = 0
        for level in sorted(levels):
            for node in self.memory.query_nodes(memory_level=level):
                attrs = dict(node.get("attrs") or {})
                dimensions, mandatory, required = self._evidence_dimensions(node)
                dimension_count = sum(1 for value in dimensions.values() if bool(value))
                ok = all(dimensions.get(name, False) for name in mandatory) and dimension_count >= required
                attrs["promotion_status"] = "accepted" if ok else "rejected"
                attrs["promotion_policy_version"] = POLICY_VERSION
                attrs["promotion_evidence_dimensions"] = dimensions
                attrs["promotion_evidence_dimension_count"] = dimension_count
                reason = None if ok else f"requires {required} evidence dimensions and mandatory={sorted(mandatory)}"
                if reason:
                    attrs["promotion_rejection_reason"] = reason
                self.memory.update_node_support_and_attrs(str(node["node_id"]), attrs, support_increment=0, step=step)
                self.memory.connection.execute(
                    """
                    INSERT INTO memory_promotion_evidence_v62(
                        node_id, memory_level, node_type, evidence_dimensions_json,
                        evidence_dimension_count, required_dimension_count,
                        validation_status, validation_reason, updated_step, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        evidence_dimensions_json=excluded.evidence_dimensions_json,
                        evidence_dimension_count=excluded.evidence_dimension_count,
                        required_dimension_count=excluded.required_dimension_count,
                        validation_status=excluded.validation_status,
                        validation_reason=excluded.validation_reason,
                        updated_step=excluded.updated_step,
                        updated_at=excluded.updated_at
                    """,
                    (str(node["node_id"]), level, str(node["node_type"]), _json(dimensions), dimension_count, required, "accepted" if ok else "rejected", reason, step, time.time()),
                )
                if ok:
                    accepted += 1
                else:
                    rejected += 1
        self.memory.connection.commit()
        return {"accepted": accepted, "rejected": rejected}

    def _evidence_dimensions(self, node: dict[str, Any]) -> tuple[dict[str, bool], set[str], int]:
        level = str(node.get("memory_level"))
        node_type = str(node.get("node_type"))
        attrs = dict(node.get("attrs") or {})
        support = int(attrs.get("support_count", attrs.get("carrier_count", 0)) or 0)
        prediction = float(attrs.get("prediction_lift", 0.0) or 0.0)
        compression = float(attrs.get("compression_gain", 0.0) or 0.0)
        transfer_tests = int(attrs.get("transfer_tests", 0) or 0)
        transfer_successes = int(attrs.get("transfer_success_count", 0) or 0)
        role_count = len(attrs.get("source_roles", []) or [])
        concept_count = len(attrs.get("concept_ids", []) or [])
        explanatory = float(attrs.get("explanatory_reach", 0.0) or 0.0)
        incoming_lower = sum(1 for edge in self.memory.edges_to(str(node["node_id"])) if str(edge.get("source_node_id", "")).startswith(tuple(f"{x}:" for x in ("M0", "M1", "M2", "M3", "M4"))))
        future_effect = attrs.get("future_option_effect") in {"positive", "negative", "neutral"} or attrs.get("future_option_delta") is not None
        if level == "M1":
            dims = {"support": support >= 3, "confidence": float(attrs.get("confidence", 0.0) or 0.0) >= 0.6, "prediction": prediction > 0.0}
            return dims, {"support", "confidence"}, 2
        if level == "M2":
            dims = {"support": support >= 3 or incoming_lower >= 2, "compression": compression > 0.0, "explanatory": incoming_lower >= 2, "prediction": prediction > 0.0}
            return dims, {"support"}, 2
        if level == "M3" and node_type == "CarrierMemory":
            dims = {"support": support >= 3, "prediction": prediction > 0.0, "compression": compression > 0.0, "context_breadth": int(attrs.get("distinct_context_count", attrs.get("carrier_distinct_context_count", 0)) or 0) >= 2}
            return dims, {"support"}, 2
        if level == "M3":
            dims = {"structural_support": int(attrs.get("carrier_count", 0) or 0) >= 2, "transfer": float(attrs.get("transfer_score", 0.0) or 0.0) > 0.0, "future_option": future_effect, "explanatory": explanatory > 0.0 or int(attrs.get("carried_interaction_count", 0) or 0) > 0}
            return dims, {"structural_support"}, 2
        if level == "M4":
            dims = {"multi_role": role_count >= 2, "transfer_tests": transfer_tests >= 2, "transfer_success": transfer_successes / max(1, transfer_tests) >= 0.5, "compression": compression > 0.0, "explanatory": explanatory > 0.0}
            return dims, {"multi_role", "transfer_tests", "transfer_success"}, 3
        if level == "M5":
            dims = {"multi_concept": concept_count >= 2, "contexts": len(attrs.get("supported_contexts", []) or []) > 0, "outcomes": len(attrs.get("predicted_outcomes", []) or []) > 0, "explanatory": explanatory > 0.0}
            return dims, {"multi_concept"}, 2
        if level == "M6":
            cost = attrs.get("cost", attrs.get("current_cost"))
            best = attrs.get("best_known_length", attrs.get("best_known_cost"))
            dims = {"successful": float(attrs.get("success_rate", 0.0) or 0.0) > 0.0, "cost_known": cost is not None and best is not None, "cost_advantage": cost is not None and best is not None and float(cost) <= float(best), "reuse": int(attrs.get("reuse_count", 0) or 0) > 0, "equivalent_outcome": attrs.get("outcome_signature") is not None or attrs.get("effects") is not None}
            return dims, {"successful", "cost_known"}, 2
        return {}, set(), 99


class V62MemoryQueryEngine(MemoryQueryEngine):
    def find_similar_roles(self, context_signature: str, action: int) -> list[dict]:
        return [item for item in super().find_similar_roles(context_signature, action) if self._usable(item["node_id"])]

    def find_concept_matches(self, context_signature: str, action: int) -> list[dict]:
        return [item for item in super().find_concept_matches(context_signature, action) if self._usable(item["node_id"])]

    def score_action(
        self,
        context_signatures: dict[int, tuple],
        action: int,
        available_actions: list[int],
        *,
        record_query: bool = False,
    ) -> MemoryActionScore:
        base = super().score_action(context_signatures, action, available_actions, record_query=record_query)
        best_context = self._best_context_signature(context_signatures, action)
        strategy_score = self._strategy_score(best_context, action)
        world_model_score = self._world_model_score(best_context, action)
        score = _clamp01(float(base.score) + 0.10 * strategy_score + 0.05 * world_model_score)
        sources = list(base.evidence_sources)
        if strategy_score > 0:
            sources.append("M6_strategy_memory")
        if world_model_score > 0:
            sources.append("M5_world_model_memory")
        return replace(base, score=score, evidence_sources=sources)

    def _usable(self, node_id: str) -> bool:
        node = self.memory.get_node(str(node_id))
        if node is None:
            return False
        return str(node.get("attrs", {}).get("promotion_status", "accepted")) not in {"rejected", "superseded"}

    def _strategy_score(self, context_signature: str, action: int) -> float:
        best = 0.0
        for node in self.memory.query_nodes(memory_level="M6", node_type="EfficientStrategyMemory"):
            if not self._usable(str(node["node_id"])):
                continue
            attrs = dict(node.get("attrs") or {})
            sequence = attrs.get("action_sequence") or []
            if not sequence or int(sequence[0]) != int(action):
                continue
            context = str(attrs.get("context_key") or "")
            context_factor = 1.0 if not context or context == context_signature else 0.5
            success = _clamp01(attrs.get("success_rate"))
            cost = float(attrs.get("cost", len(sequence)) or len(sequence) or 1)
            best_len = float(attrs.get("best_known_length", cost) or cost)
            efficiency = _clamp01(best_len / max(cost, 1e-9))
            best = max(best, context_factor * (0.7 * success + 0.3 * efficiency))
        return best

    def _world_model_score(self, context_signature: str, action: int) -> float:
        role_ids = {str(item["node_id"]) for item in self.find_similar_roles(context_signature, action)}
        concept_ids: set[str] = set()
        for role_id in role_ids:
            for edge in self.memory.edges_from(role_id, "transfers_to"):
                if self._usable(str(edge["target_node_id"])):
                    concept_ids.add(str(edge["target_node_id"]))
        best = 0.0
        for node in self.memory.query_nodes(memory_level="M5", node_type="WorldModelFragment"):
            if not self._usable(str(node["node_id"])):
                continue
            attrs = dict(node.get("attrs") or {})
            overlap = concept_ids & set(str(x) for x in attrs.get("concept_ids", []) or [])
            if not overlap:
                continue
            contexts = set(str(x) for x in attrs.get("supported_contexts", []) or [])
            context_factor = 1.0 if not contexts or context_signature in contexts else 0.5
            reach = 1.0 - math.exp(-float(attrs.get("explanatory_reach", 0.0) or 0.0) / 5.0)
            best = max(best, context_factor * _clamp01(reach))
        return best


class V62MemoryController(MemoryController):
    """Mandatory v6.2 facade: heads compute; canonical memory owns persisted state."""

    def __init__(
        self,
        memory: MemorySubstrate,
        *,
        contingency_learner: Any = None,
        graph: Any = None,
        query_engine: Any | None = None,
        promotion_engine: Any | None = None,
        context_head: Any | None = None,
        carrier_head: Any | None = None,
        lifecycle_head: Any | None = None,
        efficiency_head: Any | None = None,
    ) -> None:
        migrate_v62(memory.connection)
        v62_query = V62MemoryQueryEngine(memory, contingency_learner=contingency_learner, graph=graph)
        base_promotion = promotion_engine if isinstance(promotion_engine, MemoryPromotionEngine) else MemoryPromotionEngine(memory)
        v62_promotion = V62PromotionEngine(memory, base=base_promotion)
        super().__init__(
            memory,
            contingency_learner=contingency_learner,
            graph=graph,
            query_engine=v62_query,
            promotion_engine=v62_promotion,
        )
        self.context_head = context_head
        self.carrier_head = carrier_head
        self.lifecycle_head = lifecycle_head
        self.efficiency_head = efficiency_head
        self.significance = v62_promotion.significance

    def predict(self, context_signatures: dict[int, tuple], action: int, *, record_query: bool = False) -> MemoryPrediction:
        return self.query_engine.predict_family(context_signatures, int(action), record_query=record_query)

    def current_isf_weights(self) -> dict[str, float]:
        return self.significance.current_isf_weights()

    def record_prediction_result(self, *args: Any, **kwargs: Any) -> Any:
        if self.context_head is None:
            return None
        return self.context_head.record_prediction_result(*args, **kwargs)

    def should_expand_context(self, *args: Any, **kwargs: Any) -> bool:
        return False if self.context_head is None else bool(self.context_head.should_expand_context(*args, **kwargs))

    def context_summary(self) -> dict[str, Any]:
        return {} if self.context_head is None else dict(self.context_head.summary())

    def record_carrier_interaction(self, *args: Any, **kwargs: Any) -> Any:
        if self.carrier_head is None:
            return None
        return self.carrier_head.record_interaction(*args, **kwargs)

    def carrier_stats(self, signature: str) -> dict[str, Any]:
        return {} if self.carrier_head is None else dict(self.carrier_head.stats_for_carrier(signature))

    def register_interaction(self, *args: Any, **kwargs: Any) -> Any:
        if self.lifecycle_head is None:
            raise RuntimeError("v6.2 lifecycle head is not configured")
        return self.lifecycle_head.register_interaction(*args, **kwargs)

    @property
    def replay_candidates(self) -> Any:
        return {} if self.lifecycle_head is None else self.lifecycle_head.replay_candidates

    def record_efficiency_interaction(self, *args: Any, **kwargs: Any) -> Any:
        if self.efficiency_head is None:
            raise RuntimeError("v6.2 efficiency head is not configured")
        return self.efficiency_head.record_interaction(*args, **kwargs)
