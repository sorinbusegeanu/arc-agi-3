from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Iterable

from v6.memory.migrations.v63 import migrate_connection as migrate_v63
from v6.memory.substrate import MemoryEdge, MemoryNode, MemoryScore, concept_node_id, world_model_node_id

SCORE_POLICY_VERSION = "v63_unified_memory_fitness_v1"
RUNTIME_POLICY_VERSION = "v63_runtime_policy_v1"
ABSTRACTION_VERSION = "v63_relational_abstraction_v1"


@dataclass(frozen=True)
class V63CandidateBudget:
    max_role_candidates: int = 4096
    max_role_pair_comparisons: int = 50_000
    max_concept_candidates: int = 2048
    max_concept_pair_comparisons: int = 25_000


_BUDGET = V63CandidateBudget()
_PATCHED = False


def configure_candidate_budget(
    *,
    max_role_candidates: int | None = None,
    max_role_pair_comparisons: int | None = None,
    max_concept_candidates: int | None = None,
    max_concept_pair_comparisons: int | None = None,
) -> V63CandidateBudget:
    global _BUDGET
    _BUDGET = V63CandidateBudget(
        max_role_candidates=max(2, int(max_role_candidates or _BUDGET.max_role_candidates)),
        max_role_pair_comparisons=max(
            1,
            int(max_role_pair_comparisons or _BUDGET.max_role_pair_comparisons),
        ),
        max_concept_candidates=max(
            2,
            int(max_concept_candidates or _BUDGET.max_concept_candidates),
        ),
        max_concept_pair_comparisons=max(
            1,
            int(max_concept_pair_comparisons or _BUDGET.max_concept_pair_comparisons),
        ),
    )
    return _BUDGET


def clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def bounded_support(value: Any, *, scale: float = 5.0) -> float | None:
    try:
        support = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(support) or support < 0.0:
        return None
    return clamp01(1.0 - math.exp(-support / max(1e-9, float(scale))))


def resolve_transfer_evidence(
    attrs: dict[str, Any],
) -> tuple[float | None, float | None, float | None, str]:
    prior_raw = attrs.get("transfer_prior")
    tests = int(attrs.get("transfer_tests", 0) or 0)
    successes = int(attrs.get("transfer_success_count", 0) or 0)

    empirical_raw = attrs.get("transfer_empirical_rate")
    empirical: float | None
    if tests > 0:
        empirical = clamp01(successes / max(1, tests))
    elif empirical_raw is not None:
        empirical = clamp01(empirical_raw)
    else:
        empirical = None

    if prior_raw is not None:
        prior = clamp01(prior_raw)
    elif tests <= 0 and attrs.get("transfer_score") is not None:
        # Backward compatibility: pre-v6.3 transfer_score on untested
        # abstractions was a structural proxy, not empirical evidence.
        prior = clamp01(attrs.get("transfer_score"))
    elif attrs.get("structural_overlap_score") is not None:
        prior = clamp01(attrs.get("structural_overlap_score"))
    else:
        prior = None

    effective = empirical if empirical is not None else prior
    if tests > 0:
        status = "empirical"
    elif prior is not None:
        status = "prior_only"
    else:
        status = "untested"
    return prior, empirical, effective, status


def unified_memory_fitness(
    *,
    isf_score: float | None,
    explanatory_reach: float | None,
    transfer_prior: float | None,
    transfer_empirical: float | None,
    recurrence_score: float | None,
    efficiency_score: float | None,
) -> tuple[float, dict[str, float]]:
    # v6.3 deliberately uses a monotone equal-weight mean over only active,
    # bounded dimensions. Developmental weighting belongs inside ISF; memory
    # fitness must not re-add PE/LV/TP/EP with a second arbitrary coefficient set.
    transfer = transfer_empirical if transfer_empirical is not None else transfer_prior
    candidates = {
        "isf": isf_score,
        "explanatory_reach": explanatory_reach,
        "transfer": transfer,
        "recurrence": recurrence_score,
        "efficiency": efficiency_score,
    }
    active = {
        name: clamp01(value)
        for name, value in candidates.items()
        if value is not None
    }
    if not active:
        return 0.0, {}
    return clamp01(sum(active.values()) / len(active)), active


def install_v63_runtime_policy() -> None:
    global _PATCHED
    if _PATCHED:
        return

    # Delayed imports avoid circular imports while promotion_engine is imported
    # by the v6.2/v6.2.1 runtime modules.
    from v6.memory.v62_runtime import HierarchicalSignificanceEngine
    from v6.memory.v621_runtime import V621AbstractionEngine

    HierarchicalSignificanceEngine.current_isf_weights = _current_isf_weights_v63
    HierarchicalSignificanceEngine.rescore_all = _rescore_all_v63
    V621AbstractionEngine.promote_multi_role_concepts = _promote_multi_role_concepts_v63
    V621AbstractionEngine.promote_world_models = _promote_world_models_v63
    _PATCHED = True


def _current_isf_weights_v63(self: Any) -> dict[str, float]:
    stage = self.development_stage()
    weights = {
        # Stage 0 has no prediction signal. Novelty/recurrence is represented by
        # learning_value; PE is activated only when an actual expectation exists.
        "survival": (0.65, 0.0, 0.35, 0.0, 0.0),
        "movement_freedom": (0.25, 0.30, 0.30, 0.075, 0.075),
        "environmental_influence": (0.15, 0.20, 0.30, 0.20, 0.15),
        "graph_expansion": (0.10, 0.20, 0.25, 0.25, 0.20),
        "role_discovery": (0.10, 0.15, 0.20, 0.30, 0.25),
        "concept_transfer": (0.10, 0.10, 0.15, 0.30, 0.35),
    }[stage]
    keys = (
        "survival_impact",
        "prediction_error",
        "learning_value",
        "transfer_potential",
        "explanatory_potential",
    )
    return dict(zip(keys, weights))


def _rescore_all_v63(self: Any, *, step: int | None = None) -> dict[str, Any]:
    migrate_v63(self.memory.connection)
    stage = self.development_stage()
    scored = 0
    for level in self.LEVELS[1:]:
        for node in self.memory.query_nodes(memory_level=level):
            attrs = dict(node.get("attrs") or {})
            if str(attrs.get("promotion_status", "")) == "rejected":
                continue

            source_scores = self._source_scores(str(node["node_id"]))
            source_isf = (
                sum(source_scores) / len(source_scores)
                if source_scores
                else None
            )

            explanatory_raw = attrs.get("explanatory_reach")
            explanatory = (
                self._normalize_reach(explanatory_raw)
                if explanatory_raw is not None
                else None
            )
            transfer_prior, transfer_empirical, transfer_effective, transfer_status = (
                resolve_transfer_evidence(attrs)
            )
            support = attrs.get(
                "support_count",
                attrs.get("carrier_count", attrs.get("transfer_tests")),
            )
            recurrence = bounded_support(support)

            efficiency: float | None = None
            if level == "M6":
                comparable = bool(
                    attrs.get("outcome_signature")
                    or attrs.get("effects")
                    or attrs.get("comparable_outcome_group_id")
                )
                if comparable:
                    raw_efficiency = attrs.get(
                        "normalized_solve_efficiency",
                        attrs.get("efficiency_score"),
                    )
                    if raw_efficiency is not None:
                        efficiency = clamp01(raw_efficiency)

            fitness, components = unified_memory_fitness(
                isf_score=source_isf,
                explanatory_reach=explanatory,
                transfer_prior=transfer_prior,
                transfer_empirical=transfer_empirical,
                recurrence_score=recurrence,
                efficiency_score=efficiency,
            )

            prediction = (
                clamp01(attrs.get("prediction_lift"))
                if attrs.get("prediction_lift") is not None
                else None
            )
            compression = (
                clamp01(attrs.get("compression_gain"))
                if attrs.get("compression_gain") is not None
                else None
            )
            future = None
            if attrs.get("future_option_delta") is not None:
                future = clamp01(abs(float(attrs.get("future_option_delta") or 0.0)))
            elif attrs.get("future_option_effect") in {"positive", "negative"}:
                future = 0.5

            self.memory.upsert_score(
                MemoryScore(
                    node_id=str(node["node_id"]),
                    isf_total=fitness,
                    prediction_lift=prediction,
                    transfer_score=transfer_effective,
                    explanatory_reach=explanatory,
                    compression_gain=compression,
                    future_option_delta=future,
                    replay_priority=fitness,
                ),
                step=step,
            )
            self.memory.connection.execute(
                """
                UPDATE memory_scores
                SET hierarchical_score=?,
                    developmental_stage=?,
                    source_score_count=?,
                    score_version=?,
                    transfer_prior=?,
                    transfer_empirical_rate=?,
                    transfer_evidence_status=?,
                    memory_fitness=?,
                    recurrence_score=?,
                    efficiency_score=?,
                    score_components_json=?,
                    prospective_learning_value=?,
                    realized_learning_value=?,
                    prospective_explanatory_potential=?,
                    realized_explanatory_reach=?,
                    score_policy_version=?
                WHERE node_id=?
                """,
                (
                    fitness,
                    stage,
                    len(source_scores),
                    SCORE_POLICY_VERSION,
                    transfer_prior,
                    transfer_empirical,
                    transfer_status,
                    fitness,
                    recurrence,
                    efficiency,
                    json.dumps(components, sort_keys=True),
                    attrs.get("learning_value"),
                    attrs.get("learning_value_realized"),
                    attrs.get("explanatory_potential"),
                    explanatory_raw,
                    SCORE_POLICY_VERSION,
                    str(node["node_id"]),
                ),
            )
            attrs.update(
                {
                    "hierarchical_isf": fitness,
                    "memory_fitness": fitness,
                    "developmental_stage": stage,
                    "hierarchical_score_version": SCORE_POLICY_VERSION,
                    "source_score_count": len(source_scores),
                    "transfer_prior": transfer_prior,
                    "transfer_empirical_rate": transfer_empirical,
                    "transfer_evidence_status": transfer_status,
                    "score_components": components,
                }
            )
            self.memory.update_node_support_and_attrs(
                str(node["node_id"]),
                attrs,
                support_increment=0,
                step=step,
            )
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
        (
            json.dumps(
                {
                    "stage": stage,
                    "weights": self.current_isf_weights(),
                    "score_version": SCORE_POLICY_VERSION,
                },
                sort_keys=True,
            ),
            step,
            time.time(),
        ),
    )
    self.memory.connection.commit()
    return {
        "scored": scored,
        "developmental_stage": stage,
        "score_policy_version": SCORE_POLICY_VERSION,
    }


def _candidate_priority(node: dict[str, Any]) -> tuple[float, float, float, str]:
    attrs = dict(node.get("attrs") or {})
    _prior, _empirical, effective, _status = resolve_transfer_evidence(attrs)
    transfer = effective if effective is not None else 0.0
    support = float(
        attrs.get("support_count", attrs.get("carrier_count", 0)) or 0.0
    )
    explanatory = float(attrs.get("explanatory_reach", 0.0) or 0.0)
    return (-transfer, -support, -explanatory, str(node.get("node_id", "")))


def _bounded_pairs(ids: list[str], budget: int) -> Iterable[tuple[str, str]]:
    used = 0
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            if used >= budget:
                return
            used += 1
            yield left, right


def _audit_frontier(
    connection: Any,
    *,
    frontier_kind: str,
    candidates_seen: int,
    candidates_retained: int,
    comparisons_attempted: int,
    comparison_budget: int,
    step: int | None,
) -> None:
    payload = (
        f"{frontier_kind}|{candidates_seen}|{candidates_retained}|"
        f"{comparisons_attempted}|{comparison_budget}|{step}"
    )
    audit_id = "v63frontier:" + sha1(payload.encode("utf-8")).hexdigest()[:24]
    connection.execute(
        """
        INSERT OR REPLACE INTO abstraction_frontier_audit_v63(
            audit_id, frontier_kind, candidates_seen, candidates_retained,
            comparisons_attempted, comparison_budget, global_step, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            audit_id,
            frontier_kind,
            int(candidates_seen),
            int(candidates_retained),
            int(comparisons_attempted),
            int(comparison_budget),
            step,
            time.time(),
        ),
    )


def _promote_multi_role_concepts_v63(
    self: Any,
    *,
    step: int | None = None,
) -> int:
    migrate_v63(self.memory.connection)
    all_roles = [
        node
        for node in self.memory.query_nodes(
            memory_level="M3",
            node_type="FunctionalRoleMemory",
        )
        if str(
            node.get("attrs", {}).get("promotion_status", "candidate")
        )
        in {"accepted", "candidate"}
    ]
    roles = sorted(all_roles, key=_candidate_priority)[: _BUDGET.max_role_candidates]
    descriptors = {
        str(role["node_id"]): self._role_descriptor(role)
        for role in roles
    }
    adjacency: dict[str, set[str]] = {}
    role_ids = sorted(descriptors)
    comparisons = 0
    for left, right in _bounded_pairs(
        role_ids,
        _BUDGET.max_role_pair_comparisons,
    ):
        comparisons += 1
        if self._roles_compatible(descriptors[left], descriptors[right]):
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)

    components = self._components(role_ids, adjacency)
    created = 0
    strict_role_sets: list[set[str]] = []
    for component in components:
        if len(component) < 2:
            continue
        items = [descriptors[item] for item in sorted(component)]
        role_set = set(component)
        families = sorted(
            {value for item in items for value in item["families"]}
        )
        contexts = sorted(
            {value for item in items for value in item["contexts"]}
        )
        outcomes = sorted(
            {value for item in items for value in item["outcomes"]}
        )
        games = sorted(
            {
                value
                for item in items
                for value in item["games"]
                if value
            }
        )
        effects = sorted(
            {
                str(item["future_option_effect"] or "neutral")
                for item in items
            }
        )
        overlap_score = self._component_overlap_score(items)
        if overlap_score <= 0.0:
            continue

        direct = self._direct_concept_transfer_for_roles(role_set)
        direct_tests = int(direct["tests"])
        direct_successes = int(direct["successes"])
        direct_rate = (
            direct_successes / direct_tests
            if direct_tests > 0
            else None
        )
        cross_game = len(games) >= 2 or len(contexts) >= 2
        transfer_prior = clamp01(
            0.5 * overlap_score + 0.5 * (1.0 if cross_game else 0.0)
        )
        status = (
            "accepted"
            if direct_tests >= 2
            and direct_rate is not None
            and direct_rate >= 0.5
            and cross_game
            else "candidate"
        )

        signature_payload = {
            "roles": sorted(role_set),
            "families": families,
            "effects": effects,
        }
        signature = sha1(
            json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
        # Keep the v6.2.1 identity prefix so existing empirical transfer attempts
        # remain attached to the same stable concept node.
        node_id = concept_node_id("v621:" + signature)
        attrs = {
            "source_roles": sorted(role_set),
            "source_families": families,
            "applicability_contexts": contexts,
            "predicted_outcomes": outcomes,
            "source_games": games,
            "future_option_effects": effects,
            "transfer_prior": transfer_prior,
            "transfer_tests": direct_tests,
            "transfer_success_count": direct_successes,
            "transfer_failure_count": max(0, direct_tests - direct_successes),
            "transfer_empirical_rate": direct_rate,
            "transfer_evidence_status": (
                "empirical" if direct_tests > 0 else "prior_only"
            ),
            "transfer_score": (
                direct_rate if direct_rate is not None else transfer_prior
            ),
            "structural_overlap_score": overlap_score,
            "cross_game_evidence": cross_game,
            "explanatory_reach": len(families) + len(contexts),
            "compression_gain": max(
                0.0,
                (len(role_set) - 1) / len(role_set),
            ),
            "promotion_status": status,
            "concept_version": ABSTRACTION_VERSION,
            "validation_source": (
                "direct_concept_transfer"
                if direct_tests
                else "structural_transfer_prior"
            ),
        }
        self.memory.upsert_node(
            MemoryNode(
                node_id=node_id,
                memory_level="M4",
                node_type="ConceptMemory",
                canonical_key=signature,
                attrs=attrs,
            ),
            step=step,
        )
        for role_id in sorted(role_set):
            self.memory.upsert_edge(
                MemoryEdge(
                    role_id,
                    node_id,
                    "transfers_to",
                    edge_source=ABSTRACTION_VERSION,
                )
            )
            self.memory.upsert_edge(
                MemoryEdge(
                    node_id,
                    role_id,
                    "derived_from",
                    edge_source=ABSTRACTION_VERSION,
                )
            )
        strict_role_sets.append(role_set)
        created += 1

    if strict_role_sets:
        for concept in self.memory.query_nodes(
            memory_level="M4",
            node_type="ConceptMemory",
        ):
            attrs = dict(concept.get("attrs") or {})
            if attrs.get("concept_version") == ABSTRACTION_VERSION:
                continue
            old_roles = {
                str(value)
                for value in attrs.get("source_roles", []) or []
            }
            if old_roles and any(
                old_roles.issubset(strict)
                for strict in strict_role_sets
            ):
                attrs["promotion_status"] = "superseded"
                attrs["superseded_reason"] = "replaced_by_v63_transfer_evidence_concept"
                self.memory.update_node_support_and_attrs(
                    str(concept["node_id"]),
                    attrs,
                    support_increment=0,
                    step=step,
                )

    _audit_frontier(
        self.memory.connection,
        frontier_kind="role_to_concept",
        candidates_seen=len(all_roles),
        candidates_retained=len(roles),
        comparisons_attempted=comparisons,
        comparison_budget=_BUDGET.max_role_pair_comparisons,
        step=step,
    )
    self.memory.connection.commit()
    return created


def _promote_world_models_v63(
    self: Any,
    *,
    step: int | None = None,
) -> int:
    migrate_v63(self.memory.connection)
    all_concepts = [
        node
        for node in self.memory.query_nodes(
            memory_level="M4",
            node_type="ConceptMemory",
        )
        if str(node.get("attrs", {}).get("promotion_status", "candidate")) == "accepted"
        and node.get("attrs", {}).get("concept_version") == ABSTRACTION_VERSION
    ]
    concepts = sorted(all_concepts, key=_candidate_priority)[: _BUDGET.max_concept_candidates]
    if len(concepts) < 2:
        _audit_frontier(
            self.memory.connection,
            frontier_kind="concept_to_world_model",
            candidates_seen=len(all_concepts),
            candidates_retained=len(concepts),
            comparisons_attempted=0,
            comparison_budget=_BUDGET.max_concept_pair_comparisons,
            step=step,
        )
        self.memory.connection.commit()
        return 0

    descriptors = {
        str(node["node_id"]): self._concept_descriptor(node)
        for node in concepts
    }
    pair_relations: dict[
        tuple[str, str],
        list[tuple[str, int, float, dict[str, Any]]],
    ] = {}
    adjacency: dict[str, set[str]] = {}
    ids = sorted(descriptors)
    comparisons = 0
    for left, right in _bounded_pairs(
        ids,
        _BUDGET.max_concept_pair_comparisons,
    ):
        comparisons += 1
        relations = self._relations_between(
            left,
            right,
            descriptors[left],
            descriptors[right],
        )
        if not relations:
            continue
        pair_relations[(left, right)] = relations
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)

    created = 0
    for component in self._components(ids, adjacency):
        if len(component) < 2:
            continue
        component_relations: list[
            tuple[str, str, str, int, float, dict[str, Any]]
        ] = []
        for left in sorted(component):
            for right in sorted(component):
                if left >= right:
                    continue
                for relation in pair_relations.get((left, right), ()):
                    relation_type, support, confidence, evidence = relation
                    component_relations.append(
                        (
                            left,
                            right,
                            relation_type,
                            support,
                            confidence,
                            evidence,
                        )
                    )
        if len(component_relations) < 2:
            continue

        concept_ids = sorted(component)
        contexts = sorted(
            {
                item
                for concept_id in component
                for item in descriptors[concept_id]["contexts"]
            }
        )
        outcomes = sorted(
            {
                item
                for concept_id in component
                for item in descriptors[concept_id]["outcomes"]
            }
        )
        families = sorted(
            {
                item
                for concept_id in component
                for item in descriptors[concept_id]["families"]
            }
        )
        predictive_types = {
            "precedes",
            "enables",
            "constrains",
            "shared_outcome",
        }
        predictive_count = sum(
            1
            for relation in component_relations
            if relation[2] in predictive_types
        )
        signature = sha1(
            json.dumps(
                {
                    "concepts": concept_ids,
                    "relations": [
                        [item[0], item[1], item[2]]
                        for item in component_relations
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:20]
        # Stable identity retained from v6.2.1; semantics/version are v6.3.
        node_id = world_model_node_id("v621:" + signature)
        status = (
            "accepted"
            if predictive_count >= 1 and contexts and (outcomes or families)
            else "candidate"
        )
        attrs = {
            "concept_ids": concept_ids,
            "supported_contexts": contexts,
            "predicted_outcomes": outcomes,
            "source_families": families,
            "relation_count": len(component_relations),
            "predictive_relation_count": predictive_count,
            "relation_types": sorted({item[2] for item in component_relations}),
            "explanatory_reach": sum(
                float(descriptors[concept_id]["explanatory_reach"])
                for concept_id in component
            ),
            "promotion_status": status,
            "world_model_version": ABSTRACTION_VERSION,
        }
        self.memory.upsert_node(
            MemoryNode(
                node_id=node_id,
                memory_level="M5",
                node_type="WorldModelFragment",
                canonical_key=signature,
                attrs=attrs,
            ),
            step=step,
        )
        for concept_id in concept_ids:
            self.memory.upsert_edge(
                MemoryEdge(
                    concept_id,
                    node_id,
                    "explains",
                    edge_source=ABSTRACTION_VERSION,
                )
            )
            self.memory.upsert_edge(
                MemoryEdge(
                    node_id,
                    concept_id,
                    "depends_on",
                    edge_source=ABSTRACTION_VERSION,
                )
            )

        self.memory.connection.execute(
            "DELETE FROM world_model_relations_v621 WHERE model_id=?",
            (node_id,),
        )
        for source, target, relation_type, support, confidence, evidence in component_relations:
            relation_id = "wmrel:" + sha1(
                f"{node_id}|{source}|{target}|{relation_type}".encode("utf-8")
            ).hexdigest()[:24]
            self.memory.connection.execute(
                """
                INSERT OR REPLACE INTO world_model_relations_v621(
                    relation_id, model_id,
                    source_concept_id, target_concept_id,
                    relation_type, support_count, confidence,
                    evidence_json, updated_step, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    relation_id,
                    node_id,
                    source,
                    target,
                    relation_type,
                    int(support),
                    float(confidence),
                    json.dumps(evidence, sort_keys=True),
                    step,
                    time.time(),
                ),
            )
        created += 1

    if created:
        for model in self.memory.query_nodes(
            memory_level="M5",
            node_type="WorldModelFragment",
        ):
            attrs = dict(model.get("attrs") or {})
            if attrs.get("world_model_version") == ABSTRACTION_VERSION:
                continue
            attrs["promotion_status"] = "superseded"
            attrs["superseded_reason"] = "replaced_by_v63_relational_world_model"
            self.memory.update_node_support_and_attrs(
                str(model["node_id"]),
                attrs,
                support_increment=0,
                step=step,
            )

    _audit_frontier(
        self.memory.connection,
        frontier_kind="concept_to_world_model",
        candidates_seen=len(all_concepts),
        candidates_retained=len(concepts),
        comparisons_attempted=comparisons,
        comparison_budget=_BUDGET.max_concept_pair_comparisons,
        step=step,
    )
    self.memory.connection.commit()
    return created
