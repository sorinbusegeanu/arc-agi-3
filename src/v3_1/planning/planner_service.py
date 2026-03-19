from __future__ import annotations

from v3_1.contracts.snapshots import PlanningContext
from v3_1.planning.belief_builder import build_belief
from v3_1.planning.candidate_filters import filter_candidates
from v3_1.planning.candidate_generation import generate_candidates
from v3_1.planning.candidate_scoring import score_candidates
from v3_1.planning.decision import package_decision
from v3_1.planning.fallbacks import fallback_candidates
from v3_1.planning.reranking import rerank_candidates
from v3_1.planning.route_features import compute_route_features
from v3_1.planning.subgoal_chain import SubgoalStep, build_chain
from v3_1.planning.queries import (
    query_best_mechanic_subgoal_chain,
    query_exit_readiness,
    query_panel_match_dependencies,
    query_required_verification_before_exit,
    query_required_preconditions_for_target,
    query_trigger_then_exit_candidates,
    query_unlock_paths_for_exit,
)


MECHANIC_CHAIN_CLASSES = {
    "unlock_then_exit",
    "trigger_then_target",
    "verify_trigger_contact",
    "reobserve_remote_change",
    "verify_panel_state",
    "verify_gate_match",
    "mechanic_chain_deterministic",
    "mechanic_chain_llm",
}

POI_ESCALATION_OBJECTIVES = {
    "verify_trigger_contact",
    "reobserve_remote_change",
    "verify_panel_state",
    "verify_gate_match",
    "trigger_then_target",
    "unlock_then_exit",
}


def _apply_poi_probe_escalation(reranked: list[dict], belief: dict) -> list[dict]:
    followthrough = {
        str(key): dict(value)
        for key, value in dict(belief.get("detector_poi_followthrough", {}) or {}).items()
        if isinstance(value, dict)
    }
    adjusted = []
    for row in list(reranked or []):
        candidate = dict(row)
        target_entity_id = str(candidate.get("target_entity_id") or "")
        poi_state = dict(followthrough.get(target_entity_id, {}) or {})
        stronger_support = bool(
            int(poi_state.get("new_graph_edges", 0) or 0) > 0
            or int(poi_state.get("new_hypothesis_support", 0) or 0) > 0
            or int(poi_state.get("new_verification_candidates", 0) or 0) > 0
            or int(poi_state.get("changed_exit_linked_evidence", 0) or 0) > 0
        )
        escalation_bonus = 0.0
        if str(candidate.get("objective_type") or "") in POI_ESCALATION_OBJECTIVES and stronger_support:
            escalation_bonus = 0.22 + (0.04 * min(3, int(poi_state.get("revisit_count", 0) or 0)))
        if str(candidate.get("candidate_class") or "") == "route_probe" and stronger_support:
            escalation_bonus = -0.3
        candidate["final_score"] = float(candidate.get("final_score", candidate.get("score", 0.0)) or 0.0) + escalation_bonus
        rerank_diag = dict(candidate.get("rerank_diagnostics", {}) or {})
        rerank_diag["planner_service_probe_escalation_bonus"] = escalation_bonus
        candidate["rerank_diagnostics"] = rerank_diag
        adjusted.append(candidate)
    adjusted.sort(
        key=lambda item: (
            -float(item.get("final_score", item.get("score", 0.0)) or 0.0),
            str(item.get("candidate_id") or ""),
        )
    )
    return adjusted


def _build_selected_subgoal_chain(selected: dict | None, *, round_id: int) -> dict | None:
    candidate = dict(selected or {})
    if str(candidate.get("candidate_class") or "") not in MECHANIC_CHAIN_CLASSES:
        return None
    step_rows = []
    for row in list(candidate.get("candidate_step_plan", []) or []):
        if not isinstance(row, dict):
            continue
        try:
            step_rows.append(SubgoalStep(**dict(row)))
        except TypeError:
            continue
    if not step_rows:
        objective_type = str(candidate.get("objective_type") or "")
        step_kind_map = {
            "verify_trigger_contact": "verify_trigger_contact",
            "reobserve_remote_change": "reobserve_region",
            "verify_panel_state": "verify_panel",
            "verify_gate_match": "verify_gate",
            "trigger_then_target": "go_to_trigger",
            "unlock_then_exit": "attempt_exit",
        }
        step_kind = step_kind_map.get(objective_type)
        target_node_id = str(candidate.get("target_entity_id") or candidate.get("target_area_id") or "")
        if step_kind and target_node_id:
            step_rows.append(
                SubgoalStep(
                    step_id=f"step:{candidate.get('candidate_id') or target_node_id}:{step_kind}",
                    step_kind=step_kind,
                    target_node_id=target_node_id,
                    expected_evidence=tuple(str(value) for value in list(candidate.get("candidate_expected_evidence", []) or []) if value),
                    success_conditions=("expected_evidence_seen",),
                    failure_conditions=("missing_expected_evidence",),
                    retry_budget=2 if step_kind in {"go_to_trigger", "reobserve_region"} else 1,
                    retry_count=0,
                    depends_on_step_ids=(),
                    step_status="planned",
                    verification_points=tuple(str(value) for value in list(candidate.get("candidate_verification_points", []) or []) if value),
                    fallback_targets=tuple(str(value) for value in list(candidate.get("candidate_fallback_targets", []) or []) if value),
                )
            )
    if not step_rows:
        return None
    chain = build_chain(
        source_candidate_id=str(candidate.get("candidate_id") or ""),
        source_path_id=str(candidate.get("source_path_id") or "") or None,
        source_hypothesis_ids=list(candidate.get("supporting_hypothesis_ids", []) or []),
        target_exit_id=str(candidate.get("target_exit_id") or "") or None,
        expected_outcome_ids=list(candidate.get("expected_outcome_ids", []) or []),
        fallback_policy=str(candidate.get("fallback_policy") or "replan"),
        created_round_id=int(round_id),
        exit_readiness_score_at_creation=float(candidate.get("exit_readiness_score", 0.0) or 0.0),
        required_verification_steps=list(candidate.get("required_verification_steps", []) or []),
        verification_steps_completed=list(candidate.get("verification_steps_completed", []) or []),
        steps=tuple(step_rows),
    )
    return chain.to_dict()


def _recent_outcomes_from_memory(memory_snapshot: dict) -> list[dict]:
    working_memory = dict(memory_snapshot.get("working_memory", memory_snapshot) or {})
    raw_plan_memory = dict(working_memory.get("plan_memory", {}) or {})
    history = list(raw_plan_memory.get("history", []) or [])
    rows = []
    for row in history[-12:]:
        outcome = dict(dict(row or {}).get("outcome", {}) or {})
        outcome_payload = dict(outcome.get("outcome", {}) or {})
        decision = dict(dict(row or {}).get("decision", {}) or {})
        selected = dict(dict(decision.get("metadata", {}) or {}).get("selected_candidate", {}) or {})
        rows.append(
            {
                **outcome_payload,
                "round_id": row.get("round_id") or decision.get("round_id"),
                "candidate_class": selected.get("candidate_class"),
                "objective_type": selected.get("objective_type"),
            }
        )
    return rows


def _verification_objectives_from_missing(missing: list[str]) -> list[str]:
    ordered = []
    for item in list(missing or []):
        if str(item) == "verify_trigger_contact":
            ordered.append("verify_trigger_contact")
        elif str(item) == "reobserve_remote_change":
            ordered.append("reobserve_remote_change")
        elif str(item) == "verify_panel_or_gate":
            ordered.extend(["verify_panel_state", "verify_gate_match"])
    return list(dict.fromkeys(ordered))


def _apply_exit_readiness_selection_guard(reranked: list[dict], *, planner_trace: dict) -> list[dict]:
    rows = [dict(row) for row in list(reranked or [])]
    if not rows:
        planner_trace["selected_exit_readiness_score"] = None
        planner_trace["selected_missing_prerequisites"] = []
        planner_trace["selected_candidate_blocked_by_low_exit_readiness"] = False
        return rows
    selected = dict(rows[0])
    if str(selected.get("objective_type") or "") != "unlock_then_exit":
        planner_trace["selected_exit_readiness_score"] = float(selected.get("exit_readiness_score", 0.0) or 0.0)
        planner_trace["selected_missing_prerequisites"] = list(selected.get("missing_prerequisite_types", []) or [])
        planner_trace["selected_candidate_blocked_by_low_exit_readiness"] = False
        return rows
    readiness_score = float(selected.get("exit_readiness_score", 0.0) or 0.0)
    stronger_verification = next(
        (
            row for row in rows[1:]
            if str(row.get("objective_type") or "") in {"verify_trigger_contact", "reobserve_remote_change", "verify_panel_state", "verify_gate_match", "trigger_then_target"}
            and float(row.get("final_score", row.get("score", 0.0)) or 0.0) >= float(selected.get("final_score", selected.get("score", 0.0)) or 0.0) - 0.4
        ),
        None,
    )
    if readiness_score < 0.72 and stronger_verification is not None:
        replacement_id = str(stronger_verification.get("candidate_id") or "")
        rows.sort(key=lambda row: 0 if str(row.get("candidate_id") or "") == replacement_id else 1)
        planner_trace["selected_exit_readiness_score"] = readiness_score
        planner_trace["selected_missing_prerequisites"] = list(selected.get("missing_prerequisite_types", []) or [])
        planner_trace["selected_candidate_blocked_by_low_exit_readiness"] = True
        return rows
    planner_trace["selected_exit_readiness_score"] = readiness_score
    planner_trace["selected_missing_prerequisites"] = list(selected.get("missing_prerequisite_types", []) or [])
    planner_trace["selected_candidate_blocked_by_low_exit_readiness"] = False
    return rows


def _active_chain_snapshot(belief: dict) -> tuple[dict | None, dict | None, bool]:
    chain_summary = dict(belief.get("active_chain_summary", {}) or {})
    progress = dict(belief.get("chain_progress_summary", {}) or {})
    active_step = dict(progress.get("current_step", {}) or {})
    should_replan = bool(progress.get("should_replan", False))
    if not chain_summary:
        return None, None, should_replan
    return chain_summary, active_step, should_replan


def _prioritized_blackboard(blackboard_snapshot: dict) -> dict:
    snapshot = dict(blackboard_snapshot or {})
    observed_entities = dict(snapshot.get("observed_entities", {}))
    hypothesized_entities = dict(snapshot.get("hypothesized_entities", {}))
    observed_consequences = dict(snapshot.get("observed_consequences", {}))
    hypothesized_consequences = dict(snapshot.get("hypothesized_consequences", {}))
    observed_trigger_zones = dict(snapshot.get("observed_trigger_zones", {}))
    hypothesized_trigger_zones = dict(snapshot.get("hypothesized_trigger_zones", {}))
    observed_topology = dict(snapshot.get("observed_topology", {}))
    hypothesized_topology = dict(snapshot.get("hypothesized_topology", {}))
    snapshot["entities"] = {**hypothesized_entities, **observed_entities}
    snapshot["consequences"] = {**hypothesized_consequences, **observed_consequences}
    snapshot["trigger_zones"] = {**hypothesized_trigger_zones, **observed_trigger_zones}
    snapshot["topology_nodes"] = {
        **dict(hypothesized_topology.get("nodes", {})),
        **dict(observed_topology.get("nodes", {})),
    }
    snapshot["topology_edges"] = {
        **dict(hypothesized_topology.get("edges", {})),
        **dict(observed_topology.get("edges", {})),
    }
    snapshot["_planning_priority"] = {
        "entities": {"observed_count": len(observed_entities), "hypothesized_count": len(hypothesized_entities)},
        "consequences": {"observed_count": len(observed_consequences), "hypothesized_count": len(hypothesized_consequences)},
        "trigger_zones": {"observed_count": len(observed_trigger_zones), "hypothesized_count": len(hypothesized_trigger_zones)},
        "topology": {
            "observed_node_count": len(dict(observed_topology.get("nodes", {}))),
            "observed_edge_count": len(dict(observed_topology.get("edges", {}))),
            "hypothesized_node_count": len(dict(hypothesized_topology.get("nodes", {}))),
            "hypothesized_edge_count": len(dict(hypothesized_topology.get("edges", {}))),
        },
    }
    return snapshot


def _seed_support(candidate: dict, blackboard_snapshot: dict) -> tuple[str, int, bool]:
    observed_entities = dict(blackboard_snapshot.get("observed_entities", {}))
    hypothesized_entities = dict(blackboard_snapshot.get("hypothesized_entities", {}))
    observed_trigger_zones = dict(blackboard_snapshot.get("observed_trigger_zones", {}))
    hypothesized_trigger_zones = dict(blackboard_snapshot.get("hypothesized_trigger_zones", {}))
    observed_topology = dict(blackboard_snapshot.get("observed_topology", {}))
    hypothesized_topology = dict(blackboard_snapshot.get("hypothesized_topology", {}))
    target_entity_id = str(candidate.get("target_entity_id") or "")
    target_area_id = str(candidate.get("target_area_id") or "")
    trigger_zone_id = str(candidate.get("trigger_zone_id") or "")
    route_signature = str(candidate.get("route_signature") or "")

    observed_count = 0
    hypothesized_count = 0
    if target_entity_id:
        observed_count += 1 if target_entity_id in observed_entities else 0
        hypothesized_count += 1 if target_entity_id in hypothesized_entities else 0
    if target_area_id:
        observed_count += sum(1 for row in observed_entities.values() if str(row.get("area_id") or "") == target_area_id)
        hypothesized_count += sum(1 for row in hypothesized_entities.values() if str(row.get("area_id") or "") == target_area_id)
    if trigger_zone_id:
        observed_count += 1 if trigger_zone_id in observed_trigger_zones else 0
        hypothesized_count += 1 if trigger_zone_id in hypothesized_trigger_zones else 0
    if route_signature:
        observed_count += sum(1 for row in dict(observed_topology.get("edges", {})).values() if str(row.get("route_signature") or row.get("edge_id") or "") == route_signature)
        hypothesized_count += sum(1 for row in dict(hypothesized_topology.get("edges", {})).values() if str(row.get("route_signature") or row.get("edge_id") or "") == route_signature)
    if observed_count > 0:
        return "observed", observed_count + hypothesized_count, False
    if hypothesized_count > 0:
        return "hypothesized", hypothesized_count, True
    return "unknown", 0, False


def _annotate_seed_support(rows: list[dict], blackboard_snapshot: dict, *, penalty: float = 0.0) -> list[dict]:
    annotated = []
    for row in rows:
        payload = dict(row)
        seed_tier, seed_count, hypothesis_fallback = _seed_support(payload, blackboard_snapshot)
        payload["seed_evidence_tier"] = seed_tier
        payload["seed_source_count"] = int(seed_count)
        payload["seed_is_fallback_from_hypothesis"] = bool(hypothesis_fallback)
        if penalty and hypothesis_fallback:
            payload["score"] = float(payload.get("score", 0.0) or 0.0) - penalty
            breakdown = dict(payload.get("score_breakdown", {}))
            breakdown["hypothesized_seed_penalty"] = float(penalty)
            payload["score_breakdown"] = breakdown
        annotated.append(payload)
    return annotated


def _split_world_contracts(*, blackboard_snapshot: dict, belief: dict) -> tuple[dict, dict, dict]:
    seed_sets = dict(belief.get("candidate_seed_sets", {}))
    observed_entities = dict(blackboard_snapshot.get("observed_entities", {}))
    hypothesized_entities = dict(blackboard_snapshot.get("hypothesized_entities", {}))
    observed_triggers = dict(blackboard_snapshot.get("observed_trigger_zones", {}))
    hypothesized_triggers = dict(blackboard_snapshot.get("hypothesized_trigger_zones", {}))
    observed_consequences = dict(blackboard_snapshot.get("observed_consequences", {}))
    hypothesized_consequences = dict(blackboard_snapshot.get("hypothesized_consequences", {}))
    observed_topology = dict(blackboard_snapshot.get("observed_topology", {}))
    hypothesized_topology = dict(blackboard_snapshot.get("hypothesized_topology", {}))

    def _filter_rows(rows: list[dict], *, tier: str) -> list[dict]:
        target_store = observed_entities if tier == "observed" else hypothesized_entities
        filtered = []
        for row in list(rows or []):
            entity_id = str(row.get("entity_id") or "")
            if entity_id and entity_id in target_store:
                filtered.append(dict(row))
                continue
            if str(row.get("evidence_tier") or "") == tier:
                filtered.append(dict(row))
        return filtered

    observed_world = {
        "reachable_targets": _filter_rows(seed_sets.get("reachable_targets", []), tier="observed"),
        "frontier_targets": _filter_rows(seed_sets.get("frontier_targets", []), tier="observed"),
        "blocked_targets": _filter_rows(seed_sets.get("blocked_targets", []), tier="observed"),
        "promising_pois": _filter_rows(seed_sets.get("promising_pois", []), tier="observed"),
        "approachable_pois": _filter_rows(seed_sets.get("approachable_pois", []), tier="observed"),
        "trigger_candidates": [dict(row) for row in seed_sets.get("trigger_candidates", []) if str(row.get("entity_id") or "") in observed_triggers or str(row.get("evidence_tier") or "") == "observed"],
        "recovery_candidates": _filter_rows(seed_sets.get("recovery_candidates", []), tier="observed"),
        "entities": observed_entities,
        "consequences": observed_consequences,
        "trigger_zones": observed_triggers,
        "topology": observed_topology,
    }
    hypothesized_world = {
        "reachable_targets": _filter_rows(seed_sets.get("reachable_targets", []), tier="hypothesized"),
        "frontier_targets": _filter_rows(seed_sets.get("frontier_targets", []), tier="hypothesized"),
        "blocked_targets": _filter_rows(seed_sets.get("blocked_targets", []), tier="hypothesized"),
        "promising_pois": _filter_rows(seed_sets.get("promising_pois", []), tier="hypothesized"),
        "approachable_pois": _filter_rows(seed_sets.get("approachable_pois", []), tier="hypothesized"),
        "trigger_candidates": [dict(row) for row in seed_sets.get("trigger_candidates", []) if str(row.get("entity_id") or "") in hypothesized_triggers or str(row.get("evidence_tier") or "") == "hypothesized"],
        "recovery_candidates": _filter_rows(seed_sets.get("recovery_candidates", []), tier="hypothesized"),
        "entities": hypothesized_entities,
        "consequences": hypothesized_consequences,
        "trigger_zones": hypothesized_triggers,
        "topology": hypothesized_topology,
    }
    uncertainty_context = {
        "current_area_id": belief.get("current_area_id"),
        "planning_mode": belief.get("planning_mode"),
        "available_action_families": list(belief.get("available_action_families", [])),
        "versions": dict(belief.get("versions", {})),
        "tactical_memory": dict(belief.get("tactical_memory_view", {})),
        "evidence_index": dict(dict(belief.get("support_view", {})).get("indexes", {}).get("evidence_index", {})),
        "compatibility_alias_rows": {
            "reachable_targets": len(list(seed_sets.get("reachable_targets", []))),
            "frontier_targets": len(list(seed_sets.get("frontier_targets", []))),
            "promising_pois": len(list(seed_sets.get("promising_pois", []))),
            "approachable_pois": len(list(seed_sets.get("approachable_pois", []))),
        },
    }
    return observed_world, hypothesized_world, uncertainty_context


def _sample_rows(rows: list[dict], *, limit: int = 3, keys: tuple[str, ...] = ("entity_id", "target_key", "target_area_id", "candidate_effect_mode", "utility", "confidence", "returned_by_area_local_pois", "reachable_status", "approachable_status", "rejected_from_promising_reason", "mode_gate_reason", "score_if_considered", "displaced_by_frontier_entity_id")) -> list[dict]:
    sampled = []
    for row in list(rows or [])[:limit]:
        payload = dict(row)
        sampled.append({key: payload.get(key) for key in keys if key in payload})
    return sampled


def _compact_belief_trace(belief: dict) -> dict:
    world_view = dict(belief.get("world_view", {}))
    candidate_seed_sets = dict(belief.get("candidate_seed_sets", {}))
    tactical_memory = dict(belief.get("tactical_memory_view", {}))
    support_view = dict(belief.get("support_view", {}))
    durable_prior_view = dict(belief.get("durable_prior_view", {}))
    indexes = dict(support_view.get("indexes", {}))
    return {
        "versions": dict(belief.get("versions", {})),
        "current_area_id": belief.get("current_area_id"),
        "planning_mode": belief.get("planning_mode"),
        "structure_recall_gap": belief.get("structure_recall_gap"),
        "object_backed_node_count": belief.get("object_backed_node_count"),
        "mechanic_object_backed_node_count": belief.get("mechanic_object_backed_node_count"),
        "structure_candidate_count": belief.get("structure_candidate_count"),
        "available_action_families": list(belief.get("available_action_families", [])),
        "world_view": {
            "reachable_count": len(world_view.get("reachable_targets", [])),
            "blocked_count": len(world_view.get("blocked_targets", [])),
            "frontier_count": len(world_view.get("frontier_targets", [])),
            "local_poi_count": len(world_view.get("local_pois", [])),
            "topology_reachable_node_count": int(dict(world_view.get("topology", {})).get("reachable_node_count", 0)),
            "topology": dict(world_view.get("topology", {})),
        },
        "candidate_seed_sets": {
            "promising_poi_count": len(candidate_seed_sets.get("promising_pois", [])),
            "approachable_poi_count": len(candidate_seed_sets.get("approachable_pois", [])),
            "trigger_candidate_count": len(candidate_seed_sets.get("trigger_candidates", [])),
            "recovery_candidate_count": len(candidate_seed_sets.get("recovery_candidates", [])),
            "frontier_target_count": len(candidate_seed_sets.get("frontier_targets", [])),
            "blocked_target_count": len(candidate_seed_sets.get("blocked_targets", [])),
            "reachable_target_count": len(candidate_seed_sets.get("reachable_targets", [])),
            "promising_pois_sample": _sample_rows(candidate_seed_sets.get("promising_pois", [])),
            "approachable_pois_sample": _sample_rows(candidate_seed_sets.get("approachable_pois", [])),
            "trigger_candidates_sample": _sample_rows(candidate_seed_sets.get("trigger_candidates", [])),
            "recovery_candidates_sample": _sample_rows(candidate_seed_sets.get("recovery_candidates", [])),
        },
        "local_context_view": dict(belief.get("local_context_view", {})),
        "tactical_memory_view": {
            "cooldown_count": len(tactical_memory.get("cooldowns", {})),
            "retry_count": len(tactical_memory.get("retries", {})),
            "exhausted_count": len(tactical_memory.get("exhausted", [])),
            "exhausted_key_count": len(tactical_memory.get("exhausted_keys", [])),
            "failed_candidate_count": len(tactical_memory.get("failed_candidates", {})),
            "tactical_context": dict(tactical_memory.get("tactical_context", {})),
        },
        "support_view": {
            "consequence_action_count": len(support_view.get("consequence_support", {})),
            "trigger_support_entity_count": len(support_view.get("trigger_support", {})),
            "evidence_index_count": len(indexes.get("evidence_index", {})),
            "consequence_by_action_count": len(indexes.get("consequence_by_action", {})),
            "consequence_support": dict(support_view.get("consequence_support", {})),
            "trigger_support": dict(support_view.get("trigger_support", {})),
        },
        "durable_prior_view": {
            "version": durable_prior_view.get("version"),
            "per_target_count": len(durable_prior_view.get("per_target", {})),
            "per_poi_class_count": len(durable_prior_view.get("per_poi_class", {})),
            "per_trigger_type_count": len(durable_prior_view.get("per_trigger_type", {})),
            "candidate_outcome_count": len(durable_prior_view.get("candidate_outcomes", {})),
            "poi_pattern_count": len(durable_prior_view.get("poi_patterns", {})),
            "trigger_pattern_count": len(durable_prior_view.get("trigger_patterns", {})),
            "recovery_pattern_count": len(durable_prior_view.get("recovery_patterns", {})),
            "consequence_pattern_count": len(durable_prior_view.get("consequence_patterns", {})),
        },
    }


def _trace_payload(level: str, *, belief: dict, generated: list[dict], survivors: list[dict], blocked_candidates: list[dict], route_features: dict[str, dict], scored: list[dict], selected: dict | None, consistency_checks: dict) -> dict:
    base = {
        "selected_candidate": selected,
        "summary_metrics": {
            "candidates_generated_by_class": dict(generated[0].get("generation_diagnostics", {}).get("count_by_class", {})) if generated else {},
            "filtered_by_reason": dict(blocked_candidates[0].get("filter_audit", {}).get("block_counts_by_reason", {})) if blocked_candidates else {},
            "selected_by_class": str(selected.get("candidate_class")) if selected else None,
            "score_term_usage": sorted({key for row in scored for key in dict(row.get("score_breakdown", {})).keys()}),
            "contradiction_block_count": sum(1 for row in blocked_candidates if "hard.evidence.contradicted" in list(row.get("blocked_reasons", []))),
            "local_repeat_block_count": sum(1 for row in blocked_candidates if any(reason.startswith("soft.repeat.") for reason in list(row.get("soft_filter_reasons", [])) + list(row.get("blocked_reasons", [])))),
        },
        "debug_exports": {
            "promising_pois": _sample_rows(dict(belief.get("candidate_seed_sets", {})).get("promising_pois", [])),
            "trigger_candidates": _sample_rows(dict(belief.get("candidate_seed_sets", {})).get("trigger_candidates", [])),
            "recovery_candidates": _sample_rows(dict(belief.get("candidate_seed_sets", {})).get("recovery_candidates", [])),
            "local_context": belief.get("local_context_view", {}),
            "blocked_targets": _sample_rows(dict(belief.get("world_view", {})).get("blocked_targets", [])),
        },
        "consistency_checks": consistency_checks,
    }
    if level == "minimal":
        return base
    if level == "debug":
        return {
            **base,
            "belief": _compact_belief_trace(belief),
            "generated_candidates": generated,
            "filtered_candidates": {"survivors": survivors, "blocked": blocked_candidates},
            "route_features": route_features,
            "score_breakdown": {str(row.get("candidate_id")): dict(row.get("score_breakdown", {})) for row in scored},
        }
    return {
        **base,
        "belief": _compact_belief_trace(belief),
        "generated_candidates": generated,
        "filtered_candidates": {"survivors": survivors, "blocked": blocked_candidates},
        "route_features": route_features,
        "score_breakdown": {str(row.get("candidate_id")): dict(row.get("score_breakdown", {})) for row in scored},
    }


def plan(
    context: PlanningContext,
    blackboard_snapshot: dict,
    memory_snapshot: dict,
    planning_cfg,
    helper_results: list[dict] | None = None,
    mechanic_graph_snapshot: dict | None = None,
    deterministic_hypotheses: dict | None = None,
    llm_hypotheses: dict | None = None,
    hypothesis_registry_snapshot: dict | None = None,
):
    helper_results = helper_results or []
    planning_blackboard = _prioritized_blackboard(blackboard_snapshot)
    belief = build_belief(planning_blackboard, memory_snapshot, mechanic_graph_snapshot)
    belief["planning_input_priority"] = dict(planning_blackboard.get("_planning_priority", {}))
    belief["planning_observed_state"] = {
        "entities": dict(blackboard_snapshot.get("observed_entities", {})),
        "consequences": dict(blackboard_snapshot.get("observed_consequences", {})),
        "trigger_zones": dict(blackboard_snapshot.get("observed_trigger_zones", {})),
        "topology": dict(blackboard_snapshot.get("observed_topology", {})),
    }
    belief["planning_hypothesized_backfill"] = {
        "entities": dict(blackboard_snapshot.get("hypothesized_entities", {})),
        "consequences": dict(blackboard_snapshot.get("hypothesized_consequences", {})),
        "trigger_zones": dict(blackboard_snapshot.get("hypothesized_trigger_zones", {})),
        "topology": dict(blackboard_snapshot.get("hypothesized_topology", {})),
    }
    observed_world, hypothesized_world, uncertainty_context = _split_world_contracts(blackboard_snapshot=blackboard_snapshot, belief=belief)
    durable_prior_context = dict(belief.get("durable_prior_view", {}))
    belief["observed_world"] = observed_world
    belief["hypothesized_world"] = hypothesized_world
    belief["uncertainty_context"] = uncertainty_context
    belief["durable_prior_context"] = durable_prior_context
    belief["planner_contract_mode"] = "split_world_native"
    mechanic_graph_state = dict((mechanic_graph_snapshot or {}).get("state", mechanic_graph_snapshot or {}))
    recent_outcomes = _recent_outcomes_from_memory(memory_snapshot)
    graph_paths_to_exit = [path.__dict__ for path in query_best_mechanic_subgoal_chain(mechanic_graph_state).paths]
    trigger_nodes = list(dict(belief.get("mechanic_graph_view", {})).get("trigger_candidates", []))
    panel_nodes = [row for row in list(dict(mechanic_graph_state).get("nodes_by_id", {}).values()) if str(row.get("node_kind") or "") == "panel"]
    exit_nodes = [row for row in list(dict(mechanic_graph_state).get("nodes_by_id", {}).values()) if str(row.get("node_kind") or "") == "exit"]
    belief["mechanic_graph_paths_to_exit"] = graph_paths_to_exit
    belief["mechanic_graph_trigger_candidates"] = [
        {
            "trigger_node_id": row.get("node_id"),
            "paths_to_exit": [path.__dict__ for path in query_trigger_then_exit_candidates(mechanic_graph_state, str(row.get("node_id"))).paths],
        }
        for row in trigger_nodes
    ]
    belief["mechanic_graph_gate_conditions"] = [
        {
            "exit_node_id": row.get("node_id"),
            "prerequisite_paths": [path.__dict__ for path in query_unlock_paths_for_exit(mechanic_graph_state, str(row.get("node_id"))).paths],
            "exit_readiness": query_exit_readiness(
                mechanic_graph_state,
                str(row.get("node_id")),
                hypothesis_registry_snapshot=hypothesis_registry_snapshot,
                recent_outcomes=recent_outcomes,
            ),
        }
        for row in exit_nodes
    ]
    belief["mechanic_graph_match_relations"] = [
        {
            "panel_node_id": row.get("node_id"),
            "match_edges": [edge for edge in query_panel_match_dependencies(mechanic_graph_state, str(row.get("node_id"))).edges],
        }
        for row in panel_nodes
    ]
    belief["mechanic_graph_prerequisite_paths"] = [
        {
            "target_node_id": row.get("node_id"),
            "paths": [path.__dict__ for path in query_required_preconditions_for_target(mechanic_graph_state, str(row.get("node_id"))).paths],
            "required_verification": query_required_verification_before_exit(
                mechanic_graph_state,
                str(row.get("node_id")),
                hypothesis_registry_snapshot=hypothesis_registry_snapshot,
                recent_outcomes=recent_outcomes,
            ),
        }
        for row in exit_nodes
    ]
    belief["mechanic_graph_node_lookup"] = {str(node_id): dict(row) for node_id, row in dict(mechanic_graph_state.get("nodes_by_id", {})).items()}
    exit_readiness_by_exit = {
        str(row.get("node_id")): query_exit_readiness(
            mechanic_graph_state,
            str(row.get("node_id")),
            hypothesis_registry_snapshot=hypothesis_registry_snapshot,
            recent_outcomes=recent_outcomes,
        )
        for row in exit_nodes
    }
    belief["exit_readiness_by_exit"] = exit_readiness_by_exit
    belief["best_exit_readiness"] = max(
        list(exit_readiness_by_exit.values()),
        key=lambda row: float(dict(row).get("readiness_score", 0.0) or 0.0),
        default={},
    )
    belief["recent_exit_outcomes"] = recent_outcomes
    belief["deterministic_hypotheses"] = dict(deterministic_hypotheses or {})
    belief["llm_hypotheses"] = dict(llm_hypotheses or {})
    belief["hypothesis_registry_snapshot"] = dict(hypothesis_registry_snapshot or {})
    active_chain, active_step, chain_should_replan = _active_chain_snapshot(belief)
    generated = generate_candidates(
        memory_snapshot.get("skill_library", {}),
        belief,
        planning_cfg.max_candidates,
        observed_world=observed_world,
        hypothesized_world=hypothesized_world,
    )
    generated = _annotate_seed_support(generated, blackboard_snapshot)
    survivors, blocked_candidates = filter_candidates(
        generated,
        observed_world=observed_world,
        hypothesized_world=hypothesized_world,
        uncertainty_context=uncertainty_context,
        belief_fallback=None,
    )
    blocked_candidates = _annotate_seed_support(blocked_candidates, blackboard_snapshot)
    route_features = compute_route_features(planning_blackboard, survivors)
    scored = score_candidates(
        survivors,
        route_features,
        planning_cfg,
        observed_world=observed_world,
        hypothesized_world=hypothesized_world,
        uncertainty_context=uncertainty_context,
        durable_prior_context=durable_prior_context,
        belief_fallback=belief,
    )
    scored = _annotate_seed_support(scored, blackboard_snapshot, penalty=0.2)
    reranked = rerank_candidates(
        scored,
        helper_results,
        observed_world=observed_world,
        hypothesized_world=hypothesized_world,
        uncertainty_context=uncertainty_context,
        durable_prior_context=durable_prior_context,
        belief_fallback=belief,
    )
    reranked = _apply_poi_probe_escalation(reranked, belief)
    planner_trace_stub: dict = {}
    reranked = _apply_exit_readiness_selection_guard(reranked, planner_trace=planner_trace_stub)
    reranked = _annotate_seed_support(reranked, blackboard_snapshot)
    fallback_rows = (
        [row for row in reranked if str(row.get("objective_type") or "") == "fallback"]
        if reranked
        else fallback_candidates(generated, blocked_candidates, belief)
    )
    selected = reranked[0] if reranked else (fallback_rows[0] if fallback_rows else None)
    selected_subgoal_chain = None
    if active_chain and not chain_should_replan:
        selected_subgoal_chain = dict(active_chain)
    elif selected is not None:
        selected_subgoal_chain = _build_selected_subgoal_chain(selected, round_id=context.round_id)
    selected_subgoal_step = dict(active_step or {})
    if selected_subgoal_chain and not selected_subgoal_step:
        chain_steps = list(dict(selected_subgoal_chain).get("steps", []) or [])
        step_index = int(dict(selected_subgoal_chain).get("current_step_index", 0) or 0)
        if 0 <= step_index < len(chain_steps):
            selected_subgoal_step = dict(chain_steps[step_index] or {})
    consistency_checks = {
        "selected_candidate_not_blocked": bool(selected is None or not bool(selected.get("blocked"))),
        "selected_candidate_supported_by_current_belief": bool(selected is None or not list(selected.get("full_supporting_evidence_refs", [])) or any(str(ref) in dict(blackboard_snapshot.get("indexes", {}).get("evidence_index", {})) for ref in list(selected.get("full_supporting_evidence_refs", [])))),
        "selected_candidate_action_family_executable": bool(selected is None or str(selected.get("execution_mode") or "move") in set(belief.get("available_action_families", [])) or str(selected.get("execution_mode") or "move") == "move"),
    }
    helper_summary = {
        "remote_success_count": sum(1 for result in helper_results if str(dict(result.get("metadata", {})).get("execution_path", "")) == "remote"),
        "local_fallback_count": sum(1 for result in helper_results if str(dict(result.get("metadata", {})).get("execution_path", "")) == "local_fallback"),
        "helper_latency_ms": {str(result.get("helper_mode")): float(dict(result.get("metadata", {})).get("latency_ms", 0.0)) for result in helper_results},
        "helper_contribution_rate": {str(result.get("helper_mode")): float(dict(result.get("metadata", {})).get("contribution_rate", 0.0)) for result in helper_results},
    }
    planner_trace = _trace_payload(
        str(getattr(planning_cfg, "trace_level", "debug")),
        belief=belief,
        generated=generated,
        survivors=survivors,
        blocked_candidates=blocked_candidates,
        route_features=route_features,
        scored=reranked,
        selected=selected,
        consistency_checks=consistency_checks,
    )
    planner_trace["planning_mode"] = belief.get("planning_mode")
    planner_trace.update(planner_trace_stub)
    planner_trace["planner_contract_mode"] = "split_world_native"
    pipeline_modes = {
        "generation": all(str(row.get("seed_contract") or "") != "compatibility_fallback" for row in generated),
        "filtering": all(str(row.get("filter_input_mode") or "") == "split_world_native" for row in survivors + blocked_candidates),
        "scoring": all(str(row.get("score_contract_mode") or "") == "split_world_native" for row in scored),
        "reranking": all(str(dict(row.get("rerank_diagnostics", {})).get("rerank_contract_mode") or "") == "split_world_native" for row in reranked),
    }
    planner_trace["planning_pipeline_contract_mode"] = "split_world_native_full" if all(pipeline_modes.values()) else "split_world_native_partial"
    planner_trace["durable_prior_context"] = {
        "version": durable_prior_context.get("version"),
        "per_target_count": len(dict(durable_prior_context.get("per_target", {}))),
        "candidate_outcome_count": len(dict(durable_prior_context.get("candidate_outcomes", {}))),
    }
    planner_trace["mechanic_graph_summary"] = {
        "mechanic_graph_version": context.mechanic_graph_version,
        "path_to_exit_count": len(graph_paths_to_exit),
        "trigger_candidate_count": len(trigger_nodes),
        "panel_match_relation_count": sum(len(list(row.get("match_edges", []))) for row in belief.get("mechanic_graph_match_relations", [])),
    }
    planner_trace["selected_subgoal_chain"] = selected_subgoal_chain
    planner_trace["selected_subgoal_chain_id"] = str(dict(selected_subgoal_chain or {}).get("chain_id") or "") or None
    planner_trace["selected_subgoal_chain_status"] = str(dict(selected_subgoal_chain or {}).get("status") or ("planned" if selected_subgoal_chain else "")) or None
    planner_trace["selected_subgoal_step_id"] = str(dict(selected_subgoal_step or {}).get("step_id") or "") or None
    planner_trace["selected_subgoal_step_kind"] = str(dict(selected_subgoal_step or {}).get("step_kind") or "") or None
    planner_trace["helper_summary"] = helper_summary
    if str(getattr(planning_cfg, "trace_level", "debug")) != "minimal":
        planner_trace["helper_results"] = helper_results
    decision = package_decision(
        context=context,
        selected=selected,
        ranked_candidates=reranked,
        fallback_candidates=fallback_rows,
        blocked_candidates=blocked_candidates,
        helper_results=helper_results,
        belief=belief,
        planner_trace=planner_trace,
    )
    metadata = dict(getattr(decision, "metadata", {}) or {})
    metadata["selected_subgoal_chain"] = selected_subgoal_chain
    metadata["selected_subgoal_step"] = selected_subgoal_step
    metadata["selected_subgoal_chain_id"] = str(dict(selected_subgoal_chain or {}).get("chain_id") or "") or None
    metadata["selected_subgoal_chain_status"] = str(dict(selected_subgoal_chain or {}).get("status") or ("planned" if selected_subgoal_chain else "")) or None
    metadata["selected_subgoal_step_id"] = str(dict(selected_subgoal_step or {}).get("step_id") or "") or None
    metadata["selected_subgoal_step_kind"] = str(dict(selected_subgoal_step or {}).get("step_kind") or "") or None
    return type(decision)(**{**decision.__dict__, "metadata": metadata})
