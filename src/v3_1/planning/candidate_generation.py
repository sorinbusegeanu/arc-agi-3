from __future__ import annotations

from v3_1.planning.belief_builder import normalized_route_signature, normalized_target_key, normalized_trigger_zone_key
from v3_1.utils.ids import stable_digest

MAX_SUPPORTING_EVIDENCE_REFS = 12
GENERATION_QUOTAS = {
    "target_interaction": 6,
    "local_probe": 4,
    "frontier_move": 4,
    "route_probe": 3,
    "trigger_probe": 4,
    "recovery_move": 3,
    "fallback_action": 1,
    "unlock_trigger": 3,
    "verify_panel_state": 3,
    "verify_gate_match": 3,
    "unlock_then_exit": 2,
    "trigger_then_target": 3,
    "state_sync_probe": 3,
    "mechanic_test_deterministic": 3,
    "mechanic_chain_deterministic": 3,
    "mechanic_test_llm": 2,
    "mechanic_chain_llm": 2,
}


def _candidate_id(payload: dict) -> str:
    return f"candidate:{stable_digest(payload)}"


def _available_action_families(belief: dict) -> set[str]:
    families = set(belief.get("available_action_families", []) or [])
    return families or {"move"}


def _match_skill_id(skill_library: dict[str, dict], *, entity_id: str, area_id: str | None, objective_type: str, execution_mode: str) -> tuple[str | None, str | None]:
    preferred_skill_types = (
        ("inspect_target", "trigger_probe") if objective_type in {"interact", "test_trigger", "gather_local_info"} and execution_mode in {"interact", "click_at"}
        else ("recover_route",) if objective_type == "recover"
        else ()
    )
    for skill_type in preferred_skill_types:
        for skill_id, skill in skill_library.items():
            if str(skill.get("skill_type") or "") != skill_type:
                continue
            if entity_id and str(skill.get("entity_id") or "") == entity_id:
                return str(skill_id), skill_type
            if area_id is not None and str(skill.get("target_area_id") or "") == str(area_id):
                return str(skill_id), skill_type
    return None, None


def _compress_supporting_evidence_refs(refs: list[str] | tuple[str, ...]) -> tuple[list[str], dict]:
    ordered = [str(ref) for ref in refs if ref]
    unique = list(dict.fromkeys(ordered))
    sample = unique[:MAX_SUPPORTING_EVIDENCE_REFS]
    return sample, {
        "supporting_evidence_ref_count": len(unique),
        "supporting_evidence_ref_sample": sample,
        "supporting_evidence_signature": stable_digest(unique),
        "supporting_evidence_truncated": len(unique) > len(sample),
        "full_supporting_evidence_refs": unique,
    }


def _derived_candidate_class(*, objective_type: str, execution_mode: str, navigation_mode: str) -> str:
    if objective_type == "unlock_trigger":
        return "unlock_trigger"
    if objective_type == "verify_panel_state":
        return "verify_panel_state"
    if objective_type == "verify_gate_match":
        return "verify_gate_match"
    if objective_type == "unlock_then_exit":
        return "unlock_then_exit"
    if objective_type == "trigger_then_target":
        return "trigger_then_target"
    if objective_type == "state_sync_probe":
        return "state_sync_probe"
    if objective_type == "mechanic_test_deterministic":
        return "mechanic_test_deterministic"
    if objective_type == "mechanic_chain_deterministic":
        return "mechanic_chain_deterministic"
    if objective_type == "mechanic_test_llm":
        return "mechanic_test_llm"
    if objective_type == "mechanic_chain_llm":
        return "mechanic_chain_llm"
    if objective_type == "interact" and execution_mode == "interact":
        return "target_interaction"
    if objective_type == "interact" and execution_mode == "click_at":
        return "click_target"
    if objective_type == "gather_local_info":
        return "local_probe"
    if objective_type == "explore_frontier":
        return "frontier_move"
    if objective_type == "test_trigger":
        return "trigger_probe"
    if objective_type == "recover":
        return "recovery_move"
    if objective_type == "probe_route":
        return "route_probe"
    return "fallback_action"


def _candidate_schema(*, target: dict | None, belief: dict, objective_type: str, execution_mode: str, navigation_mode: str, rationale: str, generation_source: str, trigger_zone_id: str | None = None, support_refs: list[str] | None = None, action_overrides: dict | None = None, fallback_baseline_penalty: float = 0.0, supporting_graph_node_ids: list[str] | None = None, supporting_graph_edge_ids: list[str] | None = None, prerequisite_chain: list[str] | None = None, graph_hop_count: int = 0, hypothesized_only_dependency: bool = False) -> dict:
    target = dict(target or {})
    centroid = list(target.get("centroid", [0, 0]))
    target_entity_id = str(target.get("entity_id")) if target.get("entity_id") is not None else None
    target_area_id = str(target.get("area_id")) if target.get("area_id") is not None else belief.get("current_area_id")
    target_entity_class = str(target.get("kind") or target.get("poi_class") or "unknown")
    route_signature = normalized_route_signature(
        objective_type=objective_type,
        navigation_mode=navigation_mode,
        area_id=target_area_id,
        target_entity_id=target_entity_id,
        centroid=centroid,
        action_hint=str(target.get("action_hint")) if target.get("action_hint") is not None else None,
    )
    target_key = normalized_target_key(target_entity_id, target_area_id, target_class=target_entity_class)
    supporting_refs, support_meta = _compress_supporting_evidence_refs(list(support_refs or target.get("evidence_refs", [])))
    skill_id, skill_type = _match_skill_id(
        belief.get("_skill_library", {}),
        entity_id=target_entity_id or "",
        area_id=target_area_id,
        objective_type=objective_type,
        execution_mode=execution_mode,
    )
    candidate_class = _derived_candidate_class(objective_type=objective_type, execution_mode=execution_mode, navigation_mode=navigation_mode)
    stable_key = {
        "objective_type": objective_type,
        "execution_mode": execution_mode,
        "navigation_mode": navigation_mode,
        "target_key": target_key,
        "route_signature": route_signature,
        "trigger_zone_id": trigger_zone_id or "none",
    }
    contradiction_flags = dict(target.get("contradiction_markers", {}))
    stale_support_flags = {
        "support_refs_missing": bool(supporting_refs) and len(supporting_refs) < int(support_meta.get("supporting_evidence_ref_count", 0)),
        "target_stale": contradiction_flags.get("stale_target", False),
    }
    action = {
        "type": execution_mode,
        "target": target_entity_id,
        "centroid": centroid,
        "skill_id": skill_id,
    }
    if action_overrides:
        action.update(action_overrides)
    direct_support = min(1.0, float(target.get("confidence", 0.0)) + (0.1 * len(supporting_refs)))
    indirect_support = min(1.0, float(target.get("utility", 0.0)) + (0.25 if trigger_zone_id else 0.0))
    prior_key = normalized_target_key(target_entity_id, target_area_id, target_class=target_entity_class)
    prior_row = dict(belief.get("durable_prior_view", {}).get("per_target", {}).get(prior_key, {}))
    prior_support = min(1.0, float(prior_row.get("poi_pattern", {}).get("observations", 0) or 0) / 10.0) if prior_row else 0.0
    expected_progress_type = "state_change" if objective_type in {"interact", "test_trigger"} else "evidence_gain" if objective_type in {"gather_local_info", "probe_route"} else "route_progress" if objective_type in {"explore_frontier", "recover"} else "fallback"
    return {
        "candidate_id": _candidate_id(stable_key),
        "candidate_class": candidate_class,
        "objective_type": objective_type,
        "execution_mode": execution_mode,
        "navigation_mode": navigation_mode,
        "target_entity_id": target_entity_id,
        "target_area_id": target_area_id,
        "target_key": target_key,
        "route_signature": route_signature,
        "trigger_zone_id": trigger_zone_id,
        "target_entity_class": target_entity_class,
        "required_action_family": execution_mode,
        "effect_action_family": str(target.get("candidate_effect_mode") or execution_mode),
        "candidate_context": {
            "avatar_area": belief.get("local_context_view", {}).get("current_area_id"),
            "local_area": belief.get("current_area_id"),
            "route_signature": route_signature,
            "trigger_zone_id": trigger_zone_id,
            "target_entity_class": target_entity_class,
        },
        "expected_progress_type": expected_progress_type,
        "expected_outcomes": {
            "expected_state_change": float(target.get("candidate_effect_score", target.get("interact_effect_score", 0.0))),
            "expected_evidence_gain": min(1.0, float(target.get("novelty", 0.0)) + float(target.get("motion_score", 0.0))),
            "expected_route_progress": float(target.get("distance_score", 0.0)) if navigation_mode in {"direct", "routed"} else 0.0,
        },
        "support_strength": {
            "direct_support": direct_support,
            "indirect_support": indirect_support,
            "prior_support": prior_support,
        },
        "contradiction_flags": contradiction_flags,
        "stale_support_flags": stale_support_flags,
        "supporting_evidence_refs": supporting_refs,
        "full_supporting_evidence_refs": list(support_meta["full_supporting_evidence_refs"]),
        "generation_source": generation_source,
        "supporting_graph_node_ids": list(dict.fromkeys(str(value) for value in list(supporting_graph_node_ids or []) if value)),
        "supporting_graph_edge_ids": list(dict.fromkeys(str(value) for value in list(supporting_graph_edge_ids or []) if value)),
        "hop_count": int(graph_hop_count),
        "prerequisite_chain": list(prerequisite_chain or []),
        "depends_on_hypothesized_only_edges": bool(hypothesized_only_dependency),
        "skill_id": skill_id,
        "skill_type": skill_type,
        **{key: value for key, value in support_meta.items() if key != "full_supporting_evidence_refs"},
        "action": action,
        "confidence": float(target.get("confidence", 0.0)),
        "utility": float(target.get("utility", 0.0)),
        "novelty": float(target.get("novelty", 0.0)),
        "movement_effect_score": float(target.get("movement_effect_score", 0.0)),
        "interact_effect_score": float(target.get("interact_effect_score", 0.0)),
        "click_effect_score": float(target.get("click_effect_score", 0.0)),
        "candidate_effect_score": float(target.get("candidate_effect_score", 0.0)),
        "distance_from_avatar": float(target.get("distance_from_avatar", 0.0)),
        "distance_score": float(target.get("distance_score", 0.0)),
        "motion_variance": float(target.get("motion_variance", 0.0)),
        "motion_score": float(target.get("motion_score", 0.0)),
        "reachable_now": bool(target.get("reachable_now")),
        "reachable_later": bool(target.get("reachable_later")),
        "rationale": rationale,
        "route_required": bool(navigation_mode in {"direct", "routed"}),
        "fallback_baseline_penalty": float(fallback_baseline_penalty),
        "blocked_reasons": [],
        "blocked_reason_details": [],
    }


def _seeded_candidate(row: dict, *, seed_contract: str, observed_row_ids: list[str] | None = None, hypothesis_row_ids: list[str] | None = None) -> dict:
    payload = dict(row)
    payload["seed_contract"] = seed_contract
    payload["seed_observed_row_ids"] = list(dict.fromkeys(str(value) for value in list(observed_row_ids or []) if value))
    payload["seed_hypothesis_row_ids"] = list(dict.fromkeys(str(value) for value in list(hypothesis_row_ids or []) if value))
    payload["seed_requires_hypothesis"] = bool(payload["seed_hypothesis_row_ids"] and not payload["seed_observed_row_ids"])
    return payload


def generate_candidates(
    skill_library: dict[str, dict],
    belief: dict,
    limit: int,
    *,
    observed_world: dict | None = None,
    hypothesized_world: dict | None = None,
) -> list[dict]:
    belief = dict(belief)
    belief["_skill_library"] = skill_library
    candidates: list[dict] = []
    diagnostics = {"count_by_class": {}, "dropped_during_generation": 0, "unsupported_template_count": 0}
    class_counts: dict[str, int] = {}
    available_families = _available_action_families(belief)

    def _admit(row: dict | None) -> None:
        if row is None:
            diagnostics["unsupported_template_count"] += 1
            return
        candidate_class = str(row.get("candidate_class") or "unknown")
        quota = GENERATION_QUOTAS.get(candidate_class, limit)
        count = class_counts.get(candidate_class, 0)
        if count >= quota:
            diagnostics["dropped_during_generation"] += 1
            return
        class_counts[candidate_class] = count + 1
        diagnostics["count_by_class"][candidate_class] = class_counts[candidate_class]
        row["generation_diagnostics"] = diagnostics
        candidates.append(row)

    observed_world = dict(observed_world or belief.get("observed_world", {}))
    hypothesized_world = dict(hypothesized_world or belief.get("hypothesized_world", {}))
    observed_seeds = {
        "reachable_targets": list(observed_world.get("reachable_targets", [])),
        "promising_pois": list(observed_world.get("promising_pois", [])),
        "frontier_targets": list(observed_world.get("frontier_targets", [])),
        "trigger_candidates": list(observed_world.get("trigger_candidates", [])),
        "recovery_candidates": list(observed_world.get("recovery_candidates", [])),
    }
    hypothesis_seeds = {
        "reachable_targets": list(hypothesized_world.get("reachable_targets", [])),
        "promising_pois": list(hypothesized_world.get("promising_pois", [])),
        "frontier_targets": list(hypothesized_world.get("frontier_targets", [])),
        "trigger_candidates": list(hypothesized_world.get("trigger_candidates", [])),
        "recovery_candidates": list(hypothesized_world.get("recovery_candidates", [])),
    }

    def _row_id(target: dict) -> str | None:
        return str(target.get("entity_id") or target.get("trigger_id") or target.get("node_id") or target.get("edge_id") or "") or None

    mechanic_paths = list(belief.get("mechanic_graph_paths_to_exit", []) or [])
    mechanic_match_relations = list(belief.get("mechanic_graph_match_relations", []) or [])
    mechanic_trigger_candidates = list(belief.get("mechanic_graph_trigger_candidates", []) or [])
    deterministic_hypotheses = dict(belief.get("deterministic_hypotheses", {}) or {})
    llm_hypotheses = dict(belief.get("llm_hypotheses", {}) or {})

    for row in mechanic_paths:
        node_ids = list(row.get("node_ids", []) or [])
        edge_ids = list(row.get("edge_ids", []) or [])
        if not node_ids:
            continue
        target = {"entity_id": node_ids[0], "area_id": belief.get("current_area_id"), "kind": "poi", "confidence": row.get("support_strength", 0.0), "utility": row.get("support_strength", 0.0)}
        _admit(
            _candidate_schema(
                target=target,
                belief=belief,
                objective_type="unlock_then_exit",
                execution_mode="move",
                navigation_mode="routed",
                rationale="mechanic_path_to_exit",
                generation_source="mechanic_graph.paths_to_exit",
                supporting_graph_node_ids=node_ids,
                supporting_graph_edge_ids=edge_ids,
                graph_hop_count=max(0, len(edge_ids)),
                prerequisite_chain=node_ids,
                hypothesized_only_dependency=bool(row.get("hypothesis_only")),
            )
        )

    for row in mechanic_trigger_candidates:
        trigger_node_id = row.get("trigger_node_id")
        paths = list(row.get("paths_to_exit", []) or [])
        target = {"entity_id": trigger_node_id, "area_id": belief.get("current_area_id"), "kind": "trigger", "confidence": 0.5, "utility": 0.6}
        _admit(
            _candidate_schema(
                target=target,
                belief=belief,
                objective_type="unlock_trigger",
                execution_mode="interact" if "interact" in available_families else "move",
                navigation_mode="routed",
                rationale="mechanic_trigger_candidate",
                generation_source="mechanic_graph.trigger_candidates",
                supporting_graph_node_ids=[trigger_node_id],
                supporting_graph_edge_ids=[edge_id for path in paths for edge_id in list(path.get("edge_ids", []) or [])],
                graph_hop_count=max([len(list(path.get("edge_ids", []) or [])) for path in paths] or [0]),
                prerequisite_chain=[trigger_node_id],
                hypothesized_only_dependency=all(bool(path.get("hypothesis_only")) for path in paths) if paths else False,
            )
        )

    for row in mechanic_match_relations:
        panel_node_id = row.get("panel_node_id")
        edges = list(row.get("match_edges", []) or [])
        if not panel_node_id or not edges:
            continue
        _admit(
            _candidate_schema(
                target={"entity_id": panel_node_id, "area_id": belief.get("current_area_id"), "kind": "panel", "confidence": 0.45, "utility": 0.5},
                belief=belief,
                objective_type="verify_panel_state",
                execution_mode="move",
                navigation_mode="routed",
                rationale="mechanic_panel_match",
                generation_source="mechanic_graph.match_relations",
                supporting_graph_node_ids=[panel_node_id],
                supporting_graph_edge_ids=[str(edge.get("edge_id")) for edge in edges],
                graph_hop_count=1,
                prerequisite_chain=[panel_node_id],
                hypothesized_only_dependency=all(str(edge.get("evidence_tier") or "") != "observed" for edge in edges),
            )
        )

    for proposal in deterministic_hypotheses.values():
        proposal_kind = str(proposal.get("proposal_kind") or "")
        objective_type = "mechanic_test_deterministic" if proposal_kind == "test" else "mechanic_chain_deterministic"
        source_ids = [str(proposal.get("proposal_id") or "")]
        chain = [str(proposal.get("src_node_id") or ""), str(proposal.get("dst_node_id") or "")]
        _admit(
            _candidate_schema(
                target={"entity_id": proposal.get("src_node_id"), "area_id": belief.get("current_area_id"), "kind": "poi", "confidence": proposal.get("confidence", 0.0), "utility": proposal.get("confidence", 0.0)},
                belief=belief,
                objective_type=objective_type,
                execution_mode="move",
                navigation_mode="routed",
                rationale="deterministic_hypothesis_candidate",
                generation_source="hypothesis.deterministic",
                supporting_graph_node_ids=chain,
                supporting_graph_edge_ids=list(proposal.get("expected_edge_ids", []) or []),
                graph_hop_count=max(0, len(chain) - 1),
                prerequisite_chain=chain,
                hypothesized_only_dependency=True,
                action_overrides={
                    "hypothesis_source": "deterministic",
                    "supporting_hypothesis_ids": source_ids,
                    "requires_validation": bool(proposal.get("requires_validation", True)),
                    "expected_information_gain": float(proposal.get("expected_information_gain", 0.4) or 0.4),
                    "chain_length": max(0, len(chain) - 1),
                    "dependency_path_ids": source_ids,
                    "test_proposal_id": str(proposal.get("test_id") or proposal.get("proposal_id") or "") if proposal_kind == "test" else None,
                    "target_node_ids": list(proposal.get("target_node_ids", []) or []),
                    "estimated_cost": float(proposal.get("estimated_cost", 1.0) or 1.0),
                    "discriminates_between_proposal_ids": list(proposal.get("discriminates_between_proposal_ids", []) or []),
                },
            )
        )

    for proposal in llm_hypotheses.values():
        proposal_kind = str(proposal.get("proposal_kind") or "")
        validation_state = str(dict(belief.get("hypothesis_registry_snapshot", {}) or {}).get("validation_state", {}).get(str(proposal.get("proposal_id") or ""), "new"))
        if proposal_kind == "test" and validation_state != "validated":
            continue
        objective_type = "mechanic_test_llm" if proposal_kind == "test" else "mechanic_chain_llm"
        source_ids = [str(proposal.get("proposal_id") or "")]
        chain = [str(proposal.get("src_node_id") or ""), str(proposal.get("dst_node_id") or "")]
        _admit(
            _candidate_schema(
                target={"entity_id": proposal.get("src_node_id"), "area_id": belief.get("current_area_id"), "kind": "poi", "confidence": proposal.get("confidence", 0.0), "utility": proposal.get("confidence", 0.0)},
                belief=belief,
                objective_type=objective_type,
                execution_mode="move",
                navigation_mode="routed",
                rationale="llm_hypothesis_candidate",
                generation_source="hypothesis.llm",
                supporting_graph_node_ids=chain,
                supporting_graph_edge_ids=list(proposal.get("expected_edge_ids", []) or []),
                graph_hop_count=max(0, len(chain) - 1),
                prerequisite_chain=chain,
                hypothesized_only_dependency=True,
                action_overrides={
                    "hypothesis_source": "llm",
                    "supporting_hypothesis_ids": source_ids,
                    "requires_validation": bool(proposal.get("requires_validation", True)),
                    "expected_information_gain": float(proposal.get("expected_information_gain", 0.3) or 0.3),
                    "chain_length": max(0, len(chain) - 1),
                    "dependency_path_ids": source_ids,
                    "test_proposal_id": str(proposal.get("test_id") or proposal.get("proposal_id") or "") if proposal_kind == "test" else None,
                    "target_node_ids": list(proposal.get("target_node_ids", []) or []),
                    "estimated_cost": float(proposal.get("estimated_cost", 1.0) or 1.0),
                    "discriminates_between_proposal_ids": list(proposal.get("discriminates_between_proposal_ids", []) or []),
                },
            )
        )

    for target in observed_seeds["reachable_targets"]:
        if "interact" in available_families:
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="interact", execution_mode="interact", navigation_mode="direct" if target.get("reachable_now") else "routed", rationale="reachable_target", generation_source="observed.reachable_targets"), seed_contract="observed_only", observed_row_ids=[_row_id(target)]))
        if "click_at" in available_families:
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="interact", execution_mode="click_at", navigation_mode="direct" if target.get("reachable_now") else "routed", rationale="reachable_click_target", generation_source="observed.reachable_targets", action_overrides={"click_target_coordinates": list(target.get("centroid", [0, 0]))}), seed_contract="observed_only", observed_row_ids=[_row_id(target)]))

    for target in observed_seeds["promising_pois"]:
        if target.get("area_id") != belief.get("current_area_id"):
            continue
        if "interact" in available_families:
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="gather_local_info", execution_mode="interact", navigation_mode="local", rationale="local_probe", generation_source="observed.promising_pois"), seed_contract="observed_only", observed_row_ids=[_row_id(target)]))

    for target in observed_seeds["frontier_targets"]:
        if "move" in available_families:
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="explore_frontier", execution_mode="move", navigation_mode="routed" if not target.get("reachable_now") else "direct", rationale="frontier_target", generation_source="observed.frontier_targets"), seed_contract="observed_only", observed_row_ids=[_row_id(target)]))

    for target in observed_seeds["trigger_candidates"]:
        if "interact" in available_families:
            trigger_zone_id = normalized_trigger_zone_key(entity_id=str(target.get("entity_id")), area_id=str(target.get("area_id")) if target.get("area_id") is not None else None)
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="test_trigger", execution_mode="interact", navigation_mode="direct" if target.get("reachable_now") else "routed", rationale="trigger_supported", generation_source="observed.trigger_candidates", trigger_zone_id=trigger_zone_id), seed_contract="observed_only", observed_row_ids=[_row_id(target), trigger_zone_id]))

    for target in observed_seeds["recovery_candidates"]:
        if "move" in available_families:
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="recover", execution_mode="move", navigation_mode="routed", rationale="recovery_target", generation_source="observed.recovery_candidates"), seed_contract="observed_only", observed_row_ids=[_row_id(target)]))

    for target in hypothesis_seeds["reachable_targets"]:
        if "interact" in available_families:
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="interact", execution_mode="interact", navigation_mode="direct" if target.get("reachable_now") else "routed", rationale="hypothesis_reachable_target", generation_source="hypothesis.reachable_targets"), seed_contract="hypothesis_backfill", hypothesis_row_ids=[_row_id(target)]))
        if "click_at" in available_families:
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="interact", execution_mode="click_at", navigation_mode="direct" if target.get("reachable_now") else "routed", rationale="hypothesis_reachable_click_target", generation_source="hypothesis.reachable_targets", action_overrides={"click_target_coordinates": list(target.get("centroid", [0, 0]))}), seed_contract="hypothesis_backfill", hypothesis_row_ids=[_row_id(target)]))

    for target in hypothesis_seeds["promising_pois"]:
        if target.get("area_id") != belief.get("current_area_id"):
            continue
        if "interact" in available_families:
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="gather_local_info", execution_mode="interact", navigation_mode="local", rationale="hypothesis_local_probe", generation_source="hypothesis.promising_pois"), seed_contract="hypothesis_backfill", hypothesis_row_ids=[_row_id(target)]))

    for target in hypothesis_seeds["frontier_targets"]:
        if "move" in available_families:
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="explore_frontier", execution_mode="move", navigation_mode="routed" if not target.get("reachable_now") else "direct", rationale="hypothesis_frontier_target", generation_source="hypothesis.frontier_targets"), seed_contract="hypothesis_backfill", hypothesis_row_ids=[_row_id(target)]))

    for target in hypothesis_seeds["trigger_candidates"]:
        if "interact" in available_families:
            trigger_zone_id = normalized_trigger_zone_key(entity_id=str(target.get("entity_id")), area_id=str(target.get("area_id")) if target.get("area_id") is not None else None)
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="test_trigger", execution_mode="interact", navigation_mode="direct" if target.get("reachable_now") else "routed", rationale="hypothesis_trigger_supported", generation_source="hypothesis.trigger_candidates", trigger_zone_id=trigger_zone_id), seed_contract="hypothesis_backfill", hypothesis_row_ids=[_row_id(target), trigger_zone_id]))

    for target in hypothesis_seeds["recovery_candidates"]:
        if "move" in available_families:
            _admit(_seeded_candidate(_candidate_schema(target=target, belief=belief, objective_type="recover", execution_mode="move", navigation_mode="routed", rationale="hypothesis_recovery_target", generation_source="hypothesis.recovery_candidates"), seed_contract="hypothesis_backfill", hypothesis_row_ids=[_row_id(target)]))

    if "move" in available_families:
        observed_consequences = list(observed_world.get("consequences", {}).values()) if isinstance(observed_world.get("consequences"), dict) else list(observed_world.get("consequences", []))
        hypothesized_consequences = list(hypothesized_world.get("consequences", {}).values()) if isinstance(hypothesized_world.get("consequences"), dict) else list(hypothesized_world.get("consequences", []))
        consequence_groups: dict[str, dict[str, list[dict]]] = {}
        for row in observed_consequences:
            action_text = str(row.get("action_name") or row.get("action_key") or "").strip().lower()
            if action_text:
                consequence_groups.setdefault(action_text, {"observed": [], "hypothesized": []})["observed"].append(dict(row))
        for row in hypothesized_consequences:
            action_text = str(row.get("action_name") or row.get("action_key") or "").strip().lower()
            if action_text:
                consequence_groups.setdefault(action_text, {"observed": [], "hypothesized": []})["hypothesized"].append(dict(row))
        for action_text, grouped_rows in consequence_groups.items():
            consequence_rows = list(grouped_rows["observed"]) + list(grouped_rows["hypothesized"])
            if not consequence_rows:
                continue
            observed_ids = [str(row.get("consequence_id")) for row in grouped_rows["observed"] if row.get("consequence_id")]
            hypothesis_ids = [str(row.get("consequence_id")) for row in grouped_rows["hypothesized"] if row.get("consequence_id")]
            seed_mode = "observed_only" if observed_ids and not hypothesis_ids else "hypothesis_backfill" if hypothesis_ids and not observed_ids else "observed_plus_hypothesis"
            route_candidate = _seeded_candidate(_candidate_schema(
                    target={"area_id": belief.get("current_area_id"), "action_hint": action_text, "novelty": 0.1, "utility": min(0.5, 0.1 * len(consequence_rows)), "confidence": 0.25},
                    belief=belief,
                    objective_type="probe_route",
                    execution_mode="move",
                    navigation_mode="routed",
                    rationale="consequence_supported_route_probe",
                    generation_source="observed.consequences" if observed_ids else "hypothesis.consequences",
                    support_refs=[str(row.get("consequence_id")) for row in consequence_rows if row.get("consequence_id")],
                    action_overrides={"action_hint": action_text},
                ), seed_contract="observed_only" if observed_ids else "hypothesis_backfill", observed_row_ids=observed_ids, hypothesis_row_ids=hypothesis_ids)
            route_candidate["route_probe_seed_mode"] = seed_mode
            route_candidate["route_probe_observed_consequence_ids"] = observed_ids
            route_candidate["route_probe_hypothesis_consequence_ids"] = hypothesis_ids
            route_candidate["route_probe_used_compatibility_fallback"] = False
            _admit(route_candidate)

    _admit(
        _seeded_candidate(_candidate_schema(
            target={"area_id": belief.get("current_area_id"), "utility": 0.0, "confidence": 0.0, "novelty": 0.0},
            belief=belief,
            objective_type="fallback",
            execution_mode="move",
            navigation_mode="hold",
            rationale="always_available_fallback",
            generation_source="fallback.template",
            action_overrides={"type": "hold_position", "area_id": belief.get("current_area_id")},
            fallback_baseline_penalty=2.0,
        ), seed_contract="compatibility_fallback", observed_row_ids=[], hypothesis_row_ids=[])
    )

    deduped: dict[tuple[str, str | None, str, str], dict] = {}
    for row in candidates:
        key = (str(row["objective_type"]), row.get("target_key"), str(row.get("execution_mode")), str(row.get("navigation_mode")))
        if key not in deduped or float(row.get("confidence", 0.0)) > float(deduped[key].get("confidence", 0.0)):
            deduped[key] = row

    ranked = sorted(
        deduped.values(),
        key=lambda row: (
            row.get("objective_type") == "fallback",
            not bool(row.get("reachable_now")),
            not bool(row.get("reachable_later")),
            -float(row.get("utility", 0.0)),
            -float(row.get("novelty", 0.0)),
            -float(row.get("confidence", 0.0)),
            row["candidate_id"],
        ),
    )
    return ranked[:limit]
