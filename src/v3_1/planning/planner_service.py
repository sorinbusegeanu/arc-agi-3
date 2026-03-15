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


def plan(context: PlanningContext, blackboard_snapshot: dict, memory_snapshot: dict, planning_cfg, helper_results: list[dict] | None = None):
    helper_results = helper_results or []
    belief = build_belief(blackboard_snapshot, memory_snapshot)
    generated = generate_candidates(memory_snapshot.get("skill_library", {}), belief, planning_cfg.max_candidates)
    survivors, blocked_candidates = filter_candidates(generated, belief)
    route_features = compute_route_features(blackboard_snapshot, survivors)
    scored = score_candidates(survivors, belief, route_features, planning_cfg)
    reranked = rerank_candidates(scored, helper_results, belief)
    fallback_set = fallback_candidates(reranked, blocked_candidates, belief)
    selected = reranked[0] if reranked else (fallback_set[0] if fallback_set else None)
    consistency_checks = {
        "selected_candidate_not_blocked": bool(selected is None or not bool(selected.get("blocked"))),
        "selected_candidate_supported_by_current_belief": bool(
            selected is None
            or not list(selected.get("supporting_evidence_refs", []))
            or any(
                str(ref) in dict(blackboard_snapshot.get("indexes", {}).get("evidence_index", {}))
                for ref in list(selected.get("supporting_evidence_refs", []))
            )
        ),
        "selected_candidate_action_family_executable": bool(
            selected is None
            or str(selected.get("required_action_family") or "move") in set(belief.get("available_action_families", []))
            or str(selected.get("required_action_family") or "move") == "move"
        ),
    }
    planner_trace = {
        "belief": belief,
        "generated_candidates": generated,
        "filtered_candidates": {
            "survivors": survivors,
            "blocked": blocked_candidates,
        },
        "route_features": route_features,
        "score_breakdown": {str(row.get("candidate_id")): dict(row.get("score_breakdown", {})) for row in scored},
        "selected_candidate": selected,
        "summary_metrics": {
            "candidates_generated_by_class": dict(generated[0].get("generation_diagnostics", {}).get("count_by_class", {})) if generated else {},
            "filtered_by_reason": dict(blocked_candidates[0].get("filter_audit", {}).get("block_counts_by_reason", {})) if blocked_candidates else {},
            "selected_by_class": str(selected.get("candidate_class")) if selected else None,
            "score_term_usage": sorted({key for row in scored for key in dict(row.get("score_breakdown", {})).keys()}),
            "contradiction_block_count": sum(1 for row in blocked_candidates if "hard_contradiction_current_evidence" in list(row.get("blocked_reasons", []))),
            "local_repeat_block_count": sum(
                1
                for row in blocked_candidates
                if any(reason in {"soft_local_class_repeat", "soft_local_target_repeat", "soft_route_repeat", "soft_trigger_repeat"} for reason in list(row.get("soft_filter_reasons", [])) + list(row.get("blocked_reasons", [])))
            ),
        },
        "debug_exports": {
            "promising_pois": belief.get("promising_pois", []),
            "trigger_candidates": belief.get("trigger_candidates", []),
            "recovery_candidates": belief.get("recovery_candidates", []),
            "local_context": belief.get("local_context", {}),
            "blocked_targets": belief.get("blocked_targets", []),
        },
        "consistency_checks": consistency_checks,
    }
    return package_decision(
        context=context,
        selected=selected,
        ranked_candidates=reranked,
        fallback_candidates=fallback_set,
        blocked_candidates=blocked_candidates,
        helper_results=helper_results,
        belief=belief,
        planner_trace=planner_trace,
    )
