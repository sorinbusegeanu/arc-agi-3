from __future__ import annotations

from v3_1.contracts.messages import PlannerDecision
from v3_1.contracts.snapshots import PlanningContext


def final_action_from_candidate(candidate: dict | None) -> dict | None:
    if candidate is None:
        return None
    action = dict(candidate.get("action", {}))
    action["candidate_class"] = candidate.get("candidate_class")
    action["objective_type"] = candidate.get("objective_type")
    action["execution_mode"] = candidate.get("execution_mode")
    action["navigation_mode"] = candidate.get("navigation_mode")
    action["target_entity_id"] = candidate.get("target_entity_id")
    action["target_area_id"] = candidate.get("target_area_id")
    action["required_action_family"] = candidate.get("required_action_family")
    action["skill_id"] = candidate.get("skill_id")
    return action


def package_decision(
    *,
    context: PlanningContext,
    selected: dict | None,
    ranked_candidates: list[dict],
    fallback_candidates: list[dict],
    blocked_candidates: list[dict],
    helper_results: list[dict],
    belief: dict,
    planner_trace: dict,
) -> PlannerDecision:
    selected_action = final_action_from_candidate(selected)
    planner_stats = {
        "candidate_count": len(ranked_candidates) + len(blocked_candidates),
        "survivor_count": len(ranked_candidates),
        "blocked_count": len(blocked_candidates),
        "fallback_count": len(fallback_candidates),
        "reachable_target_count": len(dict(belief.get("world_view", {})).get("reachable_targets", [])),
        "frontier_target_count": len(dict(belief.get("world_view", {})).get("frontier_targets", [])),
        "blocked_target_count": len(dict(belief.get("world_view", {})).get("blocked_targets", [])),
    }
    rationale = "selected_best_ranked_candidate" if selected is not None else "selected_fallback_candidate"
    return PlannerDecision(
        session_id=context.session_id,
        run_id=context.run_id,
        game_id=context.game_id,
        round_id=context.round_id,
        pass_id=context.pass_id,
        plan_context_id=context.plan_context_id,
        blackboard_version=context.blackboard_version,
        memory_version=context.memory_version,
        policy_version=context.policy_version,
        ranker_version=context.ranker_version,
        selected_candidate_id=selected.get("candidate_id") if selected is not None else None,
        selected_action=selected_action,
        ranked_candidates=tuple(ranked_candidates),
        rationale=rationale,
        helper_proposal_ids=tuple(
            result.get("proposal_id", "")
            for result in helper_results
            if result.get("proposal_id")
        ),
        metadata={
            "selected_candidate": selected,
            "fallback_candidates": fallback_candidates,
            "blocked_candidates": blocked_candidates,
            "planner_stats": planner_stats,
            "planner_trace": planner_trace,
        },
    )
