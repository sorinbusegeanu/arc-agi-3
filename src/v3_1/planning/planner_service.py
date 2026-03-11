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
    return package_decision(
        context=context,
        selected=selected,
        ranked_candidates=reranked,
        fallback_candidates=fallback_set,
        blocked_candidates=blocked_candidates,
        helper_results=helper_results,
        belief=belief,
    )
