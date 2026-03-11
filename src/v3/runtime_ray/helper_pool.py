from __future__ import annotations

from typing import Dict, List

from codex_baseline_v2.executor.route_planner import RoutePlanV2

from .messages import CandidateBatch, HelperTaskRequest, HypothesisProposalBatch, PlanningContextSnapshot, RouteAnalysisResult, ScoreFeatureBatch


class PlanningHelperWorker:
    def __init__(self, worker_id: str) -> None:
        self.worker_id = worker_id

    def run(self, request: HelperTaskRequest, context: PlanningContextSnapshot):
        if request.helper_mode == "candidate_generation":
            return CandidateBatch(plan_context_id=context.plan_context_id, helper_mode=request.helper_mode, candidate_skill_ids=list(request.candidate_skill_ids))
        if request.helper_mode == "route_analysis":
            return RouteAnalysisResult(plan_context_id=context.plan_context_id, candidate_skill_ids=list(request.candidate_skill_ids), route_features={skill_id: {} for skill_id in request.candidate_skill_ids})
        if request.helper_mode == "score_features":
            return ScoreFeatureBatch(plan_context_id=context.plan_context_id, candidate_skill_ids=list(request.candidate_skill_ids), feature_rows={skill_id: {} for skill_id in request.candidate_skill_ids})
        return HypothesisProposalBatch(plan_context_id=context.plan_context_id, proposal_type=request.helper_mode, proposals=[])
