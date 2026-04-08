from __future__ import annotations

from v4_5.contracts import PlanCandidate, PlanCandidateSet, PlanDecision, PlannerContext, SCHEMA_VERSION


class PlanningAdapter:
    reused_modules = ("src/v4/planning/*", "src/v4/subgoals/*")

    def select_best(self, context: PlannerContext, candidate_sets: tuple[PlanCandidateSet, ...]) -> PlanDecision:
        candidates = []
        for candidate_set in candidate_sets:
            candidates.extend(candidate_set.candidates)
        if not candidates:
            return PlanDecision(
                schema_version=SCHEMA_VERSION,
                agent_name="PlanningAdapter",
                round_id=context.round_id,
                selected_candidate=None,
                selected_prefix=(),
                rationale_codes=("NO_CANDIDATES",),
            )
        best = sorted(
            candidates,
            key=lambda item: (
                0 if item.verified else 1,
                -float(item.score),
                item.plugin_name,
                item.candidate_id,
            ),
        )[0]
        return PlanDecision(
            schema_version=SCHEMA_VERSION,
            agent_name="PlanningAdapter",
            round_id=context.round_id,
            selected_candidate=best,
            selected_prefix=best.action_prefix,
            rationale_codes=("DETERMINISTIC_SELECTION",),
        )
