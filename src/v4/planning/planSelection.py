from __future__ import annotations

from v4.planning.planContracts import CandidatePlanV4, VerifiedPlanV4
from v4.planning.planVerifier import PlanVerifierV4
from v4.state.parsedState import ParsedStateV4


class PlanSelectionV4:
    def select(
        self,
        parsed_state: ParsedStateV4,
        candidates: tuple[CandidatePlanV4, ...],
        verifier: PlanVerifierV4,
    ) -> tuple[VerifiedPlanV4, dict[str, object]]:
        if not candidates:
            raise ValueError("no candidate plans generated")
        indexed = list(enumerate(candidates))
        indexed.sort(key=lambda item: (-item[1].score_components.total_score, item[0]))
        first_rejected: VerifiedPlanV4 | None = None
        first_accepted: VerifiedPlanV4 | None = None
        accepted_candidate_count_total = 0
        accepted_step6_count = 0
        accepted_step7_count = 0
        accepted_step8_count = 0
        step8_goal_kinds = {
            "enable_construction_path",
            "manage_construction_budget",
            "build_under_time_pressure",
            "complete_construction_path",
        }
        for _, candidate in indexed:
            verified = verifier.verify(parsed_state, candidate)
            if verified.status == "accepted":
                accepted_candidate_count_total += 1
                if candidate.goal_kind == "disambiguate_hypothesis":
                    accepted_step6_count += 1
                if candidate.goal_kind == "preserve_safety_margin":
                    accepted_step7_count += 1
                if candidate.goal_kind in step8_goal_kinds:
                    accepted_step8_count += 1
                if first_accepted is None:
                    first_accepted = verified
            if first_rejected is None:
                first_rejected = verified
        selected = first_accepted if first_accepted is not None else first_rejected
        if selected is None:
            selected = verifier.verify(parsed_state, candidates[0])
        selected_goal_kind = selected.candidate.goal_kind
        return selected, {
            "accepted_candidate_count_total": accepted_candidate_count_total,
            "accepted_step6_count": accepted_step6_count,
            "accepted_step7_count": accepted_step7_count,
            "accepted_step8_count": accepted_step8_count,
            "selected_is_step6": selected_goal_kind == "disambiguate_hypothesis",
            "selected_is_step7": selected_goal_kind == "preserve_safety_margin",
            "selected_is_step8": selected_goal_kind in step8_goal_kinds,
        }
