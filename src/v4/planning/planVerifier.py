from __future__ import annotations

from v4.planning.planContracts import CandidatePlanV4, VerifiedPlanV4
from v4.state.parsedState import ParsedStateV4
from v4.temporal import TemporalVerifierV4


class PlanVerifierV4:
    def __init__(self) -> None:
        self.temporal_verifier = TemporalVerifierV4()

    def verify(self, parsed_state: ParsedStateV4, candidate: CandidatePlanV4) -> VerifiedPlanV4:
        if not candidate.action_prefix:
            return VerifiedPlanV4(candidate=candidate, status="rejected", certified_prefix=(), rejection_reasons=("empty_action_prefix",))
        if candidate.goal_kind == "disambiguate_hypothesis":
            if parsed_state.hypothesis_reference is None:
                return VerifiedPlanV4(candidate=candidate, status="rejected", certified_prefix=(), rejection_reasons=("hypothesis_reference_missing",))
            if parsed_state.hypothesis_reference.hypothesis_count <= 1:
                return VerifiedPlanV4(candidate=candidate, status="rejected", certified_prefix=(), rejection_reasons=("no_competing_hypotheses",))
            if candidate.plan_kind != "experiment_prefix":
                return VerifiedPlanV4(candidate=candidate, status="rejected", certified_prefix=(), rejection_reasons=("invalid_disambiguation_plan_kind",))
            if not tuple(candidate.expected_effect.get("target_hypothesis_ids", ())):
                return VerifiedPlanV4(candidate=candidate, status="rejected", certified_prefix=(), rejection_reasons=("missing_target_hypotheses",))
        if candidate.goal_kind == "preserve_safety_margin":
            accepted, rejection_reasons = self.temporal_verifier.assess(parsed_state, candidate)
            if not accepted:
                return VerifiedPlanV4(
                    candidate=candidate,
                    status="rejected",
                    certified_prefix=(),
                    rejection_reasons=rejection_reasons,
                    risk_flags=("temporal_risk",),
                )
        if candidate.goal_kind == "enable_construction_path":
            if "construction_domain_present" not in candidate.required_facts:
                return VerifiedPlanV4(candidate=candidate, status="rejected", certified_prefix=(), rejection_reasons=("construction_domain_requirement_missing",))
        if candidate.goal_kind == "manage_construction_budget":
            if parsed_state.composition_reference is None:
                return VerifiedPlanV4(candidate=candidate, status="rejected", certified_prefix=(), rejection_reasons=("construction_domain_requirement_missing",))
        if candidate.goal_kind == "build_under_time_pressure":
            if parsed_state.temporal_reference is None:
                return VerifiedPlanV4(candidate=candidate, status="rejected", certified_prefix=(), rejection_reasons=("temporal_reference_missing_for_hybrid",))
            if parsed_state.temporal_reference.safe_horizon_steps <= 0:
                return VerifiedPlanV4(candidate=candidate, status="rejected", certified_prefix=(), rejection_reasons=("insufficient_safe_horizon_for_hybrid",))
        if candidate.goal_kind == "complete_construction_path":
            if parsed_state.composition_reference is None:
                return VerifiedPlanV4(candidate=candidate, status="rejected", certified_prefix=(), rejection_reasons=("construction_domain_requirement_missing",))
        if candidate.goal_kind == "reveal_information":
            if parsed_state.belief_reference is None:
                return VerifiedPlanV4(candidate=candidate, status="rejected", certified_prefix=(), rejection_reasons=("belief_reference_missing",))
            if parsed_state.belief_reference.unknown_cell_count <= 0:
                return VerifiedPlanV4(candidate=candidate, status="rejected", certified_prefix=(), rejection_reasons=("no_hidden_information_remaining",))
            if parsed_state.belief_reference.frontier_cell_count <= 0:
                return VerifiedPlanV4(candidate=candidate, status="rejected", certified_prefix=(), rejection_reasons=("no_frontier_available",))
        first_action = candidate.action_prefix[0]
        action_key = f"{first_action.action_id}:{first_action.action_name}"
        if int(first_action.action_id) not in tuple(int(action_id) for action_id in parsed_state.available_actions):
            return VerifiedPlanV4(candidate=candidate, status="rejected", certified_prefix=(), rejection_reasons=("illegal_action_not_available",))
        if parsed_state.derived_control.retry_counts.get(action_key, 0) >= 3:
            return VerifiedPlanV4(candidate=candidate, status="rejected", certified_prefix=(), rejection_reasons=("retry_limit_reached",))
        if action_key in set(parsed_state.derived_control.cooldown_action_keys):
            return VerifiedPlanV4(candidate=candidate, status="rejected", certified_prefix=(), rejection_reasons=("cooldown_active",))
        risk_flags = ()
        if parsed_state.memory_reference is not None and parsed_state.memory_reference.visited_before:
            risk_flags = ("visited_before",)
        return VerifiedPlanV4(
            candidate=candidate,
            status="accepted",
            certified_prefix=candidate.action_prefix,
            rejection_reasons=(),
            risk_flags=risk_flags,
        )
