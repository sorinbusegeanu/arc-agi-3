from __future__ import annotations

from dataclasses import replace

from v4.planning.planContracts import CandidatePlanScoreV4, CandidatePlanV4
from v4.state.parsedState import ParsedStateV4


class PlanScorerV4:
    def score(self, parsed_state: ParsedStateV4, candidate: CandidatePlanV4) -> CandidatePlanV4:
        first_action = candidate.action_prefix[0]
        action_key = f"{first_action.action_id}:{first_action.action_name}"
        progress_score = 1.0
        safety_score = 1.0
        loop_risk_score = float(parsed_state.derived_control.retry_counts.get(action_key, 0))
        certainty_score = 1.0
        if candidate.goal_kind == "disambiguate_hypothesis":
            hypothesis_count = parsed_state.hypothesis_reference.hypothesis_count if parsed_state.hypothesis_reference is not None else 0
            expected_evidence_count = len(candidate.expected_effect.get("expected_evidence", ()))
            disambiguation_bonus = 1.0 if hypothesis_count > 1 and expected_evidence_count > 0 else 0.0
            certainty_score += 0.25
            total_score = progress_score + safety_score + certainty_score - loop_risk_score + disambiguation_bonus
        elif candidate.goal_kind == "preserve_safety_margin":
            safe_horizon = parsed_state.temporal_reference.safe_horizon_steps if parsed_state.temporal_reference is not None else 0
            hazard_window_remaining = parsed_state.temporal_reference.hazard_window_remaining if parsed_state.temporal_reference is not None else None
            temporal_bonus = 1.0 if safe_horizon >= 1 else 0.0
            if hazard_window_remaining is not None and hazard_window_remaining <= 1:
                temporal_bonus += 0.5
            if candidate.plan_kind == "temporal_prefix":
                temporal_bonus += float(int(candidate.expected_effect.get("template_priority", 0) or 0)) * 0.5
            certainty_score += 0.25
            total_score = progress_score + safety_score + certainty_score - loop_risk_score + temporal_bonus
        elif candidate.goal_kind in {
            "enable_construction_path",
            "manage_construction_budget",
            "build_under_time_pressure",
            "complete_construction_path",
        }:
            cross_domain_effect_count = len(candidate.expected_effect.get("cross_domain_effect_codes", ()))
            composition_bonus = 0.5 if cross_domain_effect_count > 0 else 0.0
            if candidate.plan_kind == "hybrid_prefix":
                composition_bonus += float(int(candidate.expected_effect.get("template_priority", 0) or 0)) * 0.5
            certainty_score += 0.25
            total_score = progress_score + safety_score + certainty_score - loop_risk_score + composition_bonus
        elif candidate.goal_kind == "reveal_information":
            certainty_score += 0.25
            total_score = candidate.score_components.total_score + progress_score + safety_score + certainty_score - loop_risk_score
        else:
            subgoal_bonus = 0.5 if candidate.subgoal_kind != "immediate_progress" else 0.0
            total_score = progress_score + safety_score + certainty_score - loop_risk_score + subgoal_bonus
        return replace(
            candidate,
            score_components=CandidatePlanScoreV4(
                progress_score=progress_score,
                safety_score=safety_score,
                loop_risk_score=loop_risk_score,
                certainty_score=certainty_score,
                total_score=total_score,
            ),
        )
