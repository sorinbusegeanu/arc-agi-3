from __future__ import annotations

from v4.memory.localMemory import LocalMemoryStateV4
from v4.planning.planGenerator import PlanGeneratorV4
from v4.planning.planScorer import PlanScorerV4
from v4.planning.planSelection import PlanSelectionV4
from v4.planning.planVerifier import PlanVerifierV4
from v4.policy.policyBase import PolicyBaseV4, PolicyDecisionV4
from v4.state.parsedState import ParsedStateV4


class CertifiedPlannerPolicyV4(PolicyBaseV4):
    def __init__(self) -> None:
        self.generator = PlanGeneratorV4()
        self.scorer = PlanScorerV4()
        self.verifier = PlanVerifierV4()
        self.selection = PlanSelectionV4()

    def decide(self, parsed_state: ParsedStateV4) -> PolicyDecisionV4:
        candidates, generation_metrics = self.generator.generate(parsed_state, local_memory_snapshot=LocalMemoryStateV4())
        generated_step6_count = int(generation_metrics.get("generated_step6_count", 0) or 0)
        generated_step7_count = int(generation_metrics.get("generated_step7_count", 0) or 0)
        generated_step8_count = int(generation_metrics.get("generated_step8_count", 0) or 0)
        scored = tuple(self.scorer.score(parsed_state, candidate) for candidate in candidates)
        selected, selection_metrics = self.selection.select(parsed_state, scored, self.verifier)
        if selected.status != "accepted":
            raise ValueError("no certified plan available")
        return PolicyDecisionV4(
            short_plan=selected.certified_prefix,
            annotations={
                "policy": "certified_planner",
                "candidate_count": len(scored),
                "accepted_candidate_count": int(selection_metrics.get("accepted_candidate_count_total", 0)),
                "verified_candidate_id": selected.candidate.candidate_id,
                "verified_status": selected.status,
                "certified_prefix_length": len(selected.certified_prefix),
                "subgoal_id": selected.candidate.subgoal_id,
                "subgoal_kind": selected.candidate.subgoal_kind,
                "goal_kind": selected.candidate.goal_kind,
                "generated_step6_count": generated_step6_count,
                "generated_step7_count": generated_step7_count,
                "generated_step8_count": generated_step8_count,
                "accepted_step6_count": int(selection_metrics.get("accepted_step6_count", 0)),
                "accepted_step7_count": int(selection_metrics.get("accepted_step7_count", 0)),
                "accepted_step8_count": int(selection_metrics.get("accepted_step8_count", 0)),
                "selected_is_step6": bool(selection_metrics.get("selected_is_step6", False)),
                "selected_is_step7": bool(selection_metrics.get("selected_is_step7", False)),
                "selected_is_step8": bool(selection_metrics.get("selected_is_step8", False)),
                "generator_debug": generation_metrics.get("generator_debug", {}),
                "generation_metrics_snapshot": dict(generation_metrics),
                "extracted_subgoal_kinds": generation_metrics.get("extracted_subgoal_kinds", ()),
                "subgoal_progress_rows": generation_metrics.get("subgoal_progress_rows", ()),
                "hypothesis_count": parsed_state.hypothesis_reference.hypothesis_count if parsed_state.hypothesis_reference is not None else 0,
                "safe_horizon_steps": parsed_state.temporal_reference.safe_horizon_steps if parsed_state.temporal_reference is not None else 0,
                "hazard_window_remaining": parsed_state.temporal_reference.hazard_window_remaining if parsed_state.temporal_reference is not None else None,
                "composition_domain_count": parsed_state.composition_reference.domain_count if parsed_state.composition_reference is not None else 0,
                "composition_present_domains": parsed_state.composition_reference.present_domain_names if parsed_state.composition_reference is not None else (),
                "composition_cross_domain_effect_count": parsed_state.composition_reference.cross_domain_effect_count if parsed_state.composition_reference is not None else 0,
                "belief_frontier_cell_count": parsed_state.belief_reference.frontier_cell_count if parsed_state.belief_reference is not None else 0,
                "belief_unknown_cell_count": parsed_state.belief_reference.unknown_cell_count if parsed_state.belief_reference is not None else parsed_state.derived_control.unknown_cell_count,
                "rejection_reasons": selected.rejection_reasons,
                "risk_flags": selected.risk_flags,
            },
        )
