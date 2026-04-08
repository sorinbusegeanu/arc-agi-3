from __future__ import annotations

from v4.affordance import (
    build_click_candidate_actions,
    build_common_affordances,
    build_movement_candidate_actions,
)
from v4.composition import ComposedTransitionModelV4, HybridCandidateBuilderV4
from v4.exploration import (
    ExplorationCandidateBuilderV4,
    InformationGainScorerV4,
    SafeExplorationFilterV4,
    build_probe_templates,
)
from v4.experiments import (
    DisambiguationPlannerV4,
    ExpectedEvidenceModelV4,
    build_experiment_templates,
)
from v4.agentContract.types import V4Action
from v4.memory.localMemory import LocalMemoryStateV4
from v4.planning.planContracts import CandidatePlanScoreV4, CandidatePlanV4
from v4.policy.policyBase import legal_action_from_id
from v4.state.parsedState import ParsedStateV4
from v4.subgoals import (
    SubgoalDependencyResolverV4,
    SubgoalExtractorV4,
    SubgoalProgressEvaluatorV4,
    SubgoalSelectionV4,
)
from v4.temporal import ContingentPlanAnnotatorV4, TemporalCandidateBuilderV4, TemporalVerifierV4


class PlanGeneratorV4:
    def __init__(self) -> None:
        self.subgoal_extractor = SubgoalExtractorV4()
        self.subgoal_progress = SubgoalProgressEvaluatorV4()
        self.subgoal_dependencies = SubgoalDependencyResolverV4()
        self.subgoal_selection = SubgoalSelectionV4()
        self.information_gain_scorer = InformationGainScorerV4()
        self.safe_exploration_filter = SafeExplorationFilterV4()
        self.exploration_candidate_builder = ExplorationCandidateBuilderV4()
        self.disambiguation_planner = DisambiguationPlannerV4()
        self.expected_evidence_model = ExpectedEvidenceModelV4()
        self.temporal_verifier = TemporalVerifierV4()
        self.contingent_plan_annotator = ContingentPlanAnnotatorV4()
        self.temporal_candidate_builder = TemporalCandidateBuilderV4()
        self.composed_transition_model = ComposedTransitionModelV4()
        self.hybrid_candidate_builder = HybridCandidateBuilderV4()

    def generate(
        self,
        parsed_state: ParsedStateV4,
        local_memory_snapshot: LocalMemoryStateV4 | None = None,
    ) -> tuple[tuple[CandidatePlanV4, ...], dict[str, object]]:
        raw_game_id = str(parsed_state.current_observation.game_id)
        game_id = raw_game_id.split("-", 1)[0]
        generated_step6_count = 0
        generated_step7_count = 0
        generated_step8_count = 0
        generator_debug = {
            "entered_branch": "default",
            "step7_pre_filter_count": 0,
            "step7_post_filter_count": 0,
            "active_subgoal_kind": None,
            "normalized_game_id": game_id,
        }

        def _metrics() -> dict[str, object]:
            return {
                "generated_step6_count": generated_step6_count,
                "generated_step7_count": generated_step7_count,
                "generated_step8_count": generated_step8_count,
                "generator_debug": dict(generator_debug),
                "extracted_subgoal_kinds": extracted_subgoal_kinds,
                "subgoal_progress_rows": subgoal_progress_rows,
            }

        affordances = build_common_affordances(parsed_state)
        subgoals = self.subgoal_extractor.extract(parsed_state)
        evaluated = tuple(self.subgoal_progress.evaluate(parsed_state, subgoal) for subgoal in subgoals)
        resolved = self.subgoal_dependencies.resolve(evaluated)
        extracted_subgoal_kinds = tuple(subgoal.kind for subgoal in resolved)
        subgoal_progress_rows = tuple(
            {
                "subgoal_id": subgoal.subgoal_id,
                "subgoal_kind": subgoal.kind,
                "current_value": subgoal.progress.current_value,
                "target_value": subgoal.progress.target_value,
                "is_complete": subgoal.progress.is_complete,
                "dependency_ids": subgoal.dependency_ids,
            }
            for subgoal in resolved
        )
        active_subgoal = self.subgoal_selection.select(resolved)
        generator_debug["active_subgoal_kind"] = active_subgoal.kind
        if active_subgoal.kind == "preserve_safety_margin":
            generator_debug["entered_branch"] = "step7"
            temporal_candidates_before_filter = self.temporal_candidate_builder.build(parsed_state)
            generator_debug["step7_pre_filter_count"] = len(temporal_candidates_before_filter)
            generated_step7_count = len(temporal_candidates_before_filter)
            filtered: list[CandidatePlanV4] = []
            for candidate in temporal_candidates_before_filter:
                accepted, _ = self.temporal_verifier.assess(parsed_state, candidate)
                if not accepted:
                    continue
                note = self.contingent_plan_annotator.build_note(parsed_state, candidate)
                expected_effect = dict(candidate.expected_effect)
                if note is not None:
                    expected_effect["contingent_note"] = note.to_dict()
                filtered.append(
                    CandidatePlanV4(
                        candidate_id=candidate.candidate_id,
                        family=candidate.family,
                        plan_kind=candidate.plan_kind,
                        goal_kind=candidate.goal_kind,
                        subgoal_id=candidate.subgoal_id,
                        subgoal_kind=candidate.subgoal_kind,
                        action_prefix=candidate.action_prefix,
                        required_facts=candidate.required_facts,
                        forbidden_facts=candidate.forbidden_facts,
                        expected_effect=expected_effect,
                        score_components=candidate.score_components,
                        rationale_codes=candidate.rationale_codes + ("temporal_safe_prefix=true",),
                    )
                )
            generator_debug["step7_post_filter_count"] = len(filtered)
            if filtered:
                return tuple(filtered[:8]), _metrics()
            return (), _metrics()
        if active_subgoal.kind in {
            "enable_construction_path",
            "manage_construction_budget",
            "build_under_time_pressure",
            "complete_construction_path",
        }:
            generator_debug["entered_branch"] = "step8"
            composed_state = self.composed_transition_model.build(parsed_state)
            hybrid_candidates = self.hybrid_candidate_builder.build(parsed_state, composed_state, active_subgoal.kind)
            generated_step8_count = len(hybrid_candidates)
            enriched: list[CandidatePlanV4] = []
            for candidate in hybrid_candidates:
                expected_effect = dict(candidate.expected_effect)
                if "composition_present_domains" not in expected_effect:
                    expected_effect["composition_present_domains"] = composed_state.snapshot_reference().present_domain_names
                if "cross_domain_effect_codes" not in expected_effect:
                    expected_effect["cross_domain_effect_codes"] = composed_state.cross_domain_effect_codes
                rationale_codes = candidate.rationale_codes
                if "source=step8_composition" not in rationale_codes:
                    rationale_codes = rationale_codes + ("source=step8_composition",)
                hybrid_code = f"active_hybrid_subgoal={active_subgoal.kind}"
                if hybrid_code not in rationale_codes:
                    rationale_codes = rationale_codes + (hybrid_code,)
                enriched.append(
                    CandidatePlanV4(
                        candidate_id=candidate.candidate_id,
                        family=candidate.family,
                        plan_kind=candidate.plan_kind,
                        goal_kind=candidate.goal_kind,
                        subgoal_id=candidate.subgoal_id,
                        subgoal_kind=candidate.subgoal_kind,
                        action_prefix=candidate.action_prefix,
                        required_facts=candidate.required_facts,
                        forbidden_facts=candidate.forbidden_facts,
                        expected_effect=expected_effect,
                        score_components=candidate.score_components,
                        rationale_codes=rationale_codes,
                    )
                )
            return tuple(enriched[:8]), _metrics()
        if active_subgoal.kind == "disambiguate_hypothesis":
            templates = build_experiment_templates(parsed_state)
            experiment_candidates = self.disambiguation_planner.build(parsed_state, templates)
            generated_step6_count = len(experiment_candidates)
            enriched: list[CandidatePlanV4] = []
            for candidate in experiment_candidates:
                expected_evidence = self.expected_evidence_model.predict(parsed_state, candidate)
                enriched.append(
                    CandidatePlanV4(
                        candidate_id=candidate.candidate_id,
                        family=candidate.family,
                        plan_kind=candidate.plan_kind,
                        goal_kind=candidate.goal_kind,
                        subgoal_id=candidate.subgoal_id,
                        subgoal_kind=candidate.subgoal_kind,
                        action_prefix=candidate.action_prefix,
                        required_facts=candidate.required_facts,
                        forbidden_facts=candidate.forbidden_facts,
                        expected_effect={
                            **candidate.expected_effect,
                            "expected_evidence": tuple(item.to_dict() for item in expected_evidence),
                        },
                        score_components=candidate.score_components,
                        rationale_codes=candidate.rationale_codes + (f"expected_evidence_count={len(expected_evidence)}",),
                    )
                )
            return tuple(enriched[:8]), _metrics()
        if active_subgoal.kind == "reveal_information":
            probe_templates = build_probe_templates(parsed_state)
            probe_candidates = self.exploration_candidate_builder.build(parsed_state, probe_templates)
            filtered_candidates: list[CandidatePlanV4] = []
            for candidate in probe_candidates:
                if not self.safe_exploration_filter.allow(parsed_state, candidate):
                    continue
                info_score = self.information_gain_scorer.score(parsed_state, candidate)
                filtered_candidates.append(
                    CandidatePlanV4(
                        candidate_id=candidate.candidate_id,
                        family=candidate.family,
                        plan_kind=candidate.plan_kind,
                        goal_kind=candidate.goal_kind,
                        subgoal_id=candidate.subgoal_id,
                        subgoal_kind=candidate.subgoal_kind,
                        action_prefix=candidate.action_prefix,
                        required_facts=candidate.required_facts,
                        forbidden_facts=candidate.forbidden_facts,
                        expected_effect=candidate.expected_effect,
                        score_components=CandidatePlanScoreV4(
                            progress_score=candidate.score_components.progress_score,
                            safety_score=candidate.score_components.safety_score,
                            loop_risk_score=candidate.score_components.loop_risk_score,
                            certainty_score=candidate.score_components.certainty_score,
                            total_score=candidate.score_components.total_score + info_score.total_information_score,
                        ),
                        rationale_codes=candidate.rationale_codes + (f"information_gain={info_score.total_information_score}",),
                    )
                )
            return tuple(filtered_candidates[:8]), _metrics()
        del local_memory_snapshot
        return self._build_primitive_candidates(parsed_state, affordances.family, active_subgoal), _metrics()

    def _build_primitive_candidates(self, parsed_state: ParsedStateV4, family: str, active_subgoal) -> tuple[CandidatePlanV4, ...]:
        if family == "movement":
            actions = build_movement_candidate_actions(parsed_state)
        elif family == "click":
            actions = build_click_candidate_actions(parsed_state)
        else:
            built_actions: list[V4Action] = []
            for action_id in sorted(int(action_id) for action_id in parsed_state.available_actions)[:8]:
                try:
                    built_actions.append(legal_action_from_id(action_id, parsed_state=parsed_state))
                except ValueError:
                    continue
            actions = tuple(built_actions)
        candidates: list[CandidatePlanV4] = []
        required_facts = ("construction_domain_present",) if active_subgoal.kind == "enable_construction_path" else ()
        for action in actions[:8]:
            candidates.append(
                CandidatePlanV4(
                    candidate_id=f"candidate:{parsed_state.step_index}:{action.action_id}",
                    family=family,
                    plan_kind="primitive_prefix",
                    goal_kind=active_subgoal.kind,
                    subgoal_id=active_subgoal.subgoal_id,
                    subgoal_kind=active_subgoal.kind,
                    action_prefix=(action,),
                    required_facts=required_facts,
                    forbidden_facts=(),
                    expected_effect={
                        "action_id": action.action_id,
                        "action_name": action.action_name,
                        "state_hash_before": parsed_state.derived_control.state_hash,
                    },
                    score_components=CandidatePlanScoreV4(),
                    rationale_codes=(
                        f"family={family}",
                        "source=step1_generator",
                        f"subgoal_id={active_subgoal.subgoal_id}",
                        f"subgoal_kind={active_subgoal.kind}",
                    ),
                )
            )
        return tuple(candidates[:8])
