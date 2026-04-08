from __future__ import annotations

from v4.agentContract.types import V4Action
from v4.planning.planContracts import CandidatePlanScoreV4, CandidatePlanV4
from v4.policy.policyBase import legal_action_from_id
from v4.state.parsedState import ParsedStateV4

from .experimentTemplates import ExperimentTemplateV4


class DisambiguationPlannerV4:
    def build(self, parsed_state: ParsedStateV4, templates: tuple[ExperimentTemplateV4, ...]) -> tuple[CandidatePlanV4, ...]:
        if parsed_state.hypothesis_reference is None or parsed_state.hypothesis_reference.hypothesis_count <= 0:
            return ()
        action_name_by_id = {
            int(action_id): str(action_name)
            for action_id, action_name in zip(
                parsed_state.environment_metadata.action_ids if parsed_state.environment_metadata is not None else (),
                parsed_state.environment_metadata.action_names if parsed_state.environment_metadata is not None else (),
            )
        }
        legal_actions: list[V4Action] = []
        for action_id in sorted(int(value) for value in parsed_state.available_actions):
            try:
                legal_actions.append(legal_action_from_id(action_id, parsed_state=parsed_state))
            except ValueError:
                continue
        candidates: list[CandidatePlanV4] = []
        raw_game_id = str(parsed_state.current_observation.game_id)
        game_id = raw_game_id.split("-", 1)[0]
        for template in templates:
            if game_id == "rs01":
                target_hypothesis_ids = ("hypothesis:rs01:safe_color:A", "hypothesis:rs01:safe_color:B")
            elif game_id == "pt01":
                target_hypothesis_ids = ("hypothesis:pt01:phase:board_active", "hypothesis:pt01:phase:transition_frame")
            else:
                target_hypothesis_ids = ()
            if not target_hypothesis_ids:
                continue
            for action in legal_actions:
                mapped_action_name = action_name_by_id.get(int(action.action_id), action.action_name)
                if action.action_name not in template.allowed_action_names and mapped_action_name not in template.allowed_action_names:
                    continue
                candidates.append(
                    CandidatePlanV4(
                        candidate_id=f"candidate:experiment:{template.template_id}:{action.action_id}",
                        family="hypothesis",
                        plan_kind="experiment_prefix",
                        goal_kind="disambiguate_hypothesis",
                        subgoal_id="subgoal:hypothesis:disambiguate",
                        subgoal_kind="disambiguate_hypothesis",
                        action_prefix=(action,),
                        required_facts=("hypothesis_candidates_exist",),
                        forbidden_facts=(),
                        expected_effect={
                            "action_id": action.action_id,
                            "action_name": action.action_name,
                            "experiment_template_id": template.template_id,
                            "target_hypothesis_ids": target_hypothesis_ids,
                            "state_hash_before": parsed_state.derived_control.state_hash,
                        },
                        score_components=CandidatePlanScoreV4(),
                        rationale_codes=(
                            "source=step6_disambiguation_builder",
                            f"experiment_template_id={template.template_id}",
                            "goal_kind=disambiguate_hypothesis",
                        ),
                    )
                )
        return tuple(candidates[:8])
