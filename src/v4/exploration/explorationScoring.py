from __future__ import annotations

from dataclasses import replace

from v4.planning.planContracts import CandidatePlanScoreV4, CandidatePlanV4
from v4.policy.policyBase import legal_action_from_id
from v4.state.parsedState import ParsedStateV4

from .probeTemplates import ProbeTemplateV4


_ACTION_ALIAS_BY_ID = {
    1: "up",
    2: "down",
    3: "left",
    4: "right",
    5: "inspect",
    6: "click_at",
    7: "inspect_local",
}


def _semantic_action_name(parsed_state: ParsedStateV4, action_id: int, default_name: str) -> str:
    metadata = parsed_state.environment_metadata
    if metadata is not None and metadata.action_ids and metadata.action_names:
        action_map = {int(key): str(value) for key, value in zip(metadata.action_ids, metadata.action_names)}
        if int(action_id) in action_map and action_map[int(action_id)]:
            return action_map[int(action_id)]
    return _ACTION_ALIAS_BY_ID.get(int(action_id), default_name)


class ExplorationCandidateBuilderV4:
    def build(self, parsed_state: ParsedStateV4, probe_templates: tuple[ProbeTemplateV4, ...]) -> tuple[CandidatePlanV4, ...]:
        state_hash = getattr(parsed_state, "state_hash", parsed_state.derived_control.state_hash)
        legal_actions = []
        for action_id in sorted(int(action_id) for action_id in parsed_state.available_actions):
            try:
                base_action = legal_action_from_id(action_id, parsed_state=parsed_state)
            except ValueError:
                continue
            semantic_name = _semantic_action_name(parsed_state, action_id, base_action.action_name)
            legal_actions.append(replace(base_action, action_name=semantic_name))
        candidates: list[CandidatePlanV4] = []
        for template in probe_templates:
            if template.requires_frontier and parsed_state.belief_reference is None:
                continue
            if template.requires_frontier and parsed_state.belief_reference.frontier_cell_count <= 0:
                continue
            for action in legal_actions:
                if action.action_name not in template.allowed_action_names:
                    continue
                candidates.append(
                    CandidatePlanV4(
                        candidate_id=f"candidate:probe:{template.probe_id}:{action.action_id}",
                        family=template.family,
                        plan_kind="probe_prefix",
                        goal_kind="reveal_information",
                        subgoal_id="subgoal:hidden:reveal_information",
                        subgoal_kind="reveal_information",
                        action_prefix=(action,),
                        required_facts=("belief_frontier_exists",) if template.requires_frontier else (),
                        forbidden_facts=(),
                        expected_effect={
                            "action_id": action.action_id,
                            "action_name": action.action_name,
                            "probe_id": template.probe_id,
                            "state_hash_before": state_hash,
                        },
                        score_components=CandidatePlanScoreV4(),
                        rationale_codes=(
                            "source=step4_probe_builder",
                            f"probe_id={template.probe_id}",
                            "goal_kind=reveal_information",
                        ),
                    )
                )
                if len(candidates) >= 8:
                    return tuple(candidates)
        return tuple(candidates[:8])
