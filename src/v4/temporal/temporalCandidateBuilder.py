from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from v4.planning.planContracts import CandidatePlanScoreV4, CandidatePlanV4
from v4.policy.policyBase import legal_action_from_id
from v4.state.parsedState import ParsedStateV4


@dataclass(frozen=True)
class TemporalActionTemplateV4:
    template_id: str
    goal_kind: str
    description: str
    allowed_action_ids: tuple[int, ...]
    required_resource_names: tuple[str, ...]
    priority: int

    def __post_init__(self) -> None:
        if not self.template_id:
            raise ValueError("template_id must be non-empty")
        if not self.goal_kind:
            raise ValueError("goal_kind must be non-empty")
        if not self.description:
            raise ValueError("description must be non-empty")
        if not self.allowed_action_ids:
            raise ValueError("allowed_action_ids must be non-empty")
        if not self.required_resource_names:
            raise ValueError("required_resource_names must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TemporalCandidateBuilderV4:
    def build(self, parsed_state: ParsedStateV4) -> tuple[CandidatePlanV4, ...]:
        raw_game_id = str(parsed_state.current_observation.game_id)
        game_id = raw_game_id.split("-", 1)[0]
        if game_id != "sv01":
            return ()
        if parsed_state.temporal_reference is None:
            return ()
        legal_actions = []
        for action_id in sorted(int(action_id) for action_id in parsed_state.available_actions):
            try:
                legal_actions.append(legal_action_from_id(action_id, parsed_state=parsed_state))
            except ValueError:
                continue
        templates = (
            TemporalActionTemplateV4(
                template_id="template:sv01:restore_hunger",
                goal_kind="preserve_safety_margin",
                description="Prefer actions that restore or preserve hunger margin.",
                allowed_action_ids=(5, 6, 1, 2, 3, 4),
                required_resource_names=("hunger",),
                priority=3,
            ),
            TemporalActionTemplateV4(
                template_id="template:sv01:restore_warmth",
                goal_kind="preserve_safety_margin",
                description="Prefer actions that restore or preserve warmth margin.",
                allowed_action_ids=(5, 6, 1, 2, 3, 4),
                required_resource_names=("warmth",),
                priority=2,
            ),
            TemporalActionTemplateV4(
                template_id="template:sv01:buy_time_safely",
                goal_kind="preserve_safety_margin",
                description="Prefer low-risk actions that preserve safe temporal horizon.",
                allowed_action_ids=(1, 2, 3, 4, 5),
                required_resource_names=("timer",),
                priority=1,
            ),
        )
        candidates: list[CandidatePlanV4] = []
        for template in templates:
            for action in legal_actions:
                if action.action_id not in template.allowed_action_ids:
                    continue
                candidates.append(
                    CandidatePlanV4(
                        candidate_id=f"candidate:temporal:{template.template_id}:{action.action_id}",
                        family="temporal",
                        plan_kind="temporal_prefix",
                        goal_kind="preserve_safety_margin",
                        subgoal_id="subgoal:temporal:preserve_safety_margin",
                        subgoal_kind="preserve_safety_margin",
                        action_prefix=(action,),
                        required_facts=("temporal_state_present",),
                        forbidden_facts=(),
                        expected_effect={
                            "action_id": action.action_id,
                            "action_name": action.action_name,
                            "template_id": template.template_id,
                            "template_priority": template.priority,
                            "required_resource_names": template.required_resource_names,
                            "state_hash_before": parsed_state.derived_control.state_hash,
                        },
                        score_components=CandidatePlanScoreV4(),
                        rationale_codes=(
                            "source=step7_temporal_builder",
                            f"template_id={template.template_id}",
                            "goal_kind=preserve_safety_margin",
                        ),
                    )
                )
        return tuple(candidates[:8])
