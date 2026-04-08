from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from v4.composition.domainState import ComposedDomainStateV4
from v4.planning.planContracts import CandidatePlanScoreV4, CandidatePlanV4
from v4.policy.policyBase import legal_action_from_id
from v4.state.parsedState import ParsedStateV4


@dataclass(frozen=True)
class HybridActionTemplateV4:
    template_id: str
    goal_kind: str
    description: str
    allowed_action_ids: tuple[int, ...]
    required_effect_codes: tuple[str, ...]
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HybridCandidateBuilderV4:
    def build(
        self,
        parsed_state: ParsedStateV4,
        composed_state: ComposedDomainStateV4,
        active_subgoal_kind: str,
    ) -> tuple[CandidatePlanV4, ...]:
        raw_game_id = str(parsed_state.current_observation.game_id)
        game_id = raw_game_id.split("-", 1)[0]
        if game_id != "tb01":
            return ()
        templates_by_subgoal = {
            "enable_construction_path": (
                HybridActionTemplateV4(
                    template_id="template:tb01:move_to_staging",
                    goal_kind="enable_construction_path",
                    description="Move into staging position before building.",
                    allowed_action_ids=(1, 2, 3, 4),
                    required_effect_codes=(),
                    priority=2,
                ),
                HybridActionTemplateV4(
                    template_id="template:tb01:place_bridge_segment",
                    goal_kind="enable_construction_path",
                    description="Place a bridge segment when movement and construction coexist.",
                    allowed_action_ids=(6,),
                    required_effect_codes=("movement_and_construction_actions_coexist",),
                    priority=3,
                ),
            ),
            "manage_construction_budget": (
                HybridActionTemplateV4(
                    template_id="template:tb01:conserve_budget_move",
                    goal_kind="manage_construction_budget",
                    description="Move while conserving remaining bridge budget.",
                    allowed_action_ids=(1, 2, 3, 4),
                    required_effect_codes=("construction_budget_active",),
                    priority=3,
                ),
                HybridActionTemplateV4(
                    template_id="template:tb01:budgeted_build",
                    goal_kind="manage_construction_budget",
                    description="Build only when the budget still supports it.",
                    allowed_action_ids=(6,),
                    required_effect_codes=("construction_budget_active", "movement_and_construction_actions_coexist"),
                    priority=2,
                ),
            ),
            "build_under_time_pressure": (
                HybridActionTemplateV4(
                    template_id="template:tb01:fast_build",
                    goal_kind="build_under_time_pressure",
                    description="Build quickly under temporal pressure.",
                    allowed_action_ids=(6,),
                    required_effect_codes=("construction_under_temporal_constraints",),
                    priority=3,
                ),
                HybridActionTemplateV4(
                    template_id="template:tb01:fast_move",
                    goal_kind="build_under_time_pressure",
                    description="Move quickly under temporal pressure.",
                    allowed_action_ids=(1, 2, 3, 4),
                    required_effect_codes=("construction_under_temporal_constraints",),
                    priority=2,
                ),
            ),
            "complete_construction_path": (
                HybridActionTemplateV4(
                    template_id="template:tb01:complete_path_build",
                    goal_kind="complete_construction_path",
                    description="Build the remaining path toward the goal.",
                    allowed_action_ids=(6,),
                    required_effect_codes=("construction_path_not_yet_complete",),
                    priority=3,
                ),
                HybridActionTemplateV4(
                    template_id="template:tb01:complete_path_move",
                    goal_kind="complete_construction_path",
                    description="Move along the partially completed path.",
                    allowed_action_ids=(1, 2, 3, 4),
                    required_effect_codes=("construction_path_not_yet_complete",),
                    priority=2,
                ),
            ),
        }
        templates = templates_by_subgoal.get(active_subgoal_kind)
        if templates is None:
            return ()
        legal_actions = []
        for action_id in sorted(int(action_id) for action_id in parsed_state.available_actions):
            payload = None
            if action_id == 6:
                payload = {"x": 0, "y": 0, "game_id": str(parsed_state.current_observation.game_id)}
            try:
                legal_actions.append(legal_action_from_id(action_id, parsed_state=parsed_state, payload=payload))
            except ValueError:
                continue
        active_effects = set(composed_state.cross_domain_effect_codes)
        candidates: list[CandidatePlanV4] = []
        for template in templates:
            if any(code not in active_effects for code in template.required_effect_codes):
                continue
            for action in legal_actions:
                if action.action_id not in template.allowed_action_ids:
                    continue
                required_facts = tuple(template.required_effect_codes) if template.required_effect_codes else ("construction_domain_present",)
                candidates.append(
                    CandidatePlanV4(
                        candidate_id=f"candidate:hybrid:{template.template_id}:{action.action_id}",
                        family="hybrid",
                        plan_kind="hybrid_prefix",
                        goal_kind=active_subgoal_kind,
                        subgoal_id=f"subgoal:hybrid:{active_subgoal_kind}",
                        subgoal_kind=active_subgoal_kind,
                        action_prefix=(action,),
                        required_facts=required_facts,
                        forbidden_facts=(),
                        expected_effect={
                            "action_id": action.action_id,
                            "action_name": action.action_name,
                            "template_id": template.template_id,
                            "template_priority": template.priority,
                            "composition_present_domains": composed_state.snapshot_reference().present_domain_names,
                            "cross_domain_effect_codes": composed_state.cross_domain_effect_codes,
                            "state_hash_before": parsed_state.derived_control.state_hash,
                        },
                        score_components=CandidatePlanScoreV4(),
                        rationale_codes=(
                            "source=step8_hybrid_builder",
                            f"template_id={template.template_id}",
                            f"active_hybrid_subgoal={active_subgoal_kind}",
                        ),
                    )
                )
        return tuple(candidates[:8])
