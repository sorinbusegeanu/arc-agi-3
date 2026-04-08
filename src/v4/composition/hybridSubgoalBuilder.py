from __future__ import annotations

from v4.state.parsedState import ParsedStateV4
from v4.subgoals.subgoalTypes import SubgoalProgressV4, SubgoalV4

from .domainState import ComposedDomainStateV4


class HybridSubgoalBuilderV4:
    def build(self, parsed_state: ParsedStateV4, composed_state: ComposedDomainStateV4) -> tuple[SubgoalV4, ...]:
        raw_game_id = str(parsed_state.current_observation.game_id)
        game_id = raw_game_id.split("-", 1)[0]
        if game_id != "tb01":
            return ()
        subgoals: list[SubgoalV4] = []
        if any(slice_.domain_name == "construction" and slice_.is_present for slice_ in composed_state.domain_slices):
            subgoals.append(
                SubgoalV4(
                    subgoal_id="subgoal:hybrid:enable_construction_path",
                    family="hybrid",
                    kind="enable_construction_path",
                    description="Enable a traversable path by combining movement and construction actions.",
                    required_facts=("construction_domain_present",),
                    dependency_ids=(),
                    progress=SubgoalProgressV4(current_value=0.0, target_value=1.0, is_complete=False),
                )
            )
        if "construction_budget_active" in composed_state.cross_domain_effect_codes:
            subgoals.append(
                SubgoalV4(
                    subgoal_id="subgoal:hybrid:manage_construction_budget",
                    family="hybrid",
                    kind="manage_construction_budget",
                    description="Manage remaining bridge budget before committing to further construction.",
                    required_facts=("construction_budget_active",),
                    dependency_ids=("subgoal:hybrid:enable_construction_path",),
                    progress=SubgoalProgressV4(current_value=0.0, target_value=1.0, is_complete=False),
                )
            )
        if "construction_under_temporal_constraints" in composed_state.cross_domain_effect_codes:
            subgoals.append(
                SubgoalV4(
                    subgoal_id="subgoal:hybrid:build_under_time_pressure",
                    family="hybrid",
                    kind="build_under_time_pressure",
                    description="Choose a safe construction progression under temporal constraints.",
                    required_facts=("construction_under_temporal_constraints",),
                    dependency_ids=("subgoal:hybrid:enable_construction_path",),
                    progress=SubgoalProgressV4(current_value=0.0, target_value=1.0, is_complete=False),
                )
            )
        if "construction_path_not_yet_complete" in composed_state.cross_domain_effect_codes:
            subgoals.append(
                SubgoalV4(
                    subgoal_id="subgoal:hybrid:complete_construction_path",
                    family="hybrid",
                    kind="complete_construction_path",
                    description="Complete the remaining construction path toward the goal.",
                    required_facts=("construction_path_not_yet_complete",),
                    dependency_ids=("subgoal:hybrid:enable_construction_path",),
                    progress=SubgoalProgressV4(current_value=0.0, target_value=1.0, is_complete=False),
                )
            )
        return tuple(subgoals)
