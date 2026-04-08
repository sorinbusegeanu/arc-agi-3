from __future__ import annotations

from v4.composition import ComposedTransitionModelV4, HybridSubgoalBuilderV4
from v4.state.parsedState import ParsedStateV4

from .subgoalTypes import SubgoalProgressV4, SubgoalV4


_MOVEMENT_GAMES = {"ul01", "fs01", "fs02", "fs03", "tp01", "ic01", "va01", "pb01", "pb02", "pb03"}
_CLICK_GAMES = {"pt01", "sy01", "ff01", "sq01", "wm01", "mm01"}


class SubgoalExtractorV4:
    def __init__(self) -> None:
        self.composed_transition_model = ComposedTransitionModelV4()
        self.hybrid_subgoal_builder = HybridSubgoalBuilderV4()

    def extract(self, parsed_state: ParsedStateV4) -> tuple[SubgoalV4, ...]:
        raw_game_id = str(parsed_state.current_observation.game_id)
        game_id = raw_game_id.split("-", 1)[0]
        if game_id == "tb01":
            hybrid_subgoals: tuple[SubgoalV4, ...] = ()
            if parsed_state.composition_reference is not None:
                composed_state = self.composed_transition_model.build(parsed_state)
                hybrid_subgoals = self.hybrid_subgoal_builder.build(parsed_state, composed_state)
            subgoals = [
                *hybrid_subgoals,
                *(
                    [
                        SubgoalV4(
                            subgoal_id="subgoal:hidden:reveal_information",
                            family="hidden",
                            kind="reveal_information",
                            description="Take a safe exploratory action to reduce hidden uncertainty.",
                            required_facts=("belief_unknown_exists",),
                            dependency_ids=(),
                            progress=SubgoalProgressV4(current_value=0.0, target_value=1.0, is_complete=False),
                        )
                    ]
                    if parsed_state.belief_reference is not None and parsed_state.belief_reference.unknown_cell_count > 0
                    else []
                ),
                SubgoalV4(
                    subgoal_id="subgoal:unknown:immediate_progress",
                    family="unknown",
                    kind="immediate_progress",
                    description="Make immediate certified progress from the current state.",
                    required_facts=(),
                    dependency_ids=(),
                    progress=SubgoalProgressV4(current_value=0.0, target_value=1.0),
                ),
            ]
            return tuple(subgoals)
        if game_id == "sv01":
            subgoals = [
                *(
                    [
                        SubgoalV4(
                            subgoal_id="subgoal:temporal:preserve_safety_margin",
                            family="temporal",
                            kind="preserve_safety_margin",
                            description="Choose a low-risk action that preserves temporal and resource safety margin.",
                            required_facts=("temporal_state_present",),
                            dependency_ids=(),
                            progress=SubgoalProgressV4(current_value=0.0, target_value=1.0, is_complete=False),
                        )
                    ]
                    if parsed_state.temporal_reference is not None
                    else []
                ),
                SubgoalV4(
                    subgoal_id="subgoal:unknown:immediate_progress",
                    family="unknown",
                    kind="immediate_progress",
                    description="Make immediate certified progress from the current state.",
                    required_facts=(),
                    dependency_ids=(),
                    progress=SubgoalProgressV4(current_value=0.0, target_value=1.0),
                ),
            ]
            return tuple(subgoals)
        if game_id == "rs01":
            subgoals = [
                *(
                    [
                        SubgoalV4(
                            subgoal_id="subgoal:hypothesis:disambiguate",
                            family="hypothesis",
                            kind="disambiguate_hypothesis",
                            description="Take a low-risk discriminating action to reduce competing hypothesis uncertainty.",
                            required_facts=("hypothesis_candidates_exist",),
                            dependency_ids=(),
                            progress=SubgoalProgressV4(current_value=0.0, target_value=1.0, is_complete=False),
                        )
                    ]
                    if parsed_state.hypothesis_reference is not None and parsed_state.hypothesis_reference.hypothesis_count > 0
                    else []
                ),
                SubgoalV4(
                    subgoal_id="subgoal:unknown:immediate_progress",
                    family="unknown",
                    kind="immediate_progress",
                    description="Make immediate certified progress from the current state.",
                    required_facts=(),
                    dependency_ids=(),
                    progress=SubgoalProgressV4(current_value=0.0, target_value=1.0),
                ),
            ]
            return tuple(subgoals)
        if game_id in _MOVEMENT_GAMES:
            subgoals = [
                *(
                    [
                        SubgoalV4(
                            subgoal_id="subgoal:hypothesis:disambiguate",
                            family="hypothesis",
                            kind="disambiguate_hypothesis",
                            description="Take a low-risk discriminating action to reduce competing hypothesis uncertainty.",
                            required_facts=("hypothesis_candidates_exist",),
                            dependency_ids=(),
                            progress=SubgoalProgressV4(current_value=0.0, target_value=1.0, is_complete=False),
                        )
                    ]
                    if game_id in {"rs01", "pt01"}
                    and parsed_state.hypothesis_reference is not None
                    and parsed_state.hypothesis_reference.hypothesis_count > 0
                    else []
                ),
                *(
                    [
                        SubgoalV4(
                            subgoal_id="subgoal:hidden:reveal_information",
                            family="hidden",
                            kind="reveal_information",
                            description="Take a safe exploratory action to reduce hidden uncertainty.",
                            required_facts=("belief_unknown_exists",),
                            dependency_ids=(),
                            progress=SubgoalProgressV4(current_value=0.0, target_value=1.0, is_complete=False),
                        )
                    ]
                    if parsed_state.belief_reference is not None and parsed_state.belief_reference.unknown_cell_count > 0
                    else []
                ),
                SubgoalV4(
                    subgoal_id="subgoal:movement:immediate_progress",
                    family="movement",
                    kind="immediate_progress",
                    description="Make immediate certified progress from the current movement state.",
                    required_facts=(),
                    dependency_ids=(),
                    progress=SubgoalProgressV4(current_value=0.0, target_value=1.0),
                )
            ]
            if game_id in {"pb01", "pb02", "pb03"}:
                subgoals.append(
                    SubgoalV4(
                        subgoal_id="subgoal:movement:align_push_state",
                        family="movement",
                        kind="align_push_state",
                        description="Reach a useful push-alignment state before committing to a push.",
                        required_facts=(),
                        dependency_ids=(),
                        progress=SubgoalProgressV4(current_value=0.0, target_value=1.0),
                    )
                )
            if game_id in {"va01"}:
                subgoals.append(
                    SubgoalV4(
                        subgoal_id="subgoal:movement:expand_coverage",
                        family="movement",
                        kind="expand_coverage",
                        description="Increase certified visited coverage toward the remaining uncovered cells.",
                        required_facts=(),
                        dependency_ids=(),
                        progress=SubgoalProgressV4(current_value=0.0, target_value=1.0),
                    )
                )
            if game_id in {"fs02", "fs03"}:
                subgoals.append(
                    SubgoalV4(
                        subgoal_id="subgoal:movement:activate_required_switches",
                        family="movement",
                        kind="activate_required_switches",
                        description="Activate the required switch configuration before continuing.",
                        required_facts=(),
                        dependency_ids=(),
                        progress=SubgoalProgressV4(current_value=0.0, target_value=1.0),
                    )
                )
            return tuple(subgoals)
        if game_id in _CLICK_GAMES:
            subgoals = [
                *(
                    [
                        SubgoalV4(
                            subgoal_id="subgoal:hypothesis:disambiguate",
                            family="hypothesis",
                            kind="disambiguate_hypothesis",
                            description="Take a low-risk discriminating action to reduce competing hypothesis uncertainty.",
                            required_facts=("hypothesis_candidates_exist",),
                            dependency_ids=(),
                            progress=SubgoalProgressV4(current_value=0.0, target_value=1.0, is_complete=False),
                        )
                    ]
                    if game_id in {"rs01", "pt01"}
                    and parsed_state.hypothesis_reference is not None
                    and parsed_state.hypothesis_reference.hypothesis_count > 0
                    else []
                ),
                *(
                    [
                        SubgoalV4(
                            subgoal_id="subgoal:hidden:reveal_information",
                            family="hidden",
                            kind="reveal_information",
                            description="Take a safe exploratory action to reduce hidden uncertainty.",
                            required_facts=("belief_unknown_exists",),
                            dependency_ids=(),
                            progress=SubgoalProgressV4(current_value=0.0, target_value=1.0, is_complete=False),
                        )
                    ]
                    if parsed_state.belief_reference is not None and parsed_state.belief_reference.unknown_cell_count > 0
                    else []
                ),
                SubgoalV4(
                    subgoal_id="subgoal:click:immediate_progress",
                    family="click",
                    kind="immediate_progress",
                    description="Make immediate certified progress from the current click state.",
                    required_facts=(),
                    dependency_ids=(),
                    progress=SubgoalProgressV4(current_value=0.0, target_value=1.0),
                )
            ]
            if game_id in {"pt01"}:
                subgoals.append(
                    SubgoalV4(
                        subgoal_id="subgoal:click:advance_board_phase",
                        family="click",
                        kind="advance_board_phase",
                        description="Advance the board toward the next solved rotation phase.",
                        required_facts=(),
                        dependency_ids=(),
                        progress=SubgoalProgressV4(current_value=0.0, target_value=1.0),
                    )
                )
            if game_id in {"sq01"}:
                subgoals.append(
                    SubgoalV4(
                        subgoal_id="subgoal:click:advance_sequence",
                        family="click",
                        kind="advance_sequence",
                        description="Click the next required sequence element.",
                        required_facts=(),
                        dependency_ids=(),
                        progress=SubgoalProgressV4(current_value=0.0, target_value=1.0),
                    )
                )
            if game_id in {"mm01"}:
                subgoals.append(
                    SubgoalV4(
                        subgoal_id="subgoal:click:reveal_or_match_pair",
                        family="click",
                        kind="reveal_or_match_pair",
                        description="Reveal or complete a valid matching pair.",
                        required_facts=(),
                        dependency_ids=(),
                        progress=SubgoalProgressV4(current_value=0.0, target_value=1.0),
                    )
                )
            return tuple(subgoals)
        return (
            SubgoalV4(
                subgoal_id="subgoal:unknown:immediate_progress",
                family="unknown",
                kind="immediate_progress",
                description="Make immediate certified progress from the current state.",
                required_facts=(),
                dependency_ids=(),
                progress=SubgoalProgressV4(current_value=0.0, target_value=1.0),
            ),
        )
