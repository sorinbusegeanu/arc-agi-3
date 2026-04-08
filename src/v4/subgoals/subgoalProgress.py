from __future__ import annotations

from dataclasses import replace

from v4.composition import ComposedTransitionModelV4

from .subgoalTypes import SubgoalProgressV4, SubgoalV4
from v4.state.parsedState import ParsedStateV4


class SubgoalProgressEvaluatorV4:
    def evaluate(self, parsed_state: ParsedStateV4, subgoal: SubgoalV4) -> SubgoalV4:
        kind = subgoal.kind
        if kind == "enable_construction_path":
            return replace(subgoal, progress=SubgoalProgressV4(current_value=0.0, target_value=1.0, is_complete=False))
        if kind == "manage_construction_budget":
            return replace(subgoal, progress=SubgoalProgressV4(current_value=0.0, target_value=1.0, is_complete=False))
        if kind == "reveal_before_build":
            unknown_count = parsed_state.belief_reference.unknown_cell_count if parsed_state.belief_reference is not None else 0
            return replace(
                subgoal,
                progress=SubgoalProgressV4(
                    current_value=1.0 if unknown_count == 0 else 0.0,
                    target_value=1.0,
                    is_complete=(unknown_count == 0),
                ),
            )
        if kind == "disambiguate_before_build":
            hypothesis_count = parsed_state.hypothesis_reference.hypothesis_count if parsed_state.hypothesis_reference is not None else 0
            return replace(
                subgoal,
                progress=SubgoalProgressV4(
                    current_value=1.0 if hypothesis_count <= 1 else 0.0,
                    target_value=1.0,
                    is_complete=(hypothesis_count <= 1),
                ),
            )
        if kind == "build_under_time_pressure":
            safe_horizon = parsed_state.temporal_reference.safe_horizon_steps if parsed_state.temporal_reference is not None else 0
            return replace(
                subgoal,
                progress=SubgoalProgressV4(
                    current_value=1.0 if safe_horizon >= 1 else 0.0,
                    target_value=1.0,
                    is_complete=(safe_horizon >= 1),
                ),
            )
        if kind == "complete_construction_path":
            return replace(subgoal, progress=SubgoalProgressV4(current_value=0.0, target_value=1.0, is_complete=False))
        if kind == "preserve_safety_margin":
            if parsed_state.temporal_reference is None:
                progress = SubgoalProgressV4(current_value=0.0, target_value=1.0, is_complete=False)
            else:
                safe_horizon = parsed_state.temporal_reference.safe_horizon_steps
                progress = SubgoalProgressV4(
                    current_value=float(max(safe_horizon, 0)),
                    target_value=float(max(safe_horizon, 0) + 1),
                    is_complete=False,
                )
            return replace(subgoal, progress=progress)
        if kind == "disambiguate_hypothesis":
            hypothesis_count = parsed_state.hypothesis_reference.hypothesis_count if parsed_state.hypothesis_reference is not None else 0
            progress = SubgoalProgressV4(
                current_value=1.0 if hypothesis_count <= 1 else 0.0,
                target_value=1.0,
                is_complete=(hypothesis_count <= 1),
            )
            return replace(subgoal, progress=progress)
        if kind == "reveal_information":
            belief_reference = parsed_state.belief_reference
            unknown_count = (
                belief_reference.unknown_cell_count
                if belief_reference is not None
                else parsed_state.derived_control.unknown_cell_count
            )
            frontier_count = belief_reference.frontier_cell_count if belief_reference is not None else 0
            progress = SubgoalProgressV4(
                current_value=1.0 if unknown_count == 0 else 0.0,
                target_value=1.0,
                is_complete=(unknown_count == 0) or (frontier_count == 0 and unknown_count > 0),
            )
            return replace(subgoal, progress=progress)
        if kind in {"immediate_progress", "align_push_state", "activate_required_switches", "advance_board_phase", "advance_sequence"}:
            progress = SubgoalProgressV4(current_value=0.0, target_value=1.0, is_complete=False)
            return replace(subgoal, progress=progress)
        if kind == "expand_coverage":
            covered = max(0, int(parsed_state.derived_control.revealed_cell_count))
            unknown = max(0, int(parsed_state.derived_control.unknown_cell_count))
            target = float(max(1, covered + unknown))
            current = float(covered)
            progress = SubgoalProgressV4(current_value=current, target_value=target, is_complete=(unknown == 0 and covered > 0))
            return replace(subgoal, progress=progress)
        if kind == "reveal_or_match_pair":
            revealed = max(0, int(parsed_state.derived_control.revealed_cell_count))
            unknown = max(0, int(parsed_state.derived_control.unknown_cell_count))
            progress = SubgoalProgressV4(
                current_value=float(revealed),
                target_value=float(max(1, revealed + unknown)),
                is_complete=(unknown == 0 and revealed > 0),
            )
            return replace(subgoal, progress=progress)
        return replace(subgoal, progress=subgoal.progress)
