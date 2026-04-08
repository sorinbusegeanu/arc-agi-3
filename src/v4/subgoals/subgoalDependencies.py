from __future__ import annotations

from dataclasses import replace

from .subgoalTypes import SubgoalV4


_DEPENDENCY_BY_KIND = {
    "align_push_state": "subgoal:movement:immediate_progress",
    "expand_coverage": "subgoal:movement:immediate_progress",
    "activate_required_switches": "subgoal:movement:immediate_progress",
    "advance_board_phase": "subgoal:click:immediate_progress",
    "advance_sequence": "subgoal:click:immediate_progress",
    "reveal_or_match_pair": "subgoal:click:immediate_progress",
}


class SubgoalDependencyResolverV4:
    def resolve(self, subgoals: tuple[SubgoalV4, ...]) -> tuple[SubgoalV4, ...]:
        existing = {subgoal.subgoal_id for subgoal in subgoals}
        resolved: list[SubgoalV4] = []
        for subgoal in subgoals:
            dependency_id = _DEPENDENCY_BY_KIND.get(subgoal.kind)
            if subgoal.kind == "immediate_progress":
                resolved.append(replace(subgoal, dependency_ids=()))
            elif dependency_id is not None and dependency_id in existing:
                resolved.append(replace(subgoal, dependency_ids=(dependency_id,)))
            else:
                resolved.append(replace(subgoal, dependency_ids=()))
        return tuple(resolved)
