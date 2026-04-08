from __future__ import annotations

from .subgoalTypes import SubgoalV4


class SubgoalSelectionV4:
    def select(self, subgoals: tuple[SubgoalV4, ...]) -> SubgoalV4:
        if not subgoals:
            raise ValueError("no subgoals available")
        seen = set()
        for subgoal in subgoals:
            if subgoal.progress.is_complete:
                seen.add(subgoal.subgoal_id)
                continue
            if subgoal.kind != "immediate_progress" and (not subgoal.dependency_ids or all(dependency_id in seen for dependency_id in subgoal.dependency_ids)):
                return subgoal
            seen.add(subgoal.subgoal_id)
        seen = set()
        for subgoal in subgoals:
            if subgoal.progress.is_complete:
                seen.add(subgoal.subgoal_id)
                continue
            if not subgoal.dependency_ids or all(dependency_id in seen for dependency_id in subgoal.dependency_ids):
                return subgoal
            seen.add(subgoal.subgoal_id)
        return subgoals[0]
