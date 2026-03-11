from __future__ import annotations

from typing import Dict, List

from codex_baseline_v2.shared.schemas import DependencyGraphStateV1, SubgoalNodeV1


def subgoal_map(graph: DependencyGraphStateV1 | None) -> Dict[str, SubgoalNodeV1]:
    if graph is None:
        return {}
    return {subgoal.subgoal_id: subgoal for subgoal in graph.subgoals}


def enabled_subgoals(graph: DependencyGraphStateV1 | None) -> List[SubgoalNodeV1]:
    if graph is None:
        return []
    return [subgoal for subgoal in graph.subgoals if subgoal.status in {"enabled", "verified"}]
