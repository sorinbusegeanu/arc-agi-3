from __future__ import annotations

import ray

from v3_1.planning.planner_service import plan


@ray.remote
class PlannerAgent:
    def __init__(self, planning_cfg: object) -> None:
        self.planning_cfg = planning_cfg

    def decide(
        self,
        context,
        blackboard_snapshot: dict,
        memory_snapshot: dict,
        helper_results: list[dict] | None = None,
        mechanic_graph_snapshot: dict | None = None,
        deterministic_hypotheses: dict | None = None,
        llm_hypotheses: dict | None = None,
        hypothesis_registry_snapshot: dict | None = None,
    ):
        return plan(
            context,
            blackboard_snapshot,
            memory_snapshot,
            self.planning_cfg,
            helper_results,
            mechanic_graph_snapshot,
            deterministic_hypotheses,
            llm_hypotheses,
            hypothesis_registry_snapshot,
        )
