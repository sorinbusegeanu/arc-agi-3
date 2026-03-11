from __future__ import annotations

from typing import Dict, List, Optional

from codex_baseline_v2.planning.hierarchical_planner import plan_best_first
from codex_baseline_v2.planning.plan_memory import PlanMemoryStateV1, plan_memory_refs
from codex_baseline_v2.planning.planner_state_builder import build_planner_belief_state
from codex_baseline_v2.shared.plan_records import PlanNodeV1, PlanResultV1, SkillExecutionRecordV1, SkillSpecV1
from codex_baseline_v2.shared.schemas import BlackboardStateV2

from .messages import PlannerDecision, PlanningContextSnapshot


class PlannerActor:
    def plan(
        self,
        planning_context: PlanningContextSnapshot,
        blackboard_snapshot: Dict[str, object],
        memory_snapshot: Dict[str, object],
        helper_outputs: Optional[List[Dict[str, object]]] = None,
        ranking_inputs: Optional[Dict[str, object]] = None,
    ) -> PlannerDecision:
        blackboard = BlackboardStateV2.from_dict(blackboard_snapshot)
        skills = [SkillSpecV1.from_dict(row) for row in memory_snapshot.get("skills", [])]
        plan_memory = PlanMemoryStateV1.from_dict(memory_snapshot.get("plan_memory", {}))
        belief = build_planner_belief_state(
            blackboard,
            skills,
            candidate_skills=skills,
            plan_memory_refs=plan_memory_refs(plan_memory),
            plan_memory=plan_memory,
        )
        option_score_map = dict((ranking_inputs or {}).get("score_map", {}))
        nodes, result = plan_best_first(
            belief,
            skills,
            learned_score_map=option_score_map,
            blackboard=blackboard,
            plan_memory=plan_memory,
        )
        selected_instruction = None
        if result is not None:
            selected_node = next((node for node in nodes if node.plan_node_id == result.selected_plan_node_id), None)
            if selected_node is not None:
                selected_instruction = {
                    "instruction_id": f"ray:{planning_context.round_id}:{result.selected_plan_node_id}",
                    "mode": selected_node.notes,
                    "target_poi_id": selected_node.skill_id,
                }
        return PlannerDecision(
            game_id=planning_context.game_id,
            round_id=planning_context.round_id,
            plan_context_id=planning_context.plan_context_id,
            selected_skill_id=result.selected_skill_id if result is not None else None,
            selected_plan_node_id=result.selected_plan_node_id if result is not None else None,
            selected_instruction=selected_instruction,
            planner_reason=result.planner_reason if result is not None else "no_plan",
            plan_nodes=[node.to_dict() for node in nodes],
            plan_result=result.to_dict() if result is not None else None,
            helper_refs=[],
        )
