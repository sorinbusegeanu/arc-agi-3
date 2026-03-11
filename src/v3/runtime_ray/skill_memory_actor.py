from __future__ import annotations

from typing import Dict, List, Optional

from codex_baseline_v2.planning.plan_memory import PlanMemoryStateV1, reconcile_plan_memory
from codex_baseline_v2.planning.skill_inducer import induce_skills
from codex_baseline_v2.planning.skill_library import reconcile_skill_library
from codex_baseline_v2.shared.plan_records import SkillExecutionRecordV1, SkillSpecV1
from codex_baseline_v2.shared.schemas import BlackboardStateV2

from .messages import MemoryReconcileRequest, MemoryReconcileResult
from .versions import new_memory_version


class SkillMemoryActor:
    def __init__(self) -> None:
        self.skills: List[SkillSpecV1] = []
        self.skill_executions: List[SkillExecutionRecordV1] = []
        self.plan_memory = PlanMemoryStateV1(schema_version="v3")
        self.snapshot_registry: Dict[str, Dict[str, object]] = {}

    def reconcile(self, request: MemoryReconcileRequest) -> MemoryReconcileResult:
        blackboard = BlackboardStateV2.from_dict(request.blackboard) if request.blackboard is not None else None
        if blackboard is not None:
            self.skills = induce_skills(blackboard, existing=self.skills)
        self.skills = reconcile_skill_library(self.skills, self.skill_executions)
        self.plan_memory = reconcile_plan_memory(self.skills, self.skill_executions, self.plan_memory)
        version = new_memory_version(request.game_id, request.round_id)
        snapshot_ref = f"mem_snapshot:{version}"
        payload = {
            "skills": [row.to_dict() for row in self.skills],
            "skill_executions": [row.to_dict() for row in self.skill_executions],
            "plan_memory": self.plan_memory.to_dict(),
        }
        self.snapshot_registry[snapshot_ref] = payload
        return MemoryReconcileResult(
            game_id=request.game_id,
            round_id=request.round_id,
            memory_version=version,
            snapshot_ref=snapshot_ref,
            skills=[row.to_dict() for row in self.skills],
            skill_executions=[row.to_dict() for row in self.skill_executions],
            plan_memory=self.plan_memory.to_dict(),
            reconcile_stats={"skill_count": len(self.skills), "execution_count": len(self.skill_executions)},
        )

    def append_execution(self, execution: Dict[str, object]) -> None:
        self.skill_executions.append(SkillExecutionRecordV1.from_dict(execution))

    def get_snapshot(self, snapshot_ref: str) -> Optional[Dict[str, object]]:
        return self.snapshot_registry.get(snapshot_ref)
