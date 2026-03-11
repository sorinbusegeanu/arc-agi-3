from __future__ import annotations

from dataclasses import dataclass, field

from v3_1.contracts.snapshots import MemorySnapshot
from v3_1.contracts.versions import next_memory_version
from v3_1.memory.cooldowns import advance_cooldowns, apply_failure_cooldowns
from v3_1.memory.exhaustion import exhausted_candidates, exhaustion_snapshot
from v3_1.memory.plan_memory import update_plan_memory
from v3_1.memory.retries import update_retry_ledgers
from v3_1.memory.skill_library import rebuild_skill_library, update_skill_execution_stats
from v3_1.utils.ids import make_handle


@dataclass
class SkillMemoryState:
    session_id: str
    revision: int = 0
    state: dict = field(default_factory=lambda: {
        "skill_library": {},
        "plan_memory": {
            "history": [],
            "repeated_failures": {},
            "movement_memory": {},
            "recovery_memory": {},
            "no_progress_rounds": 0,
        },
        "cooldowns": {},
        "retries": {},
        "exhausted": [],
        "exhaustion_map": {"candidate": [], "target": [], "area": []},
    })

    @classmethod
    def from_snapshot(cls, session_id: str, snapshot: MemorySnapshot) -> "SkillMemoryState":
        return cls(session_id=session_id, revision=int(snapshot.memory_version.split(":")[-1]), state=dict(snapshot.state))

    def reconcile(self, *, round_id: int, pass_id: int, blackboard_state: dict, decision: dict | None, outcome: dict | None, retry_limit: int, cooldown_rounds: int) -> MemorySnapshot:
        selected = dict(decision.get("metadata", {}).get("selected_candidate", {})) if isinstance(decision, dict) else {}
        target_entity_id = selected.get("target_entity_id")
        target_area_id = selected.get("target_area_id")
        candidate_id = outcome.get("candidate_id") if isinstance(outcome, dict) else None
        success = bool(outcome.get("success") or outcome.get("outcome", {}).get("success")) if isinstance(outcome, dict) else False
        termination_reason = None
        if isinstance(outcome, dict):
            termination_reason = outcome.get("termination_reason") or outcome.get("outcome", {}).get("termination_reason")

        cooldowns = advance_cooldowns(self.state.get("cooldowns", {}))
        retries = update_retry_ledgers(
            self.state.get("retries", {}),
            candidate_id=candidate_id or selected.get("candidate_id"),
            target_entity_id=target_entity_id,
            target_area_id=target_area_id,
            success=success,
            termination_reason=termination_reason,
        )
        if outcome is not None and not success:
            cooldowns = apply_failure_cooldowns(
                cooldowns,
                candidate_id=candidate_id or selected.get("candidate_id"),
                target_entity_id=target_entity_id,
                target_area_id=target_area_id,
                cooldown_rounds=cooldown_rounds,
                reason=str(termination_reason or "failure"),
            )

        plan_memory = update_plan_memory(
            self.state.get("plan_memory", {}),
            decision=decision,
            outcome=outcome,
            blackboard_state=blackboard_state,
        )
        skill_library = rebuild_skill_library(
            blackboard_state.get("entities", {}),
            blackboard_state.get("trigger_zones", {}),
            self.state.get("skill_library", {}),
        )
        skill_library = update_skill_execution_stats(skill_library, decision=decision, outcome=outcome)
        exhaustion_map = exhaustion_snapshot(retries, threshold=retry_limit)

        self.state = {
            "skill_library": skill_library,
            "plan_memory": plan_memory,
            "cooldowns": cooldowns,
            "retries": retries,
            "exhausted": sorted(exhausted_candidates(retries, retry_limit)),
            "exhaustion_map": exhaustion_map,
        }
        self.revision += 1
        version = next_memory_version(self.session_id, round_id, self.revision)
        payload = {"version": version, "revision": self.revision, "state": self.state}
        return MemorySnapshot(
            snapshot_handle=make_handle("snapshot:memory", payload),
            memory_version=version,
            created_round_id=round_id,
            created_pass_id=pass_id,
            state=self.state,
        )
