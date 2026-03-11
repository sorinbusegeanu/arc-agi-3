from __future__ import annotations

import ray

from v3_1.memory.skill_memory import SkillMemoryState


@ray.remote
class MemoryAgent:
    def __init__(self, session_id: str) -> None:
        self.state = SkillMemoryState(session_id=session_id)

    def reconcile(self, *, round_id: int, pass_id: int, blackboard_state: dict, decision: dict | None, outcome: dict | None, retry_limit: int, cooldown_rounds: int):
        return self.state.reconcile(
            round_id=round_id,
            pass_id=pass_id,
            blackboard_state=blackboard_state,
            decision=decision,
            outcome=outcome,
            retry_limit=retry_limit,
            cooldown_rounds=cooldown_rounds,
        )

    def get_state(self) -> dict:
        return dict(self.state.state)
