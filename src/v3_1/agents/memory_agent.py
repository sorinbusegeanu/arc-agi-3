from __future__ import annotations

import ray

from v3_1.contracts.messages import PersistentMemoryLoadResult, PersistentMemoryFlushRequest
from v3_1.memory.skill_memory import SkillMemoryState


@ray.remote
class MemoryAgent:
    def __init__(self, session_id: str, *, load_persistent_priors_on_session_start: bool = True) -> None:
        self.state = SkillMemoryState(session_id=session_id)
        self.load_persistent_priors_on_session_start = load_persistent_priors_on_session_start

    def reconcile(self, *, round_id: int, pass_id: int, blackboard_state: dict, decision: dict | None, outcome: dict | None, retry_limit: int, cooldown_rounds: int):
        snapshot = self.state.reconcile(
            round_id=round_id,
            pass_id=pass_id,
            blackboard_state=blackboard_state,
            decision=decision,
            outcome=outcome,
            retry_limit=retry_limit,
            cooldown_rounds=cooldown_rounds,
        )
        return snapshot

    def load_persistent_priors(self, load_result: PersistentMemoryLoadResult | dict | None) -> dict:
        if self.load_persistent_priors_on_session_start:
            self.state.load_persistent_priors(load_result)
        return {"loaded": bool(load_result), "prior_keys": sorted(self.state.durable_priors.keys())}

    def build_flush_request(self, *, run_id: str, game_id: str, round_id: int, pass_id: int, flush_id: str, session_snapshot_path: str | None = None, metadata: dict | None = None) -> PersistentMemoryFlushRequest | None:
        batch = self.state.drain_durable_updates(run_id=run_id, game_id=game_id, round_id=round_id, pass_id=pass_id)
        if batch is None:
            return None
        return PersistentMemoryFlushRequest(
            session_id=self.state.session_id,
            run_id=run_id,
            game_id=game_id,
            flush_id=flush_id,
            batch=batch,
            session_snapshot_path=session_snapshot_path,
            metadata=metadata or {},
        )

    def get_state(self) -> dict:
        return dict(self.state.state)
