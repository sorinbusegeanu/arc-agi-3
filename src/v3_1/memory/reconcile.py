from __future__ import annotations

from v3_1.contracts.snapshots import MemorySnapshot
from v3_1.memory.skill_memory import SkillMemoryState


def export_memory_snapshot(snapshot: MemorySnapshot) -> dict:
    return {
        "snapshot_handle": snapshot.snapshot_handle,
        "memory_version": snapshot.memory_version,
        "created_round_id": snapshot.created_round_id,
        "created_pass_id": snapshot.created_pass_id,
        "state": snapshot.state,
    }


def import_memory_snapshot(session_id: str, snapshot_payload: dict) -> SkillMemoryState:
    snapshot = MemorySnapshot(
        snapshot_handle=str(snapshot_payload["snapshot_handle"]),
        memory_version=str(snapshot_payload["memory_version"]),
        created_round_id=int(snapshot_payload["created_round_id"]),
        created_pass_id=int(snapshot_payload["created_pass_id"]),
        state=dict(snapshot_payload["state"]),
    )
    return SkillMemoryState.from_snapshot(session_id, snapshot)


def reconcile_memory(memory: SkillMemoryState, *, round_id: int, pass_id: int, blackboard_state: dict, decision: dict | None, outcome: dict | None, retry_limit: int, cooldown_rounds: int) -> MemorySnapshot:
    return memory.reconcile(
        round_id=round_id,
        pass_id=pass_id,
        blackboard_state=blackboard_state,
        decision=decision,
        outcome=outcome,
        retry_limit=retry_limit,
        cooldown_rounds=cooldown_rounds,
    )
