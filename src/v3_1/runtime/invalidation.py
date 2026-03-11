from __future__ import annotations

from v3_1.contracts.messages import InvalidationEvent
from v3_1.contracts.versions import CompatibilityStamp, compatibility_matches, invalidation_metadata


def invalidate_if_needed(
    *,
    stale: CompatibilityStamp,
    current: CompatibilityStamp,
    session_id: str,
    run_id: str,
    game_id: str,
    round_id: int,
    pass_id: int,
    stale_task_ids: list[str] | None = None,
    changed_components: list[str] | None = None,
) -> InvalidationEvent | None:
    if compatibility_matches(stale, current):
        return None
    reason = "blackboard_changed"
    changed_components = list(changed_components or [])
    if stale.blackboard_version != current.blackboard_version:
        changed_components.append("blackboard")
    if stale.memory_version != current.memory_version:
        reason = "memory_changed"
        changed_components.append("memory")
    if stale.policy_version != current.policy_version:
        reason = "policy_changed"
        changed_components.append("policy")
    if stale.ranker_version != current.ranker_version:
        reason = "ranker_changed"
        changed_components.append("ranker")
    metadata = invalidation_metadata(stale=stale, current=current, reason=reason)
    metadata["stale_task_ids"] = list(stale_task_ids or [])
    metadata["changed_components"] = sorted(set(changed_components))
    return InvalidationEvent(
        session_id=session_id,
        run_id=run_id,
        game_id=game_id,
        round_id=round_id,
        pass_id=pass_id,
        stale_plan_context_id=stale.plan_context_id,
        current_plan_context_id=current.plan_context_id,
        blackboard_version=current.blackboard_version,
        memory_version=current.memory_version,
        policy_version=current.policy_version,
        ranker_version=current.ranker_version,
        reason=reason,
        metadata=metadata,
    )
