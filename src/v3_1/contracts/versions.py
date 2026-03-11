from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from v3_1.utils.ids import stable_digest


@dataclass(frozen=True)
class VersionVector:
    blackboard_version: str
    memory_version: str
    policy_version: str
    ranker_version: str


@dataclass(frozen=True)
class CompatibilityStamp:
    plan_context_id: str
    blackboard_version: str
    memory_version: str
    policy_version: str
    ranker_version: str
    extra: dict[str, Any] = field(default_factory=dict)


def next_blackboard_version(session_id: str, round_id: int, revision: int) -> str:
    return f"bb:{session_id}:{round_id}:{revision}"


def next_memory_version(session_id: str, round_id: int, revision: int) -> str:
    return f"mem:{session_id}:{round_id}:{revision}"


def build_plan_context_id(
    *,
    session_id: str,
    game_id: str,
    round_id: int,
    blackboard_version: str,
    memory_version: str,
    policy_version: str,
    ranker_version: str,
) -> str:
    payload = {
        "session_id": session_id,
        "game_id": game_id,
        "round_id": round_id,
        "blackboard_version": blackboard_version,
        "memory_version": memory_version,
        "policy_version": policy_version,
        "ranker_version": ranker_version,
    }
    return f"planctx:{game_id}:{round_id}:{stable_digest(payload)}"


def compatibility_matches(stamp: CompatibilityStamp, current: CompatibilityStamp) -> bool:
    return (
        stamp.plan_context_id == current.plan_context_id
        and stamp.blackboard_version == current.blackboard_version
        and stamp.memory_version == current.memory_version
        and stamp.policy_version == current.policy_version
        and stamp.ranker_version == current.ranker_version
    )


def invalidation_metadata(*, stale: CompatibilityStamp, current: CompatibilityStamp, reason: str) -> dict[str, Any]:
    return {
        "reason": reason,
        "stale_plan_context_id": stale.plan_context_id,
        "current_plan_context_id": current.plan_context_id,
        "stale_versions": {
            "blackboard_version": stale.blackboard_version,
            "memory_version": stale.memory_version,
            "policy_version": stale.policy_version,
            "ranker_version": stale.ranker_version,
        },
        "current_versions": {
            "blackboard_version": current.blackboard_version,
            "memory_version": current.memory_version,
            "policy_version": current.policy_version,
            "ranker_version": current.ranker_version,
        },
    }

