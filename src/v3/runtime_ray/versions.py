from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict


def new_blackboard_version(game_id: str, round_id: int) -> str:
    return f"bb:{game_id}:{round_id}:{int(time.time() * 1000)}"


def new_memory_version(game_id: str, round_id: int) -> str:
    return f"mem:{game_id}:{round_id}:{int(time.time() * 1000)}"


def new_plan_context_id(
    *,
    blackboard_version: str,
    memory_version: str,
    policy_version: str,
    ranker_version: str,
    session_id: str,
    game_id: str,
    round_id: int,
) -> str:
    payload = {
        "blackboard_version": blackboard_version,
        "memory_version": memory_version,
        "policy_version": policy_version,
        "ranker_version": ranker_version,
        "session_id": session_id,
        "game_id": game_id,
        "round_id": round_id,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"planctx:{game_id}:{round_id}:{digest}"


def context_matches(helper_output_context_id: str, current_context_id: str) -> bool:
    return str(helper_output_context_id) == str(current_context_id)


def package_invalidation_metadata(*, old_context_id: str, new_context_id: str, reason: str, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "old_context_id": old_context_id,
        "new_context_id": new_context_id,
        "reason": reason,
        "extra": dict(extra or {}),
        "at_ms": int(time.time() * 1000),
    }
