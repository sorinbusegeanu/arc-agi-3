from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SessionRunMetadata:
    session_id: str
    run_id: str
    game_id: str
    round_id: int
    pass_id: int
    worker_id: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BlackboardSnapshot:
    snapshot_handle: str
    blackboard_version: str
    created_round_id: int
    created_pass_id: int
    material_change: bool
    state: dict[str, Any]
    indexes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemorySnapshot:
    snapshot_handle: str
    memory_version: str
    created_round_id: int
    created_pass_id: int
    state: dict[str, Any]


@dataclass(frozen=True)
class PlanningContext:
    session_id: str
    run_id: str
    game_id: str
    round_id: int
    pass_id: int
    plan_context_id: str
    blackboard_snapshot_handle: str
    memory_snapshot_handle: str
    blackboard_version: str
    memory_version: str
    policy_version: str
    ranker_version: str
    debug: dict[str, Any] = field(default_factory=dict)

