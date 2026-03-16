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
    snapshot_kind: str = "working_memory"
    durable_checkpoint_id: str | None = None


@dataclass(frozen=True)
class DurableMemoryCheckpoint:
    checkpoint_handle: str
    db_path: str
    source_memory_version: str
    created_round_id: int
    created_pass_id: int
    metadata: dict[str, Any] = field(default_factory=dict)


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
    mechanic_graph_snapshot_handle: str | None
    blackboard_version: str
    memory_version: str
    mechanic_graph_version: str | None
    deterministic_hypotheses_handle: str | None
    llm_hypotheses_handle: str | None
    hypothesis_registry_snapshot_handle: str | None
    policy_version: str
    ranker_version: str
    durable_memory_checkpoint_handle: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)
