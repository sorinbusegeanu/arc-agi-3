from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RawStep:
    session_id: str
    run_id: str
    game_id: str
    episode_id: str
    step_idx: int
    observation: Any
    action: Any
    reward: float | None
    done: bool
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawEpisode:
    session_id: str
    run_id: str
    game_id: str
    round_id: int
    pass_id: int
    episode_id: str
    mode: str
    worker_id: str
    steps: tuple[RawStep, ...]
    total_reward: float = 0.0
    done: bool = False
    won: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalyzedEpisode:
    session_id: str
    run_id: str
    game_id: str
    round_id: int
    pass_id: int
    episode_id: str
    raw_episode_id: str
    summary: dict[str, Any]
    objects: tuple[dict[str, Any], ...]
    avatar_tracks: tuple[dict[str, Any], ...]
    points_of_interest: tuple[dict[str, Any], ...]
    areas: tuple[dict[str, Any], ...]
    motion: tuple[dict[str, Any], ...]
    blackboard_deltas: tuple["BlackboardDelta", ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BlackboardDelta:
    session_id: str
    run_id: str
    game_id: str
    round_id: int
    pass_id: int
    episode_id: str
    delta_id: str
    areas: tuple[dict[str, Any], ...] = ()
    entities: tuple[dict[str, Any], ...] = ()
    consequences: tuple[dict[str, Any], ...] = ()
    trigger_zones: tuple[dict[str, Any], ...] = ()
    topology_nodes: tuple[dict[str, Any], ...] = ()
    topology_edges: tuple[dict[str, Any], ...] = ()
    evidence: tuple[str, ...] = ()
    material_change: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HelperTaskRequest:
    session_id: str
    run_id: str
    game_id: str
    round_id: int
    pass_id: int
    helper_mode: str
    plan_context_id: str
    blackboard_version: str
    memory_version: str
    policy_version: str
    ranker_version: str
    candidate_ids: tuple[str, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HelperTaskResult:
    session_id: str
    run_id: str
    game_id: str
    round_id: int
    pass_id: int
    helper_mode: str
    plan_context_id: str
    blackboard_version: str
    memory_version: str
    policy_version: str
    ranker_version: str
    proposal_id: str
    proposals: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannerDecision:
    session_id: str
    run_id: str
    game_id: str
    round_id: int
    pass_id: int
    plan_context_id: str
    blackboard_version: str
    memory_version: str
    policy_version: str
    ranker_version: str
    selected_candidate_id: str | None
    selected_action: dict[str, Any] | None
    ranked_candidates: tuple[dict[str, Any], ...]
    rationale: str
    helper_proposal_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutorRequest:
    session_id: str
    run_id: str
    game_id: str
    round_id: int
    pass_id: int
    plan_context_id: str
    candidate_id: str | None
    action: dict[str, Any] | None
    max_steps: int
    mode: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutorOutcome:
    session_id: str
    run_id: str
    game_id: str
    round_id: int
    pass_id: int
    plan_context_id: str
    candidate_id: str | None
    episode: RawEpisode
    success: bool
    termination_reason: str
    reward_delta: float = 0.0
    outcome: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PersistenceRequest:
    session_id: str
    run_id: str
    game_id: str
    round_id: int
    pass_id: int
    artifact_kind: str
    artifact_name: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PersistenceResult:
    session_id: str
    run_id: str
    game_id: str
    round_id: int
    pass_id: int
    artifact_kind: str
    artifact_name: str
    location: str
    bytes_written: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InvalidationEvent:
    session_id: str
    run_id: str
    game_id: str
    round_id: int
    pass_id: int
    stale_plan_context_id: str
    current_plan_context_id: str
    blackboard_version: str
    memory_version: str
    policy_version: str
    ranker_version: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

