from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RawStep:
    game_id: str
    episode_id: str
    step_idx: int
    action: Dict[str, Any]
    reward: Optional[float]
    done: bool
    observation: Any
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawEpisode:
    game_id: str
    episode_id: str
    round_id: int
    mode: str
    env_worker_id: str
    steps: List[RawStep]
    done: bool
    win: bool
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalyzedEpisode:
    game_id: str
    episode_id: str
    round_id: int
    analyzed_episode: Dict[str, Any]
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BlackboardMergeRequest:
    game_id: str
    round_id: int
    analyzed_episodes: List[AnalyzedEpisode]
    prior_blackboard: Optional[Dict[str, Any]] = None
    merge_reason: str = "round_update"


@dataclass(frozen=True)
class BlackboardMergeResult:
    game_id: str
    round_id: int
    blackboard_version: str
    snapshot_ref: str
    blackboard: Dict[str, Any]
    merge_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryReconcileRequest:
    game_id: str
    round_id: int
    blackboard_snapshot_ref: Optional[str]
    blackboard: Optional[Dict[str, Any]]
    executor_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    plan_decision: Optional[Dict[str, Any]] = None
    merge_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryReconcileResult:
    game_id: str
    round_id: int
    memory_version: str
    snapshot_ref: str
    skills: List[Dict[str, Any]]
    skill_executions: List[Dict[str, Any]]
    plan_memory: Dict[str, Any]
    reconcile_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanningContextSnapshot:
    session_id: str
    game_id: str
    round_id: int
    blackboard_version: str
    memory_version: str
    policy_version: str
    ranker_version: str
    plan_context_id: str
    blackboard_snapshot_ref: str
    memory_snapshot_ref: str
    accepted_at_ms: int
    invalidation_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HelperTaskRequest:
    game_id: str
    round_id: int
    helper_mode: str
    planner_state: Dict[str, Any]
    candidate_skill_ids: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateBatch:
    plan_context_id: str
    helper_mode: str
    candidate_skill_ids: List[str]
    proposals: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class RouteAnalysisResult:
    plan_context_id: str
    candidate_skill_ids: List[str]
    route_features: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreFeatureBatch:
    plan_context_id: str
    candidate_skill_ids: List[str]
    feature_rows: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class HypothesisProposalBatch:
    plan_context_id: str
    proposal_type: str
    proposals: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class PlannerDecision:
    game_id: str
    round_id: int
    plan_context_id: str
    selected_skill_id: Optional[str]
    selected_plan_node_id: Optional[str]
    selected_instruction: Optional[Dict[str, Any]]
    planner_reason: str
    plan_nodes: List[Dict[str, Any]]
    plan_result: Optional[Dict[str, Any]]
    helper_refs: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExecutorRequest:
    game_id: str
    round_id: int
    episode_idx: int
    mode: str
    selected_skill_id: Optional[str]
    selected_plan_node_id: Optional[str]
    instruction: Optional[Dict[str, Any]]
    max_steps: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutorOutcome:
    game_id: str
    round_id: int
    episode_idx: int
    raw_episode: RawEpisode
    execution_outcome: Dict[str, Any]
    negative_planning_feedback: bool = False
    termination_reason: Optional[str] = None


@dataclass(frozen=True)
class PersistenceRequest:
    game_id: str
    round_id: int
    artifact_family: str
    payload: Dict[str, Any]
    ordering_key: str


@dataclass(frozen=True)
class InvalidationEvent:
    game_id: str
    round_id: int
    superseded_plan_context_id: str
    current_plan_context_id: str
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
