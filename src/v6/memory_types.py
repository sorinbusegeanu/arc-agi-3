from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class M0EpisodeSummary:
    game_id: str
    sampler: str
    seed: int
    episode_id: int
    level_id: str | None
    start_step: int
    end_step: int
    steps_total: int
    terminal_observed: bool
    terminal_type: str | None
    success_observed: bool | None
    unique_state_signatures: int
    repeated_state_count: int
    blocked_or_no_change_count: int
    non_preserve_count: int
    action_counts: dict[str, int]
    trajectory_cost: int
    loop_ratio: float
    wasted_action_ratio: float
    steps_to_terminal: int | None = None
    normalized_solve_efficiency: float | None = None
    equivalent_outcome_cost_gap: float | None = None
    diagnostic_only: bool = True
    notes: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class M1Contingency:
    contingency_id: str
    game_id: str
    sampler_scope: str
    context_signature: list[str]
    action: int
    outcome_signature: str
    support_count: int
    total_count: int
    prediction_accuracy: float
    prediction_error_rate: float
    entropy: float
    confidence: float
    first_seen_step: int
    last_seen_step: int
    example_episode_ids: list[int]
    terminal_effect_candidate: bool
    future_option_motif_candidate: str
    discovered: bool
    notes: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class M2TransformationFamily:
    family_id: str
    family_label_candidate: str
    games_present: list[str]
    samplers_present: list[str]
    contingency_ids: list[str]
    support_count: int
    mean_prediction_accuracy: float
    mean_context_lift: float
    dominant_outcome_signature: str
    outcome_signature_distribution: dict[str, int]
    motif_candidate_distribution: dict[str, int]
    family_coherence: float
    compression_ratio: float
    cross_game_presence: int
    stable: bool
    examples: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class M3RoleCandidate:
    role_id: str
    role_label_candidate: str
    member_family_ids: list[str]
    games_present: list[str]
    game_families_present: list[str]
    support_count: int
    cross_game_support: int
    cross_game_family_support: int
    role_consistency_score: float
    mean_neighborhood_similarity: float
    mean_family_coherence: float
    dominant_motif_profile: dict[str, float]
    incoming_edge_profile: dict[str, int]
    outgoing_edge_profile: dict[str, int]
    future_option_effect_profile: dict[str, float]
    transfer_readiness_score: float
    label_evidence: dict[str, Any] = field(default_factory=dict)
    examples: list[dict[str, Any]] = field(default_factory=list)
    status: str = "weak"
    notes: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
