from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class V2MetricsReport:
    states_observed: int
    unique_states: int
    invalid_state_count: int
    duplicate_state_count: int
    state_hash_coverage_rate: float
    state_hash_diagnostic: bool
    episodes_per_round_by_mode: Dict[str, int]
    target_attempt_count: int
    target_reach_rate_by_poi_type: Dict[str, float]
    target_contact_rate: float
    target_interaction_rate: float
    route_block_rate: float
    route_stall_rate: float
    distance_reduction_rate: float
    informative_probe_rate: float
    repeated_false_target_rate: float
    avatar_hypothesis_stability: float
    traversable_map_coverage: float
    blackboard_change_rate: float
    resume_success_count: int
    action_semantics_coverage: float = 0.0
    avatar_track_confirmation_rate: float = 0.0
    route_success_rate: float = 0.0
    contact_success_rate: float = 0.0
    post_contact_event_capture_rate: float = 0.0
    same_area_causal_link_count: int = 0
    cross_area_causal_link_count: int = 0
    mechanic_hypothesis_promotion_count: int = 0
    contradiction_rate: float = 0.0
    discriminating_probe_usage_rate: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "states_observed": int(self.states_observed),
            "unique_states": int(self.unique_states),
            "invalid_state_count": int(self.invalid_state_count),
            "duplicate_state_count": int(self.duplicate_state_count),
            "state_hash_coverage_rate": float(self.state_hash_coverage_rate),
            "state_hash_diagnostic": bool(self.state_hash_diagnostic),
            "episodes_per_round_by_mode": dict(self.episodes_per_round_by_mode),
            "target_attempt_count": int(self.target_attempt_count),
            "target_reach_rate_by_poi_type": dict(self.target_reach_rate_by_poi_type),
            "target_contact_rate": float(self.target_contact_rate),
            "target_interaction_rate": float(self.target_interaction_rate),
            "route_block_rate": float(self.route_block_rate),
            "route_stall_rate": float(self.route_stall_rate),
            "distance_reduction_rate": float(self.distance_reduction_rate),
            "informative_probe_rate": float(self.informative_probe_rate),
            "repeated_false_target_rate": float(self.repeated_false_target_rate),
            "avatar_hypothesis_stability": float(self.avatar_hypothesis_stability),
            "traversable_map_coverage": float(self.traversable_map_coverage),
            "blackboard_change_rate": float(self.blackboard_change_rate),
            "resume_success_count": int(self.resume_success_count),
            "action_semantics_coverage": float(self.action_semantics_coverage),
            "avatar_track_confirmation_rate": float(self.avatar_track_confirmation_rate),
            "route_success_rate": float(self.route_success_rate),
            "contact_success_rate": float(self.contact_success_rate),
            "post_contact_event_capture_rate": float(self.post_contact_event_capture_rate),
            "same_area_causal_link_count": int(self.same_area_causal_link_count),
            "cross_area_causal_link_count": int(self.cross_area_causal_link_count),
            "mechanic_hypothesis_promotion_count": int(self.mechanic_hypothesis_promotion_count),
            "contradiction_rate": float(self.contradiction_rate),
            "discriminating_probe_usage_rate": float(self.discriminating_probe_usage_rate),
        }
