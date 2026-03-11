from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .schemas import BlackboardStateV2, CandidatePOIV2, ConsequenceRecordV2, ExecutorOutcomeV2, TrajectoryEpisodeV2


def normalize_consequence_class(raw_class: Optional[str]) -> str:
    label = str(raw_class or "").strip()
    mapping = {
        "terminal_like": "terminal_like",
        "progress_like": "progress_like",
        "no_change": "no_change",
        "movement_change": "local_change",
        "object_state_change": "local_change",
        "local_change": "local_change",
        "transition": "global_change",
        "mixed": "global_change",
        "global_change": "global_change",
        "reward_like": "progress_like",
        "target_reached": "progress_like",
        "target_contact": "progress_like",
        "distance_reduced": "progress_like",
        "local_effect_only": "local_change",
        "global_effect_only": "global_change",
        "no_visible_effect": "no_change",
        "no_progress": "no_change",
    }
    return mapping.get(label, "no_change")


def _consequence_is_useful(consequence: ConsequenceRecordV2) -> bool:
    if (
        not consequence.reached
        and not consequence.event_ids
        and not consequence.cause_effect_link_ids
        and not consequence.topology_delta_id
        and consequence.local_change_magnitude <= 0.0
        and not consequence.distance_decreased
    ):
        return False
    if (
        consequence.contact
        and not consequence.reached
        and not consequence.event_ids
        and not consequence.cause_effect_link_ids
        and not consequence.topology_delta_id
        and consequence.local_change_magnitude <= 0.0
        and not consequence.distance_decreased
    ):
        return False
    if consequence.reached or consequence.distance_decreased or consequence.terminal_flag_changed:
        return True
    if consequence.event_ids or consequence.cause_effect_link_ids or consequence.topology_delta_id:
        return True
    if consequence.local_change_magnitude > 0.0:
        return True
    return False


def _outcome_is_useful(outcome: ExecutorOutcomeV2) -> bool:
    progress_flat = not outcome.target_progress or max(outcome.target_progress) == min(outcome.target_progress)
    no_distance_decrease = not any(record.distance_decreased for record in outcome.consequence_records)
    if outcome.reached:
        return True
    if outcome.outcome_summary in {"contact_no_effect", "contact_no_reach", "contact_reached_boundary_without_route_progress", "route_stall", "blocked", "no_progress"} and not any(
        record.event_ids or record.topology_delta_id for record in outcome.consequence_records
    ) and all(record.local_change_magnitude <= 0.0 for record in outcome.consequence_records) and progress_flat and no_distance_decrease:
        return False
    for consequence in outcome.consequence_records:
        if _consequence_is_useful(consequence):
            return True
    return False


@dataclass(frozen=True)
class RoundMetricsV2:
    unique_states: int
    unique_pois: int
    route_success_rate: float
    useful_change_rate: float
    contact_success_rate: float
    # Historical fields kept here as comments while the metric surface is intentionally narrowed:
    # episodes_collected: int
    # states_observed: int
    # invalid_state_count: int
    # duplicate_state_count: int
    # state_hash_coverage_rate: float
    # state_hash_diagnostic: bool
    # reachable_pois: int
    # poi_first_contacts: int
    # poi_interactions: int
    # hypothesis_resolution_rate: float
    # false_poi_rate: float
    # exploit_switch_rate: float
    # target_progress_mean: float
    # target_progress_median: float
    # useful_screen_change_rate: float
    # no_effect_probe_rate: float
    # candidate_avatar_count: int
    # candidate_poi_count: int
    # reachable_poi_count: int
    # target_selection_count_by_mode: Dict[str, int]
    # target_reached_count: int
    # interaction_count: int
    # false_poi_count: int
    # hypothesis_promoted_count: int
    # hypothesis_demoted_count: int
    # stagnant_probe_rate: float
    # action_semantics_coverage: float
    # avatar_track_confirmation_rate: float
    # post_contact_event_capture_rate: float
    # same_area_causal_link_count: int
    # cross_area_causal_link_count: int
    # mechanic_hypothesis_promotion_count: int
    # contradiction_rate: float
    # discriminating_probe_usage_rate: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "unique_states": int(self.unique_states),
            "unique_pois": int(self.unique_pois),
            "route_success_rate": float(self.route_success_rate),
            "useful_change_rate": float(self.useful_change_rate),
            "contact_success_rate": float(self.contact_success_rate),
        }


def compute_round_metrics(
    episodes: List[TrajectoryEpisodeV2],
    poi_table: List[CandidatePOIV2],
    reachability_table: Dict[str, str],
    consequences: List[ConsequenceRecordV2],
    controller_modes: List[str],
    avatar_count: int,
    target_progress: Optional[List[float]] = None,
    blackboard: Optional[BlackboardStateV2] = None,
    executor_outcomes: Optional[List[ExecutorOutcomeV2]] = None,
) -> RoundMetricsV2:
    target_reached_count = sum(1 for c in consequences if c.reached)
    interaction_count = sum(1 for c in consequences if c.contact)
    useful_count = sum(1 for consequence in consequences if _consequence_is_useful(consequence))
    useful_denominator = len(consequences)
    outcome_reached_count = 0
    outcome_contact_count = 0
    for outcome in list(executor_outcomes or []):
        if outcome.reached:
            outcome_reached_count += 1
        if outcome.contact:
            outcome_contact_count += 1
        if outcome.consequence_records:
            continue
        if outcome.actions:
            useful_denominator += 1
            if _outcome_is_useful(outcome):
                useful_count += 1
    target_reached_count = max(target_reached_count, outcome_reached_count)
    interaction_count = max(interaction_count, outcome_contact_count)
    useful_change_rate = float(useful_count) / float(useful_denominator) if useful_denominator > 0 else 0.0
    unique_state_hashes = set()
    for ep in episodes:
        for step in ep.steps:
            for state_hash in (step.pre_state_hash, step.post_state_hash):
                if state_hash:
                    unique_state_hashes.add(state_hash)
    denominator = float(max(1, len(controller_modes)))
    route_success_rate = float(target_reached_count) / denominator
    contact_success_rate = float(interaction_count) / denominator
    return RoundMetricsV2(
        unique_states=len(unique_state_hashes),
        unique_pois=len(poi_table),
        route_success_rate=route_success_rate,
        useful_change_rate=useful_change_rate,
        contact_success_rate=contact_success_rate,
    )
