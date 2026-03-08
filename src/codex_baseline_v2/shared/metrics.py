from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .schemas import CandidatePOIV2, ConsequenceRecordV2, TrajectoryEpisodeV2


@dataclass(frozen=True)
class RoundMetricsV2:
    episodes_collected: int
    states_observed: int
    unique_states: int
    invalid_state_count: int
    duplicate_state_count: int
    state_hash_coverage_rate: float
    state_hash_diagnostic: bool
    unique_pois: int
    reachable_pois: int
    poi_first_contacts: int
    poi_interactions: int
    hypothesis_resolution_rate: float
    false_poi_rate: float
    exploit_switch_rate: float
    route_success_rate: float
    target_progress_mean: float
    target_progress_median: float
    useful_screen_change_rate: float
    no_effect_probe_rate: float
    candidate_avatar_count: int
    candidate_poi_count: int
    reachable_poi_count: int
    target_selection_count_by_mode: Dict[str, int]
    target_reached_count: int
    interaction_count: int
    false_poi_count: int
    hypothesis_promoted_count: int
    hypothesis_demoted_count: int
    useful_change_rate: float
    stagnant_probe_rate: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "episodes_collected": int(self.episodes_collected),
            "states_observed": int(self.states_observed),
            "unique_states": int(self.unique_states),
            "invalid_state_count": int(self.invalid_state_count),
            "duplicate_state_count": int(self.duplicate_state_count),
            "state_hash_coverage_rate": float(self.state_hash_coverage_rate),
            "state_hash_diagnostic": bool(self.state_hash_diagnostic),
            "unique_pois": int(self.unique_pois),
            "reachable_pois": int(self.reachable_pois),
            "poi_first_contacts": int(self.poi_first_contacts),
            "poi_interactions": int(self.poi_interactions),
            "hypothesis_resolution_rate": float(self.hypothesis_resolution_rate),
            "false_poi_rate": float(self.false_poi_rate),
            "exploit_switch_rate": float(self.exploit_switch_rate),
            "route_success_rate": float(self.route_success_rate),
            "target_progress_mean": float(self.target_progress_mean),
            "target_progress_median": float(self.target_progress_median),
            "useful_screen_change_rate": float(self.useful_screen_change_rate),
            "no_effect_probe_rate": float(self.no_effect_probe_rate),
            "candidate_avatar_count": int(self.candidate_avatar_count),
            "candidate_poi_count": int(self.candidate_poi_count),
            "reachable_poi_count": int(self.reachable_poi_count),
            "target_selection_count_by_mode": dict(self.target_selection_count_by_mode),
            "target_reached_count": int(self.target_reached_count),
            "interaction_count": int(self.interaction_count),
            "false_poi_count": int(self.false_poi_count),
            "hypothesis_promoted_count": int(self.hypothesis_promoted_count),
            "hypothesis_demoted_count": int(self.hypothesis_demoted_count),
            "useful_change_rate": float(self.useful_change_rate),
            "stagnant_probe_rate": float(self.stagnant_probe_rate),
        }


def compute_round_metrics(
    episodes: List[TrajectoryEpisodeV2],
    poi_table: List[CandidatePOIV2],
    reachability_table: Dict[str, str],
    consequences: List[ConsequenceRecordV2],
    controller_modes: List[str],
    avatar_count: int,
    target_progress: Optional[List[float]] = None,
) -> RoundMetricsV2:
    reachable_count = sum(1 for p in poi_table if reachability_table.get(p.poi_id) == "reachable_now")
    target_reached_count = sum(1 for c in consequences if c.reached)
    interaction_count = sum(1 for c in consequences if c.contact)
    useful_change_rate = 0.0
    stagnant_probe_rate = 0.0
    if consequences:
        useful_change = sum(1 for c in consequences if c.consequence_class in {"local_change", "global_change", "progress_like", "terminal_like"})
        useful_change_rate = float(useful_change) / float(len(consequences))
        stagnant = sum(1 for c in consequences if c.consequence_class == "no_change")
        stagnant_probe_rate = float(stagnant) / float(len(consequences))
    mode_counts: Dict[str, int] = {}
    for mode in controller_modes:
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    false_poi_count = sum(1 for p in poi_table if p.confidence < 0.2)
    unique_state_hashes = set()
    invalid_state_count = 0
    duplicate_state_count = 0
    valid_state_count = 0
    for ep in episodes:
        for step in ep.steps:
            for state_hash in (step.pre_state_hash, step.post_state_hash):
                if state_hash:
                    valid_state_count += 1
                    if state_hash in unique_state_hashes:
                        duplicate_state_count += 1
                    else:
                        unique_state_hashes.add(state_hash)
                else:
                    invalid_state_count += 1
    total_states = valid_state_count + invalid_state_count
    coverage_rate = float(valid_state_count) / float(total_states) if total_states else 0.0
    state_hash_diagnostic = bool(valid_state_count > 0 and len(unique_state_hashes) == 0)
    progress_vals = target_progress or []
    progress_vals_sorted = sorted(progress_vals) if progress_vals else []
    progress_mean = float(sum(progress_vals) / max(1, len(progress_vals))) if progress_vals else 0.0
    progress_median = 0.0
    if progress_vals_sorted:
        mid = len(progress_vals_sorted) // 2
        if len(progress_vals_sorted) % 2 == 1:
            progress_median = float(progress_vals_sorted[mid])
        else:
            progress_median = float(progress_vals_sorted[mid - 1] + progress_vals_sorted[mid]) / 2.0
    exploit_switch_rate = 0.0
    if controller_modes:
        exploit_switch_rate = float(sum(1 for m in controller_modes if m == "exploit_route")) / float(len(controller_modes))
    route_success_rate = float(target_reached_count) / float(max(1, len(controller_modes)))
    hypothesis_resolution_rate = 0.0
    return RoundMetricsV2(
        episodes_collected=len(episodes),
        states_observed=valid_state_count,
        unique_states=len(unique_state_hashes),
        invalid_state_count=invalid_state_count,
        duplicate_state_count=duplicate_state_count,
        state_hash_coverage_rate=coverage_rate,
        state_hash_diagnostic=state_hash_diagnostic,
        unique_pois=len(poi_table),
        reachable_pois=reachable_count,
        poi_first_contacts=interaction_count,
        poi_interactions=interaction_count,
        hypothesis_resolution_rate=hypothesis_resolution_rate,
        false_poi_rate=float(false_poi_count) / float(max(1, len(poi_table))),
        exploit_switch_rate=exploit_switch_rate,
        route_success_rate=route_success_rate,
        target_progress_mean=progress_mean,
        target_progress_median=progress_median,
        useful_screen_change_rate=useful_change_rate,
        no_effect_probe_rate=stagnant_probe_rate,
        candidate_avatar_count=int(avatar_count),
        candidate_poi_count=len(poi_table),
        reachable_poi_count=reachable_count,
        target_selection_count_by_mode=mode_counts,
        target_reached_count=target_reached_count,
        interaction_count=interaction_count,
        false_poi_count=false_poi_count,
        hypothesis_promoted_count=0,
        hypothesis_demoted_count=0,
        useful_change_rate=useful_change_rate,
        stagnant_probe_rate=stagnant_probe_rate,
    )
