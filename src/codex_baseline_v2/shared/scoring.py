from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import ScoringConfigV2


@dataclass(frozen=True)
class POIRankInputs:
    info_gain: float
    confidence: float
    reachability_score: float
    route_confidence: float = 0.0
    access_confidence: float = 0.0
    event_novelty: float = 0.0
    contradiction_penalty: float = 0.0
    cross_area_mechanic_opportunity: float = 0.0
    stale_target_penalty: float = 0.0


def poi_rank_score(inputs: POIRankInputs, cfg: ScoringConfigV2) -> float:
    return (
        cfg.poi_rank_weight_info_gain * inputs.info_gain
        + cfg.poi_rank_weight_confidence * inputs.confidence
        + cfg.poi_rank_weight_reachability * inputs.reachability_score
        + inputs.route_confidence
        + inputs.access_confidence
        + inputs.event_novelty
        + inputs.cross_area_mechanic_opportunity
        - inputs.contradiction_penalty
        - inputs.stale_target_penalty
    )


def hypothesis_info_gain_score(info_gain: float, cfg: ScoringConfigV2) -> float:
    return cfg.hypothesis_info_gain_weight * info_gain


def controller_target_score(base_score: float, cfg: ScoringConfigV2) -> float:
    return cfg.controller_target_weight * base_score


def executor_progress_score(progress: float, cfg: ScoringConfigV2) -> float:
    return cfg.executor_progress_weight * progress


def trajectory_consequence_score(local_change: float, global_change: float, cfg: ScoringConfigV2) -> float:
    return cfg.consequence_weight_local * local_change + cfg.consequence_weight_global * global_change


def reachability_to_score(status: str) -> float:
    if status in {"reachable_now", "reachable"}:
        return 1.0
    if status == "uncertain":
        return 0.5
    if status == "cross_area_only":
        return 0.25
    if status == "blocked":
        return 0.1
    return 0.0
