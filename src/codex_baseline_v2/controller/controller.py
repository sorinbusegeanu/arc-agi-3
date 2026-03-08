from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from codex_baseline_v2.shared.config import ControllerConfigV2, ScoringConfigV2
from codex_baseline_v2.shared.scoring import POIRankInputs, controller_target_score, poi_rank_score, reachability_to_score
from codex_baseline_v2.shared.schemas import BlackboardStateV2, CandidatePOIV2, ControllerInstructionV2, SCHEMA_VERSION
from codex_baseline_v2.shared.state_identity import canonical_state_identity
from codex_baseline_v2.shared.utils import BBox


def _reachability_map(blackboard: BlackboardStateV2) -> Dict[str, str]:
    return {r.poi_id: r.status for r in blackboard.reachability_table}


def _rank_pois(
    pois: List[CandidatePOIV2],
    reachability: Dict[str, str],
    scoring_cfg: ScoringConfigV2,
    recent_failures: Dict[str, float],
) -> List[Tuple[float, CandidatePOIV2]]:
    ranked = []
    for poi in pois:
        score = poi_rank_score(
            POIRankInputs(
                info_gain=poi.expected_information_gain,
                confidence=poi.confidence,
                reachability_score=reachability_to_score(reachability.get(poi.poi_id, "uncertain")),
            ),
            scoring_cfg,
        )
        score -= recent_failures.get(poi.poi_id, 0.0)
        ranked.append((score, poi))
    ranked.sort(key=lambda r: r[0], reverse=True)
    return ranked


def select_instruction(
    blackboard: BlackboardStateV2,
    cfg: ControllerConfigV2,
    scoring_cfg: ScoringConfigV2,
    round_id: int,
) -> ControllerInstructionV2:
    rng = random.Random(cfg.random_seed + round_id)
    state_ref = None
    last_obs = blackboard.metadata.get("last_observation") if isinstance(blackboard.metadata, dict) else None
    if isinstance(last_obs, list):
        identity = canonical_state_identity(last_obs, include_payload=False)
        if identity.get("valid") and identity.get("state_hash"):
            state_ref = str(identity.get("state_hash"))[:8]
    recent_failures: Dict[str, float] = {}
    for consequence in blackboard.consequence_table[-20:]:
        if not consequence.target_poi_id:
            continue
        if consequence.consequence_class == "no_change" and not consequence.distance_decreased and not consequence.reached and not consequence.contact:
            recent_failures[consequence.target_poi_id] = recent_failures.get(consequence.target_poi_id, 0.0) + 0.75
    history = blackboard.metadata.get("instruction_history", []) if isinstance(blackboard.metadata, dict) else []
    for item in history[-5:]:
        poi_id = item.get("target_poi_id") if isinstance(item, dict) else None
        if poi_id and item.get("outcome") == "no_progress":
            recent_failures[poi_id] = recent_failures.get(poi_id, 0.0) + 0.5
    if round_id == 0 or rng.random() < cfg.unguided_probe_fraction:
        return ControllerInstructionV2(
            schema_version=SCHEMA_VERSION,
            game_id=blackboard.game_id,
            round_id=round_id,
            instruction_id=f"round{round_id:03d}:unguided_probe",
            mode="unguided_probe",
            target_poi_id=None,
            target_region=None,
            target_type=None,
            target_geometry=None,
            target_source_round=None,
            rationale=f"preserve_unguided_quota state={state_ref}" if state_ref else "preserve_unguided_quota",
            progress_metric="steps_elapsed",
            stop_condition="budget",
            ranked_alternatives=[],
        )
    reachability = _reachability_map(blackboard)
    ranked = _rank_pois(blackboard.poi_table, reachability, scoring_cfg, recent_failures)
    if not ranked:
        return ControllerInstructionV2(
            schema_version=SCHEMA_VERSION,
            game_id=blackboard.game_id,
            round_id=round_id,
            instruction_id=f"round{round_id:03d}:discriminating_probe",
            mode="discriminating_probe",
            target_poi_id=None,
            target_region=None,
            target_type=None,
            target_geometry=None,
            target_source_round=None,
            rationale="no_poi_available",
            progress_metric="steps_elapsed",
            stop_condition="budget",
            ranked_alternatives=[],
        )
    rejected: List[str] = []
    selected: Optional[CandidatePOIV2] = None
    selected_score = 0.0
    for score, candidate in ranked:
        status = reachability.get(candidate.poi_id, "uncertain")
        skip_reason = None
        if candidate.object_class == "hud_like" or status == "likely_hud":
            skip_reason = "likely_hud"
        elif candidate.reachable_now == "exhausted_for_now":
            skip_reason = "exhausted_for_now"
        elif recent_failures.get(candidate.poi_id, 0.0) >= 1.5:
            skip_reason = "recent_no_progress"
        elif candidate.bbox.area() <= 0:
            skip_reason = "no_geometry"
        elif status in {"insufficient_evidence", "unknown_traversable", "unknown_avatar"} and candidate.expected_information_gain < 0.6:
            skip_reason = "insufficient_evidence_no_route"
        if skip_reason:
            if len(rejected) < 5:
                rejected.append(f"{candidate.poi_id}:{skip_reason}")
            continue
        if status == "reachable_now" or status == "uncertain" or candidate.expected_information_gain >= 0.7:
            selected = candidate
            selected_score = score
            break
    if selected is None:
        return ControllerInstructionV2(
            schema_version=SCHEMA_VERSION,
            game_id=blackboard.game_id,
            round_id=round_id,
            instruction_id=f"round{round_id:03d}:discriminating_probe",
            mode="discriminating_probe",
            target_poi_id=None,
            target_region=None,
            target_type=None,
            target_geometry=None,
            target_source_round=None,
            rationale="no_eligible_poi",
            progress_metric="steps_elapsed",
            stop_condition="budget",
            ranked_alternatives=rejected,
        )
    mode = "poi_approach" if reachability.get(selected.poi_id) != "unreachable_now" else "discriminating_probe"
    stop_condition = "target_reached_or_budget"
    target_region = BBox(selected.bbox.x1, selected.bbox.y1, selected.bbox.x2, selected.bbox.y2)
    return ControllerInstructionV2(
        schema_version=SCHEMA_VERSION,
        game_id=blackboard.game_id,
        round_id=round_id,
        instruction_id=f"round{round_id:03d}:{mode}:{selected.poi_id}",
        mode=mode,
        target_poi_id=selected.poi_id,
        target_region=target_region,
        target_type=selected.object_class,
        target_geometry=target_region,
        target_source_round=blackboard.round_id,
        rationale=f"ranked_top score={controller_target_score(selected_score, scoring_cfg):.3f}",
        progress_metric="distance_to_target",
        stop_condition=stop_condition,
        ranked_alternatives=rejected if rejected else [p.poi_id for _, p in ranked[1:5]],
    )
