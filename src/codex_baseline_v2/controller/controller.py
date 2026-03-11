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


def _failure_penalties(blackboard: BlackboardStateV2) -> Dict[str, float]:
    penalties: Dict[str, float] = {}
    for consequence in blackboard.consequence_table[-20:]:
        if not consequence.target_poi_id:
            continue
        if consequence.consequence_class == "no_change" and not consequence.distance_decreased and not consequence.reached and not consequence.contact:
            penalties[consequence.target_poi_id] = penalties.get(consequence.target_poi_id, 0.0) + 0.75
    for record in blackboard.decision_history[-10:]:
        if record.selected_target_poi_id and (record.target_invalidated or record.outcome_summary == "no_progress"):
            penalties[record.selected_target_poi_id] = penalties.get(record.selected_target_poi_id, 0.0) + 0.5
    return penalties


def _target_region(blackboard: BlackboardStateV2, poi: CandidatePOIV2) -> BBox:
    profile = next((profile for profile in blackboard.target_access_table if profile.poi_id == poi.poi_id), None)
    if profile and profile.access_cells:
        xs = [cell[0] for cell in profile.access_cells]
        ys = [cell[1] for cell in profile.access_cells]
        return BBox(min(xs), min(ys), max(xs), max(ys))
    return poi.bbox


def _zone_region(cells: List[Tuple[int, int]]) -> Optional[BBox]:
    if not cells:
        return None
    xs = [cell[0] for cell in cells]
    ys = [cell[1] for cell in cells]
    return BBox(min(xs), min(ys), max(xs), max(ys))


def _rank_pois(
    blackboard: BlackboardStateV2,
    scoring_cfg: ScoringConfigV2,
    failure_penalties: Dict[str, float],
) -> List[Tuple[float, CandidatePOIV2]]:
    reachability = _reachability_map(blackboard)
    recent_event_ids = {event.event_id for event in blackboard.event_table[-10:]}
    ranked: List[Tuple[float, CandidatePOIV2]] = []
    for poi in blackboard.poi_table:
        if poi.object_class == "hud_like":
            continue
        if poi.confidence < 0.2:
            continue
        if max(1, poi.observation_count or poi.evidence_count) < 2:
            continue
        if poi.source_type == "motion_hotspot":
            continue
        base = poi_rank_score(
            POIRankInputs(
                info_gain=poi.expected_information_gain,
                confidence=poi.confidence,
                reachability_score=reachability_to_score(reachability.get(poi.poi_id, "uncertain")),
            ),
            scoring_cfg,
        )
        access_bonus = 0.6 if poi.access_profile_id else 0.0
        novelty_bonus = 0.8 if not any(event_id in recent_event_ids for event_id in poi.linked_event_ids) else 0.0
        unresolved_bonus = 0.5 if blackboard.mechanic_hypotheses or blackboard.cause_effect_table else 0.0
        cross_area_bonus = 0.4 if poi.area_id and any(area.area_id != poi.area_id for area in blackboard.area_table) else 0.0
        blocked_penalty = 1.0 if reachability.get(poi.poi_id) in {"blocked", "unreachable"} else 0.0
        no_access_penalty = 1.0 if not poi.access_profile_id else 0.0
        hud_penalty = 1.0 if poi.object_class == "hud_like" else 0.0
        stale_penalty = failure_penalties.get(poi.poi_id, 0.0)
        score = base + access_bonus + novelty_bonus + unresolved_bonus + cross_area_bonus - blocked_penalty - no_access_penalty - hud_penalty - stale_penalty
        ranked.append((score, poi))
    ranked.sort(key=lambda item: item[0], reverse=True)
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

    ranked = _rank_pois(blackboard, scoring_cfg, _failure_penalties(blackboard))
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

    hidden_candidates = sorted(
        blackboard.trigger_zone_table,
        key=lambda zone: (zone.hidden_trigger_confidence - 0.15 * zone.null_count - 0.1 * zone.contradiction_count, zone.activation_count),
        reverse=True,
    )
    chain_candidates = sorted(
        blackboard.causal_chain_hypotheses,
        key=lambda chain: (chain.confidence, chain.support_count, -chain.contradiction_count),
        reverse=True,
    )
    counterfactual_candidates = sorted(
        blackboard.counterfactual_traces,
        key=lambda trace: (1.0 - float(trace.supports_reference), trace.confidence),
        reverse=True,
    )
    if hidden_candidates and hidden_candidates[0].hidden_trigger_confidence >= 0.45:
        selected_zone = hidden_candidates[0]
        region = selected_zone.bbox or _zone_region(selected_zone.cells)
        return ControllerInstructionV2(
            schema_version=SCHEMA_VERSION,
            game_id=blackboard.game_id,
            round_id=round_id,
            instruction_id=f"round{round_id:03d}:hidden_trigger_probe:{selected_zone.trigger_zone_id}",
            mode="step_on_region" if selected_zone.condition_type in {"step_on", "cross"} else "dwell_on_region",
            target_poi_id=None,
            target_region=region,
            target_type="trigger_zone",
            target_geometry=region,
            target_source_round=blackboard.round_id,
            rationale=f"intent_class=hidden_trigger_probe trigger_zone_id={selected_zone.trigger_zone_id}",
            progress_metric="zone_entry",
            stop_condition="probe_budget",
            ranked_alternatives=[zone.trigger_zone_id for zone in hidden_candidates[1:5]],
        )
    if chain_candidates and 0.4 <= chain_candidates[0].confidence <= 0.85:
        selected_chain = chain_candidates[0]
        linked_zone = next((zone for zone in blackboard.trigger_zone_table if zone.trigger_zone_id == selected_chain.trigger_zone_id), None)
        region = linked_zone.bbox if linked_zone is not None else None
        return ControllerInstructionV2(
            schema_version=SCHEMA_VERSION,
            game_id=blackboard.game_id,
            round_id=round_id,
            instruction_id=f"round{round_id:03d}:causal_chain_verification:{selected_chain.chain_id}",
            mode="repeat_route_fragment",
            target_poi_id=selected_chain.trigger_poi_id,
            target_region=region,
            target_type="causal_chain",
            target_geometry=region,
            target_source_round=blackboard.round_id,
            rationale=f"intent_class=causal_chain_verification chain_id={selected_chain.chain_id} trigger_zone_id={selected_chain.trigger_zone_id}",
            progress_metric="event_sequence_match",
            stop_condition="verification_budget",
            ranked_alternatives=[chain.chain_id for chain in chain_candidates[1:5]],
        )
    if counterfactual_candidates and counterfactual_candidates[0].confidence >= 0.4:
        selected_trace = counterfactual_candidates[0]
        linked_zone = next((zone for zone in blackboard.trigger_zone_table if zone.trigger_zone_id == selected_trace.target_trigger_zone_id), None)
        region = linked_zone.bbox if linked_zone is not None else None
        return ControllerInstructionV2(
            schema_version=SCHEMA_VERSION,
            game_id=blackboard.game_id,
            round_id=round_id,
            instruction_id=f"round{round_id:03d}:counterfactual:{selected_trace.counterfactual_id}",
            mode="counterfactual_avoid_contact",
            target_poi_id=selected_trace.target_poi_id,
            target_region=region,
            target_type="counterfactual",
            target_geometry=region,
            target_source_round=blackboard.round_id,
            rationale=f"intent_class=counterfactual_disambiguation_probe counterfactual_id={selected_trace.counterfactual_id} trigger_zone_id={selected_trace.target_trigger_zone_id}",
            progress_metric="ambiguity_reduction",
            stop_condition="probe_budget",
            ranked_alternatives=[trace.counterfactual_id for trace in counterfactual_candidates[1:5]],
        )

    selected_score, selected = ranked[0]
    competing_links = [link for link in blackboard.cause_effect_table if link.cause_poi_id == selected.poi_id]
    mode = "poi_approach"
    if len(competing_links) >= 2:
        mode = "discriminating_probe"
    elif any(link.confidence > 0.6 for link in competing_links):
        mode = "poi_interaction"
    elif any(h.cross_area_supported for h in blackboard.mechanic_hypotheses):
        mode = "exploit"

    target_region = _target_region(blackboard, selected)
    if mode == "poi_approach" and (selected.access_profile_id is None or target_region is None):
        mode = "discriminating_probe"
        target_region = None
    return ControllerInstructionV2(
        schema_version=SCHEMA_VERSION,
        game_id=blackboard.game_id,
        round_id=round_id,
        instruction_id=f"round{round_id:03d}:{mode}:{selected.poi_id}",
        mode=mode,
        target_poi_id=selected.poi_id if target_region is not None else None,
        target_region=target_region,
        target_type=selected.object_class,
        target_geometry=target_region,
        target_source_round=blackboard.round_id,
        rationale=f"intent_class=poi_interaction_probe score={controller_target_score(selected_score, scoring_cfg):.3f}",
        progress_metric="route_distance",
        stop_condition="target_reached_or_budget",
        ranked_alternatives=[poi.poi_id for _, poi in ranked[1:5]],
    )
