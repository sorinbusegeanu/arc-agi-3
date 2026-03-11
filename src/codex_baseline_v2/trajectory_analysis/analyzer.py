from __future__ import annotations

from dataclasses import replace
from multiprocessing.pool import Pool
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from codex_baseline_v2.shared.config import TrajectoryAnalysisConfigV2, V2Config
from codex_baseline_v2.shared.metrics import normalize_consequence_class
from codex_baseline_v2.shared.schemas import BlackboardStateV2, CandidatePOIV2, DecisionRecordV2, ObjectRecordV2, SCHEMA_VERSION, TrajectoryEpisodeV2
from codex_baseline_v2.inference.dependency_updater import update_dependency_graph
from codex_baseline_v2.inference.latent_state_inducer import induce_latent_states
from codex_baseline_v2.inference.mechanic_graph_builder import build_mechanic_graph
from codex_baseline_v2.shared.utils import bbox_iou, merge_bboxes, normalize_palette
from codex_baseline_v2.trajectory_analysis.action_semantics import infer_action_semantics_from_episodes
from codex_baseline_v2.trajectory_analysis.area_model import infer_areas_from_episodes, merge_area_table
from codex_baseline_v2.trajectory_analysis.avatar_tracking import update_avatar_tracks_from_observation_summaries
from codex_baseline_v2.trajectory_analysis.causal_links import link_interventions_to_events
from codex_baseline_v2.trajectory_analysis.causal_chains import induce_causal_chain_hypotheses
from codex_baseline_v2.trajectory_analysis.contrast_cases import build_contrast_cases
from codex_baseline_v2.trajectory_analysis.counterfactuals import build_counterfactual_traces
from codex_baseline_v2.trajectory_analysis.event_extraction import extract_change_events_from_episodes
from codex_baseline_v2.trajectory_analysis.event_graph import attach_parent_child_event_ids, build_event_edges
from codex_baseline_v2.trajectory_analysis.evidence_ledger import update_evidence_ledger
from codex_baseline_v2.trajectory_analysis.effect_signatures import build_effect_signatures
from codex_baseline_v2.trajectory_analysis.hidden_triggers import induce_hidden_trigger_hypotheses
from codex_baseline_v2.trajectory_analysis.interventions import build_intervention_records
from codex_baseline_v2.trajectory_analysis.mechanic_induction import induce_mechanic_hypotheses
from codex_baseline_v2.trajectory_analysis.navigation_graph import build_navigation_graph_from_episodes, merge_navigation_graph
from codex_baseline_v2.trajectory_analysis.probe_outcomes import derive_probe_outcomes
from codex_baseline_v2.trajectory_analysis.reachability import classify_reachability
from codex_baseline_v2.trajectory_analysis.sequence_mining import mine_event_sequence_patterns
from codex_baseline_v2.trajectory_analysis.spatial_intervention import update_spatial_intervention_field
from codex_baseline_v2.trajectory_analysis.target_access import infer_target_access_profiles
from codex_baseline_v2.trajectory_analysis.topology_deltas import infer_topology_deltas
from codex_baseline_v2.trajectory_analysis.trigger_zones import build_trigger_zone_candidates, merge_trigger_zones


def _extract_coord_from_poi(poi: CandidatePOIV2) -> Optional[Tuple[int, int]]:
    centroid = getattr(poi, "centroid", None)
    if centroid is None or len(centroid) != 2:
        return None
    return (int(round(float(centroid[0]))), int(round(float(centroid[1]))))


def _session_dir_from_artifacts(session_state_or_artifacts: Any) -> Optional[str]:
    if isinstance(session_state_or_artifacts, str):
        return session_state_or_artifacts
    if isinstance(session_state_or_artifacts, dict):
        session_dir = session_state_or_artifacts.get("session_dir")
        if isinstance(session_dir, str):
            return session_dir
        storage_root = session_state_or_artifacts.get("storage_root")
        game_id = session_state_or_artifacts.get("game_id")
        if isinstance(storage_root, str) and isinstance(game_id, str):
            return os.path.join(storage_root, f"game_{game_id}")
    return None


def extract_all_poi_coordinates_after_round(round_id: int, session_state_or_artifacts: Any) -> List[Tuple[int, int]]:
    if isinstance(session_state_or_artifacts, BlackboardStateV2):
        return [coord for coord in (_extract_coord_from_poi(poi) for poi in session_state_or_artifacts.poi_table) if coord is not None]
    if isinstance(session_state_or_artifacts, dict):
        if "poi_table" in session_state_or_artifacts:
            blackboard = BlackboardStateV2.from_dict(session_state_or_artifacts)
            return [coord for coord in (_extract_coord_from_poi(poi) for poi in blackboard.poi_table) if coord is not None]
        nested = session_state_or_artifacts.get("blackboard") or session_state_or_artifacts.get("round_one_blackboard")
        if nested is not None:
            return extract_all_poi_coordinates_after_round(round_id, nested)
    session_dir = _session_dir_from_artifacts(session_state_or_artifacts)
    if session_dir is None:
        return []
    snapshot_path = os.path.join(
        session_dir,
        f"round_{int(round_id):03d}",
        "blackboard_snapshots",
        f"blackboard_round_{int(round_id):03d}.json",
    )
    if not os.path.exists(snapshot_path):
        return []
    with open(snapshot_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    blackboard = BlackboardStateV2.from_dict(payload)
    return [coord for coord in (_extract_coord_from_poi(poi) for poi in blackboard.poi_table) if coord is not None]


def extract_round_one_poi_coordinates_from_session_artifacts(session_state_or_artifacts: Any) -> List[Tuple[int, int]]:
    return extract_all_poi_coordinates_after_round(1, session_state_or_artifacts)


def _merge_palette(prior: List[int], current: List[int]) -> List[int]:
    return normalize_palette(list(prior) + list(current))


def _merge_pois(prior: List[CandidatePOIV2], current: List[CandidatePOIV2]) -> List[CandidatePOIV2]:
    iou_threshold = 0.30
    merged = list(prior)

    def merge_pair(prev: CandidatePOIV2, poi: CandidatePOIV2) -> CandidatePOIV2:
        bbox = merge_bboxes([prev.bbox, poi.bbox]) or poi.bbox
        return CandidatePOIV2(
            schema_version=SCHEMA_VERSION,
            poi_id=prev.poi_id,
            game_id=prev.game_id,
            source_type=prev.source_type,
            bbox=bbox,
            centroid=bbox.centroid(),
            object_class=prev.object_class,
            reachable_now=poi.reachable_now if poi.reachable_now != "uncertain" else prev.reachable_now,
            confidence=max(prev.confidence, poi.confidence),
            expected_information_gain=max(prev.expected_information_gain, poi.expected_information_gain),
            expected_interaction_type=poi.expected_interaction_type if poi.expected_interaction_type != "unknown" else prev.expected_interaction_type,
            evidence_count=prev.evidence_count + poi.evidence_count,
            observation_count=max(1, prev.observation_count or prev.evidence_count) + max(1, poi.observation_count or poi.evidence_count),
            first_seen_episode=prev.first_seen_episode or poi.first_seen_episode,
            last_seen_episode=poi.last_seen_episode or prev.last_seen_episode,
            last_seen_step=poi.last_seen_step if poi.last_seen_step is not None else prev.last_seen_step,
            first_seen_ref=prev.first_seen_ref or poi.first_seen_ref,
            last_seen_ref=poi.last_seen_ref or prev.last_seen_ref,
            type_confidence=max(prev.type_confidence, poi.type_confidence),
            utility_confidence=max(prev.utility_confidence, poi.utility_confidence),
            rejection_reasons=sorted(set(prev.rejection_reasons) | set(poi.rejection_reasons)),
            demotion_reasons=sorted(set(prev.demotion_reasons) | set(poi.demotion_reasons)),
            area_id=poi.area_id or prev.area_id,
            stable_entity_id=poi.stable_entity_id or prev.stable_entity_id,
            access_profile_id=poi.access_profile_id or prev.access_profile_id,
            last_interaction_round=poi.last_interaction_round or prev.last_interaction_round,
            interaction_count=prev.interaction_count + poi.interaction_count,
            linked_event_ids=sorted(set(prev.linked_event_ids) | set(poi.linked_event_ids)),
            linked_mechanic_hypothesis_ids=sorted(set(prev.linked_mechanic_hypothesis_ids) | set(poi.linked_mechanic_hypothesis_ids)),
        )

    for poi in current:
        best_idx = -1
        best_iou = 0.0
        for idx, existing in enumerate(merged):
            overlap = bbox_iou(existing.bbox, poi.bbox)
            if overlap > best_iou:
                best_iou = overlap
                best_idx = idx
        if best_idx < 0 or best_iou < iou_threshold:
            merged.append(poi)
            continue
        merged[best_idx] = merge_pair(merged[best_idx], poi)

    changed = True
    while changed and len(merged) > 1:
        changed = False
        best_pair = None
        best_iou = 0.0
        for idx in range(len(merged)):
            for jdx in range(idx + 1, len(merged)):
                overlap = bbox_iou(merged[idx].bbox, merged[jdx].bbox)
                if overlap >= iou_threshold and overlap > best_iou:
                    best_iou = overlap
                    best_pair = (idx, jdx)
        if best_pair is None:
            break
        idx, jdx = best_pair
        merged[idx] = merge_pair(merged[idx], merged[jdx])
        del merged[jdx]
        changed = True

    return merged


def _filter_pois(pois: List[CandidatePOIV2], cfg: TrajectoryAnalysisConfigV2) -> List[CandidatePOIV2]:
    filtered: List[CandidatePOIV2] = []
    for poi in pois:
        if poi.object_class == "hud_like":
            continue
        if poi.source_type == "motion_hotspot":
            continue
        if poi.bbox.area() <= 2:
            continue
        if max(1, poi.observation_count or poi.evidence_count) < max(1, cfg.min_poi_persistence):
            continue
        if poi.confidence < 0.2:
            continue
        filtered.append(poi)
    return filtered


def _derive_avatar_objects(episodes: List[TrajectoryEpisodeV2]) -> List[ObjectRecordV2]:
    avatars = []
    for episode in episodes:
        for step in episode.steps:
            if step.observation_summary:
                avatars.extend(step.observation_summary.avatar_candidates)
    return avatars


def _traversable_from_navigation(navigation_cells) -> Dict[str, object]:
    if not navigation_cells:
        return {"width": 0, "height": 0, "points": []}
    max_x = max(cell.cell[0] for cell in navigation_cells)
    max_y = max(cell.cell[1] for cell in navigation_cells)
    return {
        "width": max_x + 1,
        "height": max_y + 1,
        "points": [{"x": cell.cell[0], "y": cell.cell[1], "visits": cell.visit_count} for cell in navigation_cells],
    }


def _chunk_payloads(episodes: List[TrajectoryEpisodeV2], chunk_count: int) -> List[List[TrajectoryEpisodeV2]]:
    if not episodes:
        return []
    chunk_size = max(1, (len(episodes) + chunk_count - 1) // chunk_count)
    return [episodes[idx : idx + chunk_size] for idx in range(0, len(episodes), chunk_size)]


def _chunk_list(values: List[object], chunk_count: int) -> List[List[object]]:
    if not values:
        return []
    chunk_size = max(1, (len(values) + chunk_count - 1) // chunk_count)
    return [values[idx : idx + chunk_size] for idx in range(0, len(values), chunk_size)]


def _infer_areas_chunk(payload) -> List:
    episodes, existing_areas = payload
    return infer_areas_from_episodes(episodes, existing_areas)


def _build_navigation_chunk(payload) -> tuple[list, list]:
    episodes = payload
    return build_navigation_graph_from_episodes(episodes)


def _extract_events_chunk(payload) -> List:
    episodes, cfg, area_table = payload
    return extract_change_events_from_episodes(episodes, cfg, area_table, [], prior_events=None)


def _build_interventions_chunk(payload) -> List:
    episodes, cfg = payload
    return build_intervention_records(episodes, [], cfg)


def _infer_target_access_chunk(payload) -> List:
    pois, navigation_cells, navigation_edges, avatar_tracks, routing_cfg = payload
    return infer_target_access_profiles(pois, navigation_cells, navigation_edges, avatar_tracks, routing_cfg)


def _classify_reachability_chunk(payload) -> List:
    pois, avatar_tracks, navigation_cells, navigation_edges, access_profiles, routing_cfg = payload
    return classify_reachability(pois, avatar_tracks, navigation_cells, navigation_edges, access_profiles, routing_cfg)


def _build_trigger_zone_chunk(payload) -> List:
    episodes, blackboard, hidden_cfg = payload
    return build_trigger_zone_candidates(episodes, blackboard, hidden_cfg)


def _derive_probe_outcomes_chunk(payload) -> List:
    interventions, steps, events, probe_cfg, trigger_zones = payload
    return derive_probe_outcomes(interventions, steps, events, probe_cfg, trigger_zones=trigger_zones)


def analyze_trajectories(
    episodes: List[TrajectoryEpisodeV2],
    cfg: TrajectoryAnalysisConfigV2,
    round_id: int,
    prior_blackboard: Optional[BlackboardStateV2] = None,
    workers: int = 1,
    pool: Optional[Pool] = None,
) -> BlackboardStateV2:
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()
    game_id = prior_blackboard.game_id if prior_blackboard is not None else (episodes[0].game_id if episodes else "unknown_game")
    summaries = [step.observation_summary for episode in episodes for step in episode.steps if step.observation_summary is not None]
    new_palette = normalize_palette([color for summary in summaries for color in summary.palette])
    area_chunks = _chunk_payloads(episodes, max(1, min(workers, len(episodes)))) if workers > 1 else []
    if pool is not None and area_chunks:
        partial_areas = pool.map(
            _infer_areas_chunk,
            [(chunk, prior_blackboard.area_table if prior_blackboard else None) for chunk in area_chunks],
        )
        merged_new_areas = []
        for chunk_areas in partial_areas:
            merged_new_areas = merge_area_table(merged_new_areas, chunk_areas)
        area_table = merge_area_table(prior_blackboard.area_table if prior_blackboard else [], merged_new_areas)
    else:
        area_table = merge_area_table(prior_blackboard.area_table if prior_blackboard else [], infer_areas_from_episodes(episodes, prior_blackboard.area_table if prior_blackboard else None))
    timings["area_inference"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    avatar_tracks, _ = update_avatar_tracks_from_observation_summaries(
        summaries,
        V2Config().avatar_tracking,
        existing_tracks=prior_blackboard.avatar_track_table if prior_blackboard else None,
        existing_signatures=None,
    )
    timings["avatar_tracking"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    action_semantics_table, action_context_table = infer_action_semantics_from_episodes(
        episodes,
        V2Config().action_semantics,
        existing_table=prior_blackboard.action_semantics_table if prior_blackboard else None,
        existing_context_table=prior_blackboard.action_context_table if prior_blackboard else None,
    )
    timings["action_semantics"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if pool is not None and area_chunks:
        nav_chunks = pool.map(_build_navigation_chunk, area_chunks)
        new_nav_cells = []
        new_nav_edges = []
        for chunk_cells, chunk_edges in nav_chunks:
            new_nav_cells, new_nav_edges = merge_navigation_graph(new_nav_cells, new_nav_edges, chunk_cells, chunk_edges)
    else:
        new_nav_cells, new_nav_edges = build_navigation_graph_from_episodes(episodes)
    timings["navigation_graph"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    navigation_cells, navigation_edges = merge_navigation_graph(
        prior_blackboard.navigation_cells if prior_blackboard else [],
        prior_blackboard.navigation_edges if prior_blackboard else [],
        new_nav_cells,
        new_nav_edges,
    )
    current_pois = [poi for summary in summaries for poi in summary.candidate_pois]
    poi_table = _filter_pois(_merge_pois(prior_blackboard.poi_table if prior_blackboard else [], current_pois), cfg)
    if pool is not None and workers > 1 and len(poi_table) > 1:
        poi_chunks = _chunk_list(poi_table, max(1, min(workers, len(poi_table))))
        access_chunks = pool.map(
            _infer_target_access_chunk,
            [(chunk, navigation_cells, navigation_edges, avatar_tracks, V2Config().routing) for chunk in poi_chunks],
        )
        target_access_table = [profile for chunk in access_chunks for profile in chunk]
    else:
        target_access_table = infer_target_access_profiles(poi_table, navigation_cells, navigation_edges, avatar_tracks, V2Config().routing)
    timings["poi_merge_and_target_access"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if pool is not None and area_chunks:
        event_chunks = pool.map(
            _extract_events_chunk,
            [(chunk, V2Config().event_extraction, area_table) for chunk in area_chunks],
        )
        new_events = [event for chunk in event_chunks for event in chunk]
        new_events.sort(key=lambda event: (event.episode_id, event.start_step_idx, event.end_step_idx))
        events = list(prior_blackboard.event_table if prior_blackboard else []) + new_events
    else:
        events = extract_change_events_from_episodes(
            episodes,
            V2Config().event_extraction,
            area_table,
            avatar_tracks,
            prior_events=prior_blackboard.event_table if prior_blackboard else None,
        )
    timings["event_extraction"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    execution_episodes = [ep for ep in episodes if ep.metadata.get("mode") not in {"random_probe", "unguided_probe"}]
    if pool is not None and execution_episodes and workers > 1:
        exec_chunks = _chunk_payloads(execution_episodes, max(1, min(workers, len(execution_episodes))))
        intervention_chunks = pool.map(_build_interventions_chunk, [(chunk, V2Config().executor) for chunk in exec_chunks])
        interventions = list(prior_blackboard.intervention_table if prior_blackboard else []) + [
            record for chunk in intervention_chunks for record in chunk
        ]
    else:
        interventions = list(prior_blackboard.intervention_table if prior_blackboard else []) + build_intervention_records(execution_episodes, [], V2Config().executor)
    timings["interventions"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    contrast_cases = build_contrast_cases(interventions, events, prior_blackboard.contrast_cases if prior_blackboard else [], V2Config().causality)
    cause_effect_table = list(prior_blackboard.cause_effect_table if prior_blackboard else []) + link_interventions_to_events(interventions, events, contrast_cases, V2Config().causality)
    topology_delta_table = list(prior_blackboard.topology_delta_table if prior_blackboard else []) + infer_topology_deltas(
        events,
        prior_blackboard.navigation_edges if prior_blackboard else [],
        navigation_edges,
        area_table,
        V2Config().area_model,
    )
    timings["causal_links_and_topology"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    trigger_t0 = time.perf_counter()
    trigger_zone_table = merge_trigger_zones(
        prior_blackboard.trigger_zone_table if prior_blackboard else [],
        [],
        V2Config().hidden_trigger,
    )
    trigger_zone_table = merge_trigger_zones(
        trigger_zone_table,
        build_trigger_zone_candidates(episodes, prior_blackboard, V2Config().hidden_trigger),
        V2Config().hidden_trigger,
    )
    timings["trigger_zone_build"] = time.perf_counter() - trigger_t0
    all_steps = [step for episode in episodes for step in episode.steps]
    spatial_t0 = time.perf_counter()
    spatial_intervention_field = update_spatial_intervention_field(
        prior_blackboard.spatial_intervention_field if prior_blackboard else [],
        episodes,
        interventions,
        events,
        V2Config().hidden_trigger,
    )
    timings["spatial_intervention_field"] = time.perf_counter() - spatial_t0
    probe_t0 = time.perf_counter()
    probe_outcome_table = list(prior_blackboard.probe_outcome_table if prior_blackboard else []) + derive_probe_outcomes(
        interventions,
        all_steps,
        events,
        V2Config().probe_mode,
        trigger_zones=trigger_zone_table,
    )
    timings["probe_outcomes"] = time.perf_counter() - probe_t0
    timings["trigger_and_probe_models"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    event_edge_table = list(prior_blackboard.event_edge_table if prior_blackboard else []) + build_event_edges(events, interventions, V2Config().causal_chain)
    events = attach_parent_child_event_ids(events, event_edge_table)
    effect_signature_table = build_effect_signatures(events, topology_delta_table, V2Config().sequence_mining)
    signature_lookup = {(sig.event_type, sig.locality, sig.area_relation): sig.effect_signature_id for sig in effect_signature_table}
    events = [
        replace(
            event,
            effect_signature_id=event.effect_signature_id or signature_lookup.get((event.event_type, event.locality, "same_area" if event.pre_area_id == event.post_area_id else "cross_area")),
        )
        for event in events
    ]
    event_sequence_patterns = list(prior_blackboard.event_sequence_patterns if prior_blackboard else []) + mine_event_sequence_patterns(
        interventions,
        events,
        event_edge_table,
        effect_signature_table,
        V2Config().sequence_mining,
    )
    causal_chain_hypotheses = list(prior_blackboard.causal_chain_hypotheses if prior_blackboard else []) + induce_causal_chain_hypotheses(
        interventions,
        trigger_zone_table,
        event_sequence_patterns,
        event_edge_table,
        V2Config().causal_chain,
    )
    hidden_trigger_hypotheses = list(prior_blackboard.hidden_trigger_hypotheses if prior_blackboard else []) + induce_hidden_trigger_hypotheses(
        trigger_zone_table,
        interventions,
        spatial_intervention_field,
        events,
        V2Config().hidden_trigger,
    )
    counterfactual_traces = list(prior_blackboard.counterfactual_traces if prior_blackboard else []) + build_counterfactual_traces(
        interventions,
        events,
        hidden_trigger_hypotheses,
        causal_chain_hypotheses,
        V2Config().probe_mode,
    )
    timings["event_graph_sequences_and_counterfactuals"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    latent_states = induce_latent_states(
        events,
        interventions,
        trigger_zone_table,
        event_edge_table,
        area_table,
        existing=prior_blackboard.latent_states if prior_blackboard else None,
    )
    mechanic_graph = build_mechanic_graph(
        cause_effect_table,
        events,
        latent_states,
        topology_delta_table,
        causal_chain_hypotheses,
        existing=prior_blackboard.mechanic_graph if prior_blackboard else None,
        round_id=round_id,
        step_id=max((event.end_step_idx for event in events), default=0),
    )
    mechanic_hypotheses = induce_mechanic_hypotheses(
        cause_effect_table,
        events,
        poi_table,
        area_table,
        V2Config().mechanic_induction,
        existing=prior_blackboard.mechanic_hypotheses if prior_blackboard else None,
    )
    mechanic_hypotheses = [
        replace(
            mech,
            chain_hypothesis_ids=sorted(set(mech.chain_hypothesis_ids) | {chain.chain_id for chain in causal_chain_hypotheses if any(event_id in mech.support_event_ids for event_id in chain.ordered_event_ids)}),
            hidden_trigger_hypothesis_ids=sorted(set(mech.hidden_trigger_hypothesis_ids) | {hidden.hidden_hypothesis_id for hidden in hidden_trigger_hypotheses if hidden.effect_signature_id is not None}),
            event_sequence_pattern_ids=sorted(set(mech.event_sequence_pattern_ids) | {pattern.pattern_id for pattern in event_sequence_patterns if any(event_id in mech.support_event_ids for event_id in pattern.source_event_ids)}),
        )
        for mech in mechanic_hypotheses
    ]
    evidence_ledger = update_evidence_ledger(
        prior_blackboard.evidence_ledger if prior_blackboard else [],
        [("poi", poi.poi_id, "reachable", [poi.first_seen_ref] if poi.first_seen_ref else [], []) for poi in poi_table],
        round_id,
    )
    timings["latent_mechanic_and_ledger"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if pool is not None and workers > 1 and len(poi_table) > 1:
        poi_chunks = _chunk_list(poi_table, max(1, min(workers, len(poi_table))))
        access_by_poi = {profile.poi_id: profile for profile in target_access_table}
        reach_chunks = pool.map(
            _classify_reachability_chunk,
            [
                (
                    chunk,
                    avatar_tracks,
                    navigation_cells,
                    navigation_edges,
                    [access_by_poi[poi.poi_id] for poi in chunk if poi.poi_id in access_by_poi],
                    V2Config().routing,
                )
                for chunk in poi_chunks
            ],
        )
        reachability_table = [row for chunk in reach_chunks for row in chunk]
    else:
        reachability_table = classify_reachability(
            poi_table,
            avatar_tracks,
            navigation_cells,
            navigation_edges,
            target_access_table,
            V2Config().routing,
        )
    dependency_graph = update_dependency_graph(
        mechanic_graph,
        reachability_table,
        existing=prior_blackboard.dependency_graph if prior_blackboard else None,
        round_id=round_id,
        step_id=max((event.end_step_idx for event in events), default=0),
    )
    timings["reachability_and_dependency_graph"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    reach_lookup = {r.poi_id: r for r in reachability_table}
    poi_table = [
        CandidatePOIV2(
            schema_version=poi.schema_version,
            poi_id=poi.poi_id,
            game_id=poi.game_id,
            source_type=poi.source_type,
            bbox=poi.bbox,
            centroid=poi.centroid,
            object_class=poi.object_class,
            reachable_now=reach_lookup.get(poi.poi_id).status if poi.poi_id in reach_lookup else poi.reachable_now,
            confidence=poi.confidence,
            expected_information_gain=poi.expected_information_gain,
            expected_interaction_type=poi.expected_interaction_type,
            evidence_count=poi.evidence_count,
            observation_count=poi.observation_count,
            first_seen_episode=poi.first_seen_episode,
            last_seen_episode=poi.last_seen_episode,
            last_seen_step=poi.last_seen_step,
            first_seen_ref=poi.first_seen_ref,
            last_seen_ref=poi.last_seen_ref,
            type_confidence=poi.type_confidence,
            utility_confidence=poi.utility_confidence,
            rejection_reasons=poi.rejection_reasons,
            demotion_reasons=poi.demotion_reasons,
            area_id=poi.area_id,
            stable_entity_id=poi.stable_entity_id,
            access_profile_id=next((profile.poi_id for profile in target_access_table if profile.poi_id == poi.poi_id), poi.access_profile_id),
            last_interaction_round=poi.last_interaction_round,
            interaction_count=poi.interaction_count,
            linked_event_ids=poi.linked_event_ids,
            linked_mechanic_hypothesis_ids=poi.linked_mechanic_hypothesis_ids,
        )
        for poi in poi_table
    ]
    avatar_hypotheses = _derive_avatar_objects(episodes) or (prior_blackboard.avatar_hypotheses if prior_blackboard else [])
    consequence_table = list(prior_blackboard.consequence_table if prior_blackboard else [])
    consequence_event_ids = {
        event_id
        for record in consequence_table
        for event_id in getattr(record, "event_ids", [])
    }
    for event in events:
        if event.event_id in consequence_event_ids:
            continue
        consequence_table.append(
            __import__("codex_baseline_v2.shared.schemas", fromlist=["ConsequenceRecordV2"]).ConsequenceRecordV2(
                schema_version=SCHEMA_VERSION,
                game_id=game_id,
                poi_id=event.trigger_target_poi_id or "unknown",
                round_id=round_id,
                episode_id=event.episode_id,
                instruction_id=event.trigger_instruction_id,
                target_poi_id=event.trigger_target_poi_id,
                distance_decreased=False,
                reached=False,
                contact=False,
                local_change_magnitude=sum(delta.pixel_change_ratio for delta in event.region_deltas),
                global_change_magnitude=sum(delta.pixel_change_ratio for delta in event.region_deltas),
                reward_delta=event.reward_delta,
                terminal_flag_changed=event.terminal_flag_changed,
                object_change_summary=event.event_type,
                followup_poi_ids=[],
                consequence_class=normalize_consequence_class(event.event_type),
                event_ids=[event.event_id],
                cause_effect_link_ids=[],
                area_id=event.post_area_id,
                topology_delta_id=None,
            )
        )
        consequence_event_ids.add(event.event_id)
    decision_history = list(prior_blackboard.decision_history if prior_blackboard else [])
    traversable_map = _traversable_from_navigation(navigation_cells)
    timings["final_blackboard_assembly"] = time.perf_counter() - t0
    return BlackboardStateV2(
        schema_version=SCHEMA_VERSION,
        game_id=game_id,
        round_id=round_id,
        palette=_merge_palette(prior_blackboard.palette if prior_blackboard else [], new_palette),
        poi_table=poi_table,
        reachability_table=reachability_table,
        consequence_table=consequence_table,
        avatar_hypotheses=avatar_hypotheses,
        traversable_map=traversable_map,
        unresolved_hypotheses=list(prior_blackboard.unresolved_hypotheses if prior_blackboard else []),
        falsified_hypotheses=list(prior_blackboard.falsified_hypotheses if prior_blackboard else []),
        action_semantics_table=action_semantics_table,
        action_context_table=action_context_table,
        avatar_track_table=avatar_tracks,
        target_access_table=target_access_table,
        navigation_cells=navigation_cells,
        navigation_edges=navigation_edges,
        area_table=area_table,
        event_table=events,
        intervention_table=interventions,
        cause_effect_table=cause_effect_table,
        topology_delta_table=topology_delta_table,
        mechanic_hypotheses=mechanic_hypotheses,
        evidence_ledger=evidence_ledger,
        decision_history=decision_history,
        contrast_cases=contrast_cases,
        trigger_zone_table=trigger_zone_table,
        spatial_intervention_field=spatial_intervention_field,
        probe_outcome_table=probe_outcome_table,
        event_edge_table=event_edge_table,
        event_sequence_patterns=event_sequence_patterns,
        causal_chain_hypotheses=causal_chain_hypotheses,
        hidden_trigger_hypotheses=hidden_trigger_hypotheses,
        counterfactual_traces=counterfactual_traces,
        effect_signature_table=effect_signature_table,
        latent_states=latent_states,
        mechanic_graph=mechanic_graph,
        dependency_graph=dependency_graph,
        metadata={
            **{k: v for k, v in (prior_blackboard.metadata if prior_blackboard else {}).items() if k != "last_observation"},
            "analysis_stage_timings": {k: round(v, 6) for k, v in timings.items()},
        },
    )
