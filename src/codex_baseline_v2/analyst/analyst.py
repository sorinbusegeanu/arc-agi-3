from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import replace
import json
from multiprocessing import get_context
from multiprocessing.pool import Pool
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from codex_baseline_v2.analyst.frame_analyst import AvatarCandidateAccumulator, score_avatar_candidates
from codex_baseline_v2.analyst.poi_miner import mine_pois
from codex_baseline_v2.shared.config import AnalystConfigV2, AvatarTrackingConfigV2, TrajectoryAnalysisConfigV2
from codex_baseline_v2.shared.schemas import ObjectRecordV2, ObservationSummaryV2, SCHEMA_VERSION, TrajectoryEpisodeV2, TrajectoryStepV2
from codex_baseline_v2.shared.state_identity import canonical_state_identity
from codex_baseline_v2.shared.utils import (
    BBox,
    bbox_from_points,
    connected_components_per_color,
    grid_diff,
    grid_palette,
    merge_bboxes,
    compact_context_key,
)
from codex_baseline_v2.trajectory_analysis.area_model import assign_area_id_to_summary, infer_areas_from_episodes
from codex_baseline_v2.trajectory_analysis.avatar_tracking import update_avatar_tracks_from_observation_summaries


_STATE_SIG_CACHE: "OrderedDict[Tuple[Tuple[int, ...], ...], Dict[str, object]]" = OrderedDict()
_COMP_CACHE: "OrderedDict[Tuple[Tuple[int, ...], ...], Dict[int, List[List[Tuple[int, int]]]]]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_LIMIT = 128

_WORKER_ANALYST_CFG: Optional[AnalystConfigV2] = None
_WORKER_AVATAR_CFG: Optional[AvatarTrackingConfigV2] = None
_WORKER_AREA_TABLE = None


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


def _extract_step_visit_coord(step: TrajectoryStepV2) -> Optional[Tuple[int, int]]:
    if step.actual_avatar_centroid is not None:
        return (int(round(float(step.actual_avatar_centroid[0]))), int(round(float(step.actual_avatar_centroid[1]))))
    if step.predicted_avatar_centroid is not None:
        return (int(round(float(step.predicted_avatar_centroid[0]))), int(round(float(step.predicted_avatar_centroid[1]))))
    summary = step.observation_summary
    if summary is not None and summary.avatar_candidates:
        centroid = summary.avatar_candidates[0].centroid
        return (int(round(float(centroid[0]))), int(round(float(centroid[1]))))
    return None


def extract_main_sprite_pixel_visits_from_session_artifacts(session_state_or_artifacts: Any) -> List[Tuple[int, int]]:
    episodes: List[TrajectoryEpisodeV2] = []
    if isinstance(session_state_or_artifacts, dict):
        raw_episodes = session_state_or_artifacts.get("analyzed_episodes") or session_state_or_artifacts.get("episodes")
        if isinstance(raw_episodes, list):
            episodes = [episode if isinstance(episode, TrajectoryEpisodeV2) else TrajectoryEpisodeV2.from_dict(episode) for episode in raw_episodes]
    session_dir = _session_dir_from_artifacts(session_state_or_artifacts)
    if not episodes and session_dir is not None and os.path.isdir(session_dir):
        round_dirs = sorted(
            entry for entry in os.listdir(session_dir)
            if entry.startswith("round_") and os.path.isdir(os.path.join(session_dir, entry))
        )
        for round_dir in round_dirs:
            artifact_path = os.path.join(session_dir, round_dir, "normalized_trajectories", "episodes.jsonl")
            if not os.path.exists(artifact_path):
                continue
            with open(artifact_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    payload = json.loads(line)
                    episodes.append(TrajectoryEpisodeV2.from_dict(payload))
    visits: List[Tuple[int, int]] = []
    for episode in episodes:
        for step in episode.steps:
            coord = _extract_step_visit_coord(step)
            if coord is not None:
                visits.append(coord)
    return visits


def _grid_key(grid: Optional[List[List[int]]]) -> Optional[Tuple[Tuple[int, ...], ...]]:
    if grid is None:
        return None
    return tuple(tuple(int(value) for value in row) for row in grid)


def _cache_get(cache: OrderedDict, key):
    with _CACHE_LOCK:
        value = cache.get(key)
        if value is None:
            return None
        cache.move_to_end(key)
        return value


def _cache_put(cache: OrderedDict, key, value) -> None:
    with _CACHE_LOCK:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > _CACHE_LIMIT:
            cache.popitem(last=False)


def _state_signature_cached(grid: Optional[List[List[int]]]) -> Dict[str, object]:
    key = _grid_key(grid)
    if key is None:
        return canonical_state_identity(grid, include_payload=False)
    cached = _cache_get(_STATE_SIG_CACHE, key)
    if cached is not None:
        return cached
    value = canonical_state_identity(grid, include_payload=False)
    _cache_put(_STATE_SIG_CACHE, key, value)
    return value


def _connected_components_cached(grid: List[List[int]]) -> Dict[int, List[List[Tuple[int, int]]]]:
    key = _grid_key(grid)
    if key is None:
        return connected_components_per_color(grid)
    cached = _cache_get(_COMP_CACHE, key)
    if cached is not None:
        return cached
    value = connected_components_per_color(grid)
    _cache_put(_COMP_CACHE, key, value)
    return value


def _clone_summary(
    summary: ObservationSummaryV2,
    *,
    area_id: Optional[str] = None,
    avatar_track_hypotheses=None,
    navigation_context_key: Optional[str] = None,
) -> ObservationSummaryV2:
    return replace(
        summary,
        area_id=summary.area_id if area_id is None else area_id,
        avatar_track_hypotheses=summary.avatar_track_hypotheses if avatar_track_hypotheses is None else avatar_track_hypotheses,
        navigation_context_key=summary.navigation_context_key if navigation_context_key is None else navigation_context_key,
    )


def _clone_step(
    step: TrajectoryStepV2,
    *,
    observation_summary: Optional[ObservationSummaryV2] = None,
    area_id: Optional[str] = None,
    avatar_track_id: Optional[str] = None,
    predicted_avatar_centroid=None,
    actual_avatar_centroid=None,
    action_context_key: Optional[str] = None,
) -> TrajectoryStepV2:
    return replace(
        step,
        observation_summary=step.observation_summary if observation_summary is None else observation_summary,
        area_id=step.area_id if area_id is None else area_id,
        avatar_track_id=step.avatar_track_id if avatar_track_id is None else avatar_track_id,
        predicted_avatar_centroid=step.predicted_avatar_centroid if predicted_avatar_centroid is None else predicted_avatar_centroid,
        actual_avatar_centroid=step.actual_avatar_centroid if actual_avatar_centroid is None else actual_avatar_centroid,
        action_context_key=step.action_context_key if action_context_key is None else action_context_key,
    )


def _score_background_candidates(
    grid: List[List[int]],
    cfg: AnalystConfigV2,
    comps: Optional[Dict[int, List[List[Tuple[int, int]]]]] = None,
) -> List[Dict[str, float]]:
    height = len(grid)
    width = len(grid[0]) if height else 0
    total = max(1, height * width)
    counts: Dict[int, int] = {}
    border_counts: Dict[int, int] = {}
    for y, row in enumerate(grid):
        for x, v in enumerate(row):
            val = int(v)
            counts[val] = counts.get(val, 0) + 1
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                border_counts[val] = border_counts.get(val, 0) + 1
    comps = comps if comps is not None else _connected_components_cached(grid)
    candidates = []
    for color, count in counts.items():
        freq = count / float(total)
        border = border_counts.get(color, 0) / float(max(1, (width * 2 + height * 2 - 4)))
        largest_comp = 0
        for comp in comps.get(color, []):
            largest_comp = max(largest_comp, len(comp))
        connected = largest_comp / float(max(1, count))
        score = cfg.bg_frequency_weight * freq + cfg.bg_border_weight * border + cfg.bg_connected_weight * connected
        candidates.append({"color": color, "score": score, "frequency": freq, "border": border, "connected": connected})
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def _classify_object(bbox: BBox, area: int, grid_height: int, grid_width: int, cfg: AnalystConfigV2) -> Tuple[str, float]:
    height_ratio = bbox.height() / float(max(1, grid_height))
    width_ratio = bbox.width() / float(max(1, grid_width))
    if height_ratio <= cfg.hud_height_ratio and width_ratio >= cfg.hud_width_ratio:
        return "hud_like", 0.65
    if bbox.width() >= grid_width - 2 and bbox.height() >= grid_height - 2:
        return "world_object", 0.5
    if height_ratio < 0.3 and width_ratio < 0.3 and area >= 3:
        return "obstacle", 0.45
    if area <= 2:
        return "unknown", 0.3
    return "world_object", 0.5


def _rank_tracks_for_summary(
    summary: ObservationSummaryV2,
    tracks,
    cfg: AvatarTrackingConfigV2,
):
    if not tracks:
        return []
    anchor = summary.avatar_candidates[0].centroid if summary.avatar_candidates else None
    if anchor is None:
        ranked = sorted(tracks, key=lambda row: (row.posterior, row.support_count, -row.missing_count), reverse=True)
        return ranked[: cfg.max_hypotheses]

    def _score(track) -> Tuple[float, float, int]:
        dist = abs(track.centroid[0] - anchor[0]) + abs(track.centroid[1] - anchor[1])
        return (-dist, track.posterior, track.support_count)

    ranked = sorted(tracks, key=_score, reverse=True)
    return ranked[: cfg.max_hypotheses]


def _finalize_episode_tracks(steps_out: List[TrajectoryStepV2]) -> List[TrajectoryStepV2]:
    summaries = [step.observation_summary for step in steps_out if step.observation_summary is not None]
    if not summaries:
        return steps_out
    track_cfg = AvatarTrackingConfigV2()
    tracks, _ = update_avatar_tracks_from_observation_summaries(summaries, cfg=track_cfg)
    finalized: List[TrajectoryStepV2] = []
    for step in steps_out:
        summary = step.observation_summary
        if summary is None:
            finalized.append(step)
            continue
        ranked_tracks = _rank_tracks_for_summary(summary, tracks, track_cfg)
        primary = ranked_tracks[0] if ranked_tracks else None
        updated_summary = _clone_summary(summary, avatar_track_hypotheses=ranked_tracks)
        finalized.append(
            _clone_step(
                step,
                observation_summary=updated_summary,
                area_id=updated_summary.area_id,
                avatar_track_id=primary.track_id if primary is not None else None,
                predicted_avatar_centroid=primary.predicted_centroid if primary is not None else None,
                actual_avatar_centroid=(
                    primary.centroid if primary is not None
                    else (summary.avatar_candidates[0].centroid if summary.avatar_candidates else None)
                ),
                action_context_key=updated_summary.navigation_context_key,
            )
        )
    return finalized


def analyze_episode(episode: TrajectoryEpisodeV2, cfg: AnalystConfigV2, accumulator: AvatarCandidateAccumulator) -> TrajectoryEpisodeV2:
    steps_out: List[TrajectoryStepV2] = []
    prev_obs: Optional[List[List[int]]] = None
    prev_objects: List[ObjectRecordV2] = []
    for step in episode.steps:
        obs = step.observation
        if obs is None:
            steps_out.append(step)
            prev_obs = obs
            prev_objects = []
            continue
        state_signature = _state_signature_cached(obs)
        palette = grid_palette(obs)
        comps = _connected_components_cached(obs)
        bg_candidates = _score_background_candidates(obs, cfg, comps=comps)
        bg_colors = [c["color"] for c in bg_candidates[:2]]
        objects: List[ObjectRecordV2] = []
        hud_regions: List[BBox] = []
        world_regions: List[BBox] = []
        for color, comps_list in comps.items():
            for idx, comp in enumerate(comps_list):
                if len(comp) < cfg.min_area:
                    continue
                bbox = bbox_from_points(comp)
                if bbox is None:
                    continue
                area = len(comp)
                aspect_ratio = float(bbox.width()) / float(max(1, bbox.height()))
                obj_class, confidence = _classify_object(bbox, area, len(obs), len(obs[0]), cfg)
                if obj_class == "hud_like":
                    hud_regions.append(bbox)
                else:
                    world_regions.append(bbox)
                objects.append(
                    ObjectRecordV2(
                        schema_version=SCHEMA_VERSION,
                        object_id=f"{episode.episode_id}:{step.step_idx}:{color}:{idx}",
                        game_id=episode.game_id,
                        episode_id=episode.episode_id,
                        bbox=bbox,
                        centroid=bbox.centroid(),
                        color=int(color),
                        area=area,
                        aspect_ratio=aspect_ratio,
                        object_class=obj_class,
                        confidence=confidence,
                        evidence_refs=[f"{episode.episode_id}:{step.step_idx}"],
                        first_seen_ref=f"{episode.episode_id}:{step.step_idx}",
                        last_seen_ref=f"{episode.episode_id}:{step.step_idx}",
                    )
                )
        clustered_objects: List[ObjectRecordV2] = []
        by_color: Dict[int, List[ObjectRecordV2]] = {}
        for obj in objects:
            by_color.setdefault(obj.color, []).append(obj)
        for color, objs in by_color.items():
            elongated = [o for o in objs if o.aspect_ratio >= 3.0 or o.aspect_ratio <= 0.33]
            if len(elongated) < 2:
                continue
            merged_bbox = merge_bboxes([o.bbox for o in elongated])
            if merged_bbox is None:
                continue
            clustered_objects.append(
                ObjectRecordV2(
                    schema_version=SCHEMA_VERSION,
                    object_id=f"cluster:{episode.episode_id}:{step.step_idx}:{color}",
                    game_id=episode.game_id,
                    episode_id=episode.episode_id,
                    bbox=merged_bbox,
                    centroid=merged_bbox.centroid(),
                    color=int(color),
                    area=sum(o.area for o in elongated),
                    aspect_ratio=float(merged_bbox.width()) / float(max(1, merged_bbox.height())),
                    object_class="world_object",
                    confidence=0.4,
                    evidence_refs=[f"{episode.episode_id}:{step.step_idx}"],
                    first_seen_ref=f"{episode.episode_id}:{step.step_idx}",
                    last_seen_ref=f"{episode.episode_id}:{step.step_idx}",
                )
            )
        if clustered_objects:
            objects.extend(clustered_objects)
        active_regions: List[BBox] = []
        static_regions: List[BBox] = []
        motion_points: List[Tuple[int, int]] = []
        if prev_obs is not None:
            _, diff_points = grid_diff(prev_obs, obs)
            motion_points = diff_points
            diff_bbox = bbox_from_points(diff_points)
            if diff_bbox:
                active_regions.append(diff_bbox)
        action_id = step.action.action_id if step.action.action_type == "discrete" else None
        avatar_candidates, avatar_table, avatar_rejections = score_avatar_candidates(
            objects=objects,
            prev_objects=prev_objects,
            motion_points=motion_points,
            action_id=action_id,
            cfg=cfg,
            accumulator=accumulator,
            game_id=episode.game_id,
            episode_id=episode.episode_id,
        )
        candidate_pois = mine_pois(
            objects=objects,
            bg_colors=bg_colors,
            motion_points=motion_points,
            cfg=cfg,
            episode_id=episode.episode_id,
            step_idx=step.step_idx,
            grid_width=len(obs[0]) if obs else 0,
            grid_height=len(obs),
        )
        summary = ObservationSummaryV2(
            schema_version=SCHEMA_VERSION,
            game_id=episode.game_id,
            episode_id=episode.episode_id,
            step_idx=step.step_idx,
            palette=palette,
            background_candidates=bg_candidates,
            foreground_candidates=[c for c in palette if c not in bg_colors],
            objects=objects,
            active_regions=active_regions,
            static_regions=static_regions,
            hud_region_candidates=hud_regions,
            world_region_candidates=world_regions,
            avatar_candidates=avatar_candidates,
            candidate_pois=candidate_pois,
            area_id=f"area:{episode.episode_id}",
            avatar_track_hypotheses=[],
            state_signature_id=state_signature.get("state_hash"),
            navigation_context_key=compact_context_key(None, "obs", "far", action_id),
            avatar_candidate_table=avatar_table,
            avatar_rejection_reasons=avatar_rejections,
        )
        steps_out.append(
            TrajectoryStepV2(
                schema_version=step.schema_version,
                game_id=step.game_id,
                episode_id=step.episode_id,
                step_idx=step.step_idx,
                action=step.action,
                pre_state_hash=step.pre_state_hash or state_signature.get("state_hash"),
                post_state_hash=step.post_state_hash,
                state_hash_valid=step.state_hash_valid,
                instruction_id=step.instruction_id,
                target_poi_id=step.target_poi_id,
                target_type=step.target_type,
                target_geometry=step.target_geometry,
                target_source_round=step.target_source_round,
                reward=step.reward,
                done=step.done,
                observation=step.observation,
                area_id=summary.area_id,
                avatar_track_id=None,
                predicted_avatar_centroid=None,
                actual_avatar_centroid=avatar_candidates[0].centroid if avatar_candidates else None,
                event_ids=list(step.event_ids),
                intervention_id=step.intervention_id,
                action_context_key=summary.navigation_context_key,
                observation_summary=summary,
                info=step.info,
            )
        )
        prev_obs = obs
        prev_objects = objects
    steps_out = _finalize_episode_tracks(steps_out)
    return TrajectoryEpisodeV2(
        schema_version=episode.schema_version,
        game_id=episode.game_id,
        episode_id=episode.episode_id,
        steps=steps_out,
        done=episode.done,
        win=episode.win,
        seed=episode.seed,
        metadata=episode.metadata,
    )


def _apply_area_ids(analyzed: List[TrajectoryEpisodeV2]) -> List[TrajectoryEpisodeV2]:
    known_areas = []
    out = []
    for episode in analyzed:
        steps = []
        for step in episode.steps:
            if step.observation_summary is None:
                steps.append(step)
                continue
            area_id = assign_area_id_to_summary(step.observation_summary, known_areas)
            updated_summary = _clone_summary(step.observation_summary, area_id=area_id or step.observation_summary.area_id)
            steps.append(
                _clone_step(
                    step,
                    observation_summary=updated_summary,
                    area_id=updated_summary.area_id,
                    action_context_key=updated_summary.navigation_context_key,
                )
            )
        out.append(replace(episode, steps=steps))
    return out


def _apply_area_ids_with_table(
    analyzed: List[TrajectoryEpisodeV2],
    area_table,
) -> List[TrajectoryEpisodeV2]:
    out = []
    for episode in analyzed:
        steps = []
        for step in episode.steps:
            if step.observation_summary is None:
                steps.append(step)
                continue
            area_id = assign_area_id_to_summary(step.observation_summary, area_table)
            updated_summary = _clone_summary(step.observation_summary, area_id=area_id or step.observation_summary.area_id)
            steps.append(
                _clone_step(
                    step,
                    observation_summary=updated_summary,
                    area_id=updated_summary.area_id,
                    action_context_key=updated_summary.navigation_context_key,
                )
            )
        out.append(replace(episode, steps=steps))
    return out


def _pool_init_episode_analyzer(analyst_cfg: AnalystConfigV2, avatar_cfg: AvatarTrackingConfigV2) -> None:
    global _WORKER_ANALYST_CFG, _WORKER_AVATAR_CFG
    _WORKER_ANALYST_CFG = analyst_cfg
    _WORKER_AVATAR_CFG = avatar_cfg


def _pool_init_area_assign(area_table) -> None:
    global _WORKER_AREA_TABLE
    _WORKER_AREA_TABLE = area_table


def _analyze_episode_chunk(payload: Tuple[List[TrajectoryEpisodeV2], AnalystConfigV2]) -> List[TrajectoryEpisodeV2]:
    episodes, cfg = payload
    accumulator = AvatarCandidateAccumulator()
    return [analyze_episode(ep, cfg, accumulator) for ep in episodes]


def _analyze_episode_chunk_local(episodes: List[TrajectoryEpisodeV2]) -> List[TrajectoryEpisodeV2]:
    accumulator = AvatarCandidateAccumulator()
    cfg = _WORKER_ANALYST_CFG or AnalystConfigV2()
    return [analyze_episode(ep, cfg, accumulator) for ep in episodes]


def _apply_area_ids_chunk(payload: Tuple[List[TrajectoryEpisodeV2], list]) -> List[TrajectoryEpisodeV2]:
    episodes, area_table = payload
    return _apply_area_ids_with_table(episodes, area_table)


def _apply_area_ids_chunk_local(episodes: List[TrajectoryEpisodeV2]) -> List[TrajectoryEpisodeV2]:
    return _apply_area_ids_with_table(episodes, _WORKER_AREA_TABLE or [])


def _build_episode_chunks(
    episodes: List[TrajectoryEpisodeV2],
    workers: int,
    chunk_size: Optional[int],
) -> List[List[TrajectoryEpisodeV2]]:
    if not episodes:
        return []
    if chunk_size is not None and chunk_size > 0:
        return [episodes[idx : idx + int(chunk_size)] for idx in range(0, len(episodes), int(chunk_size))]
    if len(episodes) <= workers:
        return [[episode] for episode in episodes]
    chunk_count = max(1, min(workers, len(episodes)))
    balanced_chunk_size = max(1, (len(episodes) + chunk_count - 1) // chunk_count)
    return [episodes[idx : idx + balanced_chunk_size] for idx in range(0, len(episodes), balanced_chunk_size)]


def analyze_episodes(
    episodes: List[TrajectoryEpisodeV2],
    cfg: AnalystConfigV2,
    tuning_cfg: Optional[TrajectoryAnalysisConfigV2] = None,
) -> List[TrajectoryEpisodeV2]:
    accumulator = AvatarCandidateAccumulator()
    analyzed = [analyze_episode(ep, cfg, accumulator) for ep in episodes]
    area_table = infer_areas_from_episodes(analyzed)
    return _apply_area_ids_with_table(analyzed, area_table)


def analyze_episodes_parallel(
    episodes: List[TrajectoryEpisodeV2],
    cfg: AnalystConfigV2,
    workers: int,
    pool: Optional[Pool] = None,
    tuning_cfg: Optional[TrajectoryAnalysisConfigV2] = None,
    stats_out: Optional[Dict[str, object]] = None,
) -> List[TrajectoryEpisodeV2]:
    tuning_cfg = tuning_cfg or TrajectoryAnalysisConfigV2()
    effective_workers = int(tuning_cfg.episode_analysis_workers or workers)
    if effective_workers <= 1 or len(episodes) <= 1:
        if stats_out is not None:
            stats_out.update(
                {
                    "episodes": len(episodes),
                    "total_steps": sum(len(ep.steps) for ep in episodes),
                    "avg_steps_per_episode": (sum(len(ep.steps) for ep in episodes) / float(max(1, len(episodes)))) if episodes else 0.0,
                    "workers_used": 1,
                    "effective_chunk_count": 1 if episodes else 0,
                    "multiprocessing_active": False,
                }
            )
        return analyze_episodes(episodes, cfg, tuning_cfg=tuning_cfg)
    chunks = _build_episode_chunks(episodes, effective_workers, tuning_cfg.episode_analysis_chunk_size)
    effective_chunk_count = len(chunks)
    total_steps = sum(len(ep.steps) for ep in episodes)
    multiprocessing_active = pool is not None or effective_chunk_count > 1
    print(
        f"[v2] episode_analysis_mode episodes={len(episodes)} total_steps={total_steps} avg_steps_per_episode={round(total_steps / float(max(1, len(episodes))), 3)} workers_used={effective_workers} effective_chunk_count={effective_chunk_count} multiprocessing_active={str(multiprocessing_active).lower()}",
        flush=True,
    )

    print(f"[v2] episode_analysis_substage_start stage=map episodes={len(episodes)} workers={effective_workers}", flush=True)
    map_t0 = time.perf_counter()
    if pool is not None:
        analyzed_chunks = pool.map(_analyze_episode_chunk, [(chunk, cfg) for chunk in chunks if chunk])
    else:
        payloads = [chunk for chunk in chunks if chunk]
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=min(effective_workers, len(payloads)),
            initializer=_pool_init_episode_analyzer,
            initargs=(cfg, AvatarTrackingConfigV2()),
        ) as local_pool:
            analyzed_chunks = local_pool.map(_analyze_episode_chunk_local, payloads)
    map_s = time.perf_counter() - map_t0
    print(f"[v2] episode_analysis_substage_stop stage=map duration_s={round(map_s, 3)}", flush=True)

    flatten_map_t0 = time.perf_counter()
    analyzed = [episode for chunk in analyzed_chunks for episode in chunk]
    analyzed.sort(key=lambda episode: episode.episode_id)
    flatten_map_s = time.perf_counter() - flatten_map_t0

    print(f"[v2] episode_analysis_substage_start stage=area_table episodes={len(episodes)}", flush=True)
    area_table_t0 = time.perf_counter()
    area_table = infer_areas_from_episodes(analyzed)
    area_table_s = time.perf_counter() - area_table_t0
    print(f"[v2] episode_analysis_substage_stop stage=area_table duration_s={round(area_table_s, 3)}", flush=True)

    assigned_chunks = [analyzed]
    area_assign_s = 0.0
    flatten_assign_s = 0.0
    if tuning_cfg.parallel_area_assign:
        print(f"[v2] episode_analysis_substage_start stage=area_assign_map episodes={len(episodes)} workers={effective_workers}", flush=True)
        area_assign_chunks = _build_episode_chunks(analyzed, effective_workers, tuning_cfg.episode_analysis_chunk_size)
        area_assign_payloads = [(chunk, area_table) for chunk in area_assign_chunks if chunk]
        area_assign_t0 = time.perf_counter()
        if pool is not None:
            assigned_chunks = pool.map(_apply_area_ids_chunk, area_assign_payloads)
        else:
            payloads = [chunk for chunk in area_assign_chunks if chunk]
            ctx = get_context("spawn")
            with ctx.Pool(
                processes=min(effective_workers, len(payloads)),
                initializer=_pool_init_area_assign,
                initargs=(area_table,),
            ) as local_pool:
                assigned_chunks = local_pool.map(_apply_area_ids_chunk_local, payloads)
        area_assign_s = time.perf_counter() - area_assign_t0
        print(f"[v2] episode_analysis_substage_stop stage=area_assign_map duration_s={round(area_assign_s, 3)}", flush=True)
        flatten_assign_t0 = time.perf_counter()
        assigned = [episode for chunk in assigned_chunks for episode in chunk]
        assigned.sort(key=lambda episode: episode.episode_id)
        flatten_assign_s = time.perf_counter() - flatten_assign_t0
    else:
        print(f"[v2] episode_analysis_substage_start stage=area_assign_map episodes={len(episodes)} workers=1", flush=True)
        area_assign_t0 = time.perf_counter()
        assigned = _apply_area_ids_with_table(analyzed, area_table)
        area_assign_s = time.perf_counter() - area_assign_t0
        print(f"[v2] episode_analysis_substage_stop stage=area_assign_map duration_s={round(area_assign_s, 3)}", flush=True)
        flatten_assign_s = 0.0

    if stats_out is not None:
        stats_out.update(
            {
                "map": map_s,
                "flatten_sort_after_map": flatten_map_s,
                "area_table": area_table_s,
                "area_assign_map": area_assign_s,
                "flatten_sort_after_area_assign": flatten_assign_s,
                "episodes": len(episodes),
                "total_steps": total_steps,
                "avg_steps_per_episode": total_steps / float(max(1, len(episodes))),
                "workers_used": effective_workers,
                "effective_chunk_count": effective_chunk_count,
                "multiprocessing_active": multiprocessing_active,
            }
        )
    print(
        f"[v2] episode_analysis_substages episodes={len(episodes)} total_steps={total_steps} avg_steps_per_episode={round(total_steps / float(max(1, len(episodes))), 3)} workers_used={effective_workers} effective_chunk_count={effective_chunk_count} multiprocessing_active={str(multiprocessing_active).lower()} map_s={round(map_s, 3)} flatten_sort_after_map_s={round(flatten_map_s, 3)} area_table_s={round(area_table_s, 3)} area_assign_map_s={round(area_assign_s, 3)} flatten_sort_after_area_assign_s={round(flatten_assign_s, 3)}",
        flush=True,
    )
    return assigned
