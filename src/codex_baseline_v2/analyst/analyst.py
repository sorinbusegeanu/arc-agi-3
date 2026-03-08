from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from codex_baseline_v2.shared.config import AnalystConfigV2
from codex_baseline_v2.analyst.frame_analyst import AvatarCandidateAccumulator, score_avatar_candidates
from codex_baseline_v2.analyst.poi_miner import mine_pois
from codex_baseline_v2.shared.schemas import (
    ObjectRecordV2,
    ObservationSummaryV2,
    SCHEMA_VERSION,
    TrajectoryEpisodeV2,
    TrajectoryStepV2,
)
from codex_baseline_v2.shared.state_identity import canonical_state_identity
from codex_baseline_v2.shared.utils import (
    BBox,
    bbox_from_points,
    connected_components_per_color,
    grid_diff,
    grid_palette,
    merge_bboxes,
)


def _score_background_candidates(grid: List[List[int]], cfg: AnalystConfigV2) -> List[Dict[str, float]]:
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
    comps = connected_components_per_color(grid)
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
        palette = grid_palette(obs)
        bg_candidates = _score_background_candidates(obs, cfg)
        bg_colors = [c["color"] for c in bg_candidates[:2]]
        comps = connected_components_per_color(obs)
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
        # Emit clustered elongated structures for grouped regions.
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
                pre_state_hash=step.pre_state_hash or canonical_state_identity(step.observation, include_payload=False).get("state_hash"),
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
                observation_summary=summary,
                info=step.info,
            )
        )
        prev_obs = obs
        prev_objects = objects
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


def analyze_episodes(episodes: List[TrajectoryEpisodeV2], cfg: AnalystConfigV2) -> List[TrajectoryEpisodeV2]:
    accumulator = AvatarCandidateAccumulator()
    return [analyze_episode(ep, cfg, accumulator) for ep in episodes]
