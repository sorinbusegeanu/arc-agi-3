from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

from codex_baseline_v2.shared.config import AreaModelConfigV2
from codex_baseline_v2.shared.schemas import AreaStateV2, ObservationSummaryV2, SCHEMA_VERSION, TrajectoryEpisodeV2
from codex_baseline_v2.shared.utils import normalize_palette


def _palette_overlap(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(v) for v in a)
    sb = set(int(v) for v in b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / float(max(1, len(sa | sb)))


def _stable_object_set(summary: ObservationSummaryV2) -> Set[str]:
    return {obj.object_id for obj in summary.objects if obj.object_class != "hud_like"}


def _stable_similarity(summary: ObservationSummaryV2, area: AreaStateV2) -> float:
    current = _stable_object_set(summary)
    prior = set(area.stable_object_ids)
    if not current and not prior:
        return 1.0
    return len(current & prior) / float(max(1, len(current | prior)))


def _topology_similarity(summary: ObservationSummaryV2, area: AreaStateV2) -> float:
    current_count = len(summary.world_region_candidates)
    prior_count = len(area.stable_object_ids) + len(area.dynamic_object_ids)
    if current_count == 0 and prior_count == 0:
        return 1.0
    return 1.0 - abs(current_count - prior_count) / float(max(1, max(current_count, prior_count)))


def assign_area_id_to_summary(summary: ObservationSummaryV2, area_table: List[AreaStateV2]) -> Optional[str]:
    best_area_id: Optional[str] = None
    best_score = 0.0
    for area in area_table:
        score = 0.0
        if summary.state_signature_id and summary.state_signature_id == area.canonical_observation_hash:
            score += 0.6
        score += 0.25 * _palette_overlap(summary.palette, area.palette)
        score += 0.1 * _stable_similarity(summary, area)
        score += 0.05 * _topology_similarity(summary, area)
        if score > best_score:
            best_area_id = area.area_id
            best_score = score
    return best_area_id if best_score >= 0.55 else None


def _new_area(game_id: str, summary: ObservationSummaryV2) -> AreaStateV2:
    return AreaStateV2(
        schema_version=SCHEMA_VERSION,
        game_id=game_id,
        area_id="area:%s:%s" % (summary.episode_id, summary.step_idx),
        canonical_observation_hash=summary.state_signature_id,
        palette=normalize_palette(summary.palette),
        width=max([bbox.x2 for bbox in summary.world_region_candidates], default=-1) + 1,
        height=max([bbox.y2 for bbox in summary.world_region_candidates], default=-1) + 1,
        entry_cells=[],
        exit_cells=[],
        stable_object_ids=sorted(_stable_object_set(summary)),
        dynamic_object_ids=sorted(obj.object_id for obj in summary.avatar_candidates),
        topology_signature_id="topo:%s:%s" % (len(summary.world_region_candidates), len(summary.active_regions)),
        visit_count=1,
        confidence=0.55,
    )


def infer_areas_from_episodes(
    episodes: List[TrajectoryEpisodeV2],
    existing_areas: Optional[List[AreaStateV2]] = None,
) -> list[AreaStateV2]:
    areas: Dict[str, AreaStateV2] = {row.area_id: row for row in (existing_areas or [])}
    for episode in episodes:
        for step in episode.steps:
            summary = step.observation_summary
            if summary is None:
                continue
            matched_area_id = assign_area_id_to_summary(summary, list(areas.values()))
            if matched_area_id is None:
                area = _new_area(episode.game_id, summary)
                areas[area.area_id] = area
                continue
            prior = areas[matched_area_id]
            areas[matched_area_id] = AreaStateV2(
                schema_version=SCHEMA_VERSION,
                game_id=prior.game_id,
                area_id=prior.area_id,
                canonical_observation_hash=prior.canonical_observation_hash or summary.state_signature_id,
                palette=normalize_palette(prior.palette + summary.palette),
                width=max(prior.width, max([bbox.x2 for bbox in summary.world_region_candidates], default=-1) + 1),
                height=max(prior.height, max([bbox.y2 for bbox in summary.world_region_candidates], default=-1) + 1),
                entry_cells=prior.entry_cells,
                exit_cells=prior.exit_cells,
                stable_object_ids=sorted(set(prior.stable_object_ids) | _stable_object_set(summary)),
                dynamic_object_ids=sorted(set(prior.dynamic_object_ids) | {obj.object_id for obj in summary.avatar_candidates}),
                topology_signature_id=prior.topology_signature_id or "topo:%s:%s" % (len(summary.world_region_candidates), len(summary.active_regions)),
                visit_count=prior.visit_count + 1,
                confidence=min(1.0, prior.confidence + 0.05),
            )
    return sorted(areas.values(), key=lambda row: row.area_id)


def merge_area_table(existing_areas: List[AreaStateV2], new_areas: List[AreaStateV2]) -> list[AreaStateV2]:
    merged: Dict[str, AreaStateV2] = {row.area_id: row for row in existing_areas}
    for area in new_areas:
        if area.area_id in merged:
            prior = merged[area.area_id]
            merged[area.area_id] = AreaStateV2(
                schema_version=SCHEMA_VERSION,
                game_id=prior.game_id,
                area_id=prior.area_id,
                canonical_observation_hash=prior.canonical_observation_hash or area.canonical_observation_hash,
                palette=normalize_palette(prior.palette + area.palette),
                width=max(prior.width, area.width),
                height=max(prior.height, area.height),
                entry_cells=sorted(set(prior.entry_cells + area.entry_cells)),
                exit_cells=sorted(set(prior.exit_cells + area.exit_cells)),
                stable_object_ids=sorted(set(prior.stable_object_ids) | set(area.stable_object_ids)),
                dynamic_object_ids=sorted(set(prior.dynamic_object_ids) | set(area.dynamic_object_ids)),
                topology_signature_id=prior.topology_signature_id or area.topology_signature_id,
                visit_count=prior.visit_count + area.visit_count,
                confidence=max(prior.confidence, area.confidence),
            )
            continue
        duplicate = None
        for prior in merged.values():
            score = 0.45 * _palette_overlap(prior.palette, area.palette)
            score += 0.35 * (len(set(prior.stable_object_ids) & set(area.stable_object_ids)) / float(max(1, len(set(prior.stable_object_ids) | set(area.stable_object_ids)))))
            score += 0.2 * (1.0 if prior.topology_signature_id and prior.topology_signature_id == area.topology_signature_id else 0.0)
            if score >= 0.65:
                duplicate = prior.area_id
                break
        if duplicate is None:
            merged[area.area_id] = area
            continue
        prior = merged[duplicate]
        merged[duplicate] = AreaStateV2(
            schema_version=SCHEMA_VERSION,
            game_id=prior.game_id,
            area_id=prior.area_id,
            canonical_observation_hash=prior.canonical_observation_hash or area.canonical_observation_hash,
            palette=normalize_palette(prior.palette + area.palette),
            width=max(prior.width, area.width),
            height=max(prior.height, area.height),
            entry_cells=sorted(set(prior.entry_cells + area.entry_cells)),
            exit_cells=sorted(set(prior.exit_cells + area.exit_cells)),
            stable_object_ids=sorted(set(prior.stable_object_ids) | set(area.stable_object_ids)),
            dynamic_object_ids=sorted(set(prior.dynamic_object_ids) | set(area.dynamic_object_ids)),
            topology_signature_id=prior.topology_signature_id or area.topology_signature_id,
            visit_count=prior.visit_count + area.visit_count,
            confidence=max(prior.confidence, area.confidence),
        )
    return sorted(merged.values(), key=lambda row: row.area_id)
