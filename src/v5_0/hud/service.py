from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Any

from v5_0.contracts.avatar_types import (
    CrossResetHUDEvidence,
    HUDDetectionReport,
    HUDDiagnostics,
    HUDHintDiagnostics,
    HUDHintReport,
    HUDEpisodeReport,
    HUDMask,
    HUDRegion,
    MultiResetAvatarReport,
    HUDTargetSelection,
)
from v5_0.hud.edge_band_analyzer import extract_edge_band_components
from v5_0.hud.hint_summary import build_hud_hint_summaries
from v5_0.hud.poi_matcher import match_hud_hints_to_pois
from v5_0.hud.repeated_change_analyzer import score_repeated_edge_band_changes
from v5_0.hud.target_ranker import rank_pois_from_hud_matches
from v5_0.hud.text_color_sampler import extract_hud_color_summaries, sample_hud_cell_values
from v5_0.hud.world_filter import (
    build_persistent_hud_mask,
    reject_avatar_like_edge_components,
    reject_world_like_edge_components,
)


def detect_hud_for_episode(
    transitions,
    selected_avatar,
    poi_report=None,
    episode_index: int = 0,
    avatar_report=None,
    include_samples: bool = True,
) -> dict[str, Any]:
    if selected_avatar is not None and getattr(selected_avatar, "failure_reason", None) is not None:
        empty_mask = HUDMask(height=0, width=0, true_cell_count=0, rows_active=(), cols_active=(), regions=())
        empty_diag = HUDDiagnostics(
            edge_band_component_count=0,
            repeated_change_component_count=0,
            avatar_overlap_rejections=0,
            world_motion_rejections=0,
            persistent_region_count=0,
            cross_reset_cluster_count=0,
            ambiguous_regions=0,
            text_or_color_sample_count=0,
            static_edge_survivor_count=0,
        )
        return {
            "report": HUDDetectionReport(mask=empty_mask, regions=(), diagnostics=empty_diag, failure_reason="avatar_not_stable"),
            "value_samples": (),
            "value_summaries": {},
        }

    edge_components = extract_edge_band_components(transitions, selected_avatar=selected_avatar, poi_report=poi_report)
    repeat_scores = score_repeated_edge_band_changes(transitions, edge_components)

    scored_components = []
    for item in edge_components:
        metrics = repeat_scores.get(
            item["hud_region_id"],
            {
                "stability_score": 0.0,
                "persistence_score": 0.0,
                "change_repeat_score": 0.0,
                "positional_stability": 0.0,
                "same_side_stability": 0.0,
                "edge_lock_score": 0.0,
                "world_coupling_ratio": 0.0,
            },
        )
        scored = dict(item)
        scored["stability_score"] = float(metrics.get("stability_score", 0.0))
        scored["persistence_score"] = float(metrics.get("persistence_score", 0.0))
        scored["change_repeat_score"] = float(metrics.get("change_repeat_score", 0.0))
        scored["positional_stability"] = float(metrics.get("positional_stability", 0.0))
        scored["same_side_stability"] = float(metrics.get("same_side_stability", 0.0))
        scored["edge_lock_score"] = float(metrics.get("edge_lock_score", scored.get("edge_locked_fraction", 0.0)))
        scored["world_coupling_ratio"] = float(metrics.get("world_coupling_ratio", 0.0))
        scored_components.append(scored)

    avatar_filtered, avatar_rejections = reject_avatar_like_edge_components(
        tuple(scored_components),
        selected_avatar,
        avatar_report=avatar_report,
        episode_transitions=transitions,
    )
    world_filtered, world_rejections = reject_world_like_edge_components(
        avatar_filtered,
        selected_avatar,
        avatar_reports=avatar_report,
        poi_report=poi_report,
        episode_transitions=transitions,
    )

    enriched = tuple(_with_confidence(item) for item in world_filtered)
    with_static_flag = tuple(_with_static_edge_hud_flag(item) for item in enriched)
    mask = build_persistent_hud_mask(with_static_flag, transitions, min_persistence_steps=2)
    surviving = tuple(item for item in with_static_flag if str(item["hud_region_id"]) in set(mask.regions))
    regions = _to_regions(surviving, episode_index)

    if len(regions) > 1 and abs(regions[0].confidence - regions[1].confidence) <= 0.05:
        regions = _mark_ambiguous(regions)

    value_samples = sample_hud_cell_values(mask, regions, transitions, episode_index) if include_samples else ()
    value_summaries = extract_hud_color_summaries(value_samples, regions) if include_samples else {}

    diagnostics = HUDDiagnostics(
        edge_band_component_count=len(edge_components),
        repeated_change_component_count=sum(1 for item in edge_components if item["change_step_indices"]),
        avatar_overlap_rejections=int(avatar_rejections),
        world_motion_rejections=int(world_rejections),
        persistent_region_count=len(regions),
        cross_reset_cluster_count=0,
        ambiguous_regions=sum(1 for region in regions if "close_score" in set(region.ambiguity_flags)),
        text_or_color_sample_count=len(value_samples),
        static_edge_survivor_count=sum(1 for item in surviving if bool(item.get("static_edge_hud", False))),
    )
    static_survivors_with_mask = bool(mask.true_cell_count > 0 and any(bool(item.get("static_edge_hud", False)) for item in surviving))
    failure_reason = None if regions or static_survivors_with_mask else "no_hud_region"
    report = HUDDetectionReport(mask=mask, regions=regions, diagnostics=diagnostics, failure_reason=failure_reason)
    return {
        "report": report,
        "value_samples": value_samples,
        "value_summaries": value_summaries,
    }


def detect_hud_multi_reset(
    avatar_multi_report: MultiResetAvatarReport,
    poi_multi_bundle=None,
) -> dict[str, Any]:
    poi_by_episode = {}
    if isinstance(poi_multi_bundle, dict):
        poi_by_episode = {item.episode_index: item.poi_report for item in poi_multi_bundle.get("episodes", ())}

    episode_reports: list[HUDEpisodeReport] = []
    episode_samples: dict[int, tuple] = {}
    successful: list[tuple[int, HUDDetectionReport]] = []

    for episode in avatar_multi_report.episodes:
        poi_report = poi_by_episode.get(episode.episode_index)
        result = detect_hud_for_episode(
            episode.transitions,
            episode.report.selected,
            poi_report=poi_report,
            episode_index=episode.episode_index,
            avatar_report=episode.report,
            include_samples=True,
        )
        report = result["report"]
        episode_reports.append(HUDEpisodeReport(episode_index=episode.episode_index, hud_report=report))
        episode_samples[episode.episode_index] = result["value_samples"]
        if episode.report.selected.failure_reason is None and report.failure_reason is None:
            successful.append((episode.episode_index, report))

    if avatar_multi_report.selected.failure_reason is not None:
        empty_mask = HUDMask(height=0, width=0, true_cell_count=0, rows_active=(), cols_active=(), regions=())
        diagnostics = HUDDiagnostics(
            edge_band_component_count=0,
            repeated_change_component_count=0,
            avatar_overlap_rejections=0,
            world_motion_rejections=0,
            persistent_region_count=0,
            cross_reset_cluster_count=0,
            ambiguous_regions=0,
            text_or_color_sample_count=0,
            static_edge_survivor_count=0,
        )
        final_report = HUDDetectionReport(mask=empty_mask, regions=(), diagnostics=diagnostics, failure_reason="avatar_not_stable")
        return {
            "episodes": tuple(episode_reports),
            "cross_reset_evidence": (),
            "report": final_report,
            "value_samples": episode_samples,
        }

    clusters = _cluster_regions(successful)
    evidence = _build_cross_reset_evidence(clusters)
    final_regions = _regions_from_evidence(evidence)
    final_mask = _build_cross_reset_mask(final_regions, episode_reports)
    diagnostics = HUDDiagnostics(
        edge_band_component_count=sum(ep.hud_report.diagnostics.edge_band_component_count for ep in episode_reports),
        repeated_change_component_count=sum(ep.hud_report.diagnostics.repeated_change_component_count for ep in episode_reports),
        avatar_overlap_rejections=sum(ep.hud_report.diagnostics.avatar_overlap_rejections for ep in episode_reports),
        world_motion_rejections=sum(ep.hud_report.diagnostics.world_motion_rejections for ep in episode_reports),
        persistent_region_count=len(final_regions),
        cross_reset_cluster_count=len(evidence),
        ambiguous_regions=sum(1 for item in final_regions if "close_score" in set(item.ambiguity_flags)),
        text_or_color_sample_count=sum(len(samples) for samples in episode_samples.values()),
        static_edge_survivor_count=sum(ep.hud_report.diagnostics.static_edge_survivor_count for ep in episode_reports),
    )
    failure_reason = None if final_regions else "no_hud_region"
    return {
        "episodes": tuple(episode_reports),
        "cross_reset_evidence": evidence,
        "report": HUDDetectionReport(mask=final_mask, regions=final_regions, diagnostics=diagnostics, failure_reason=failure_reason),
        "value_samples": episode_samples,
    }


def _with_confidence(component: dict[str, Any]) -> dict[str, Any]:
    stability = float(component.get("stability_score", 0.0))
    persistence = float(component.get("persistence_score", 0.0))
    edge_lock = float(component.get("edge_locked_fraction", component.get("edge_lock_score", 0.0)))
    positional = float(component.get("positional_stability", 0.0))
    repeat_score = float(component.get("change_repeat_score", 0.0))
    world_penalty = float(component.get("world_motion_penalty", 0.0))
    confidence = max(
        0.0,
        min(
            1.0,
            0.45 * stability + 0.25 * persistence + 0.15 * edge_lock + 0.12 * positional + 0.08 * repeat_score - 0.25 * world_penalty,
        ),
    )
    world_rej_score = max(0.0, min(1.0, 1.0 - world_penalty))
    out = dict(component)
    out["confidence"] = float(confidence)
    out["world_overlap_rejection_score"] = float(world_rej_score)
    return out


def _with_static_edge_hud_flag(component: dict[str, Any]) -> dict[str, Any]:
    out = dict(component)
    persistence_steps = len(set(int(v) for v in component.get("seen_step_indices", ())))
    edge_locked = float(component.get("edge_locked_fraction", 0.0)) >= 0.75
    positional = float(component.get("positional_stability", 0.0))
    world_penalty = float(component.get("world_motion_penalty", 0.0))
    out["static_edge_hud"] = bool(edge_locked and persistence_steps >= 2 and positional >= 0.4 and world_penalty <= 0.9)
    return out


def _to_regions(components: tuple[dict[str, Any], ...], episode_index: int) -> tuple[HUDRegion, ...]:
    out: list[HUDRegion] = []
    for item in sorted(components, key=lambda v: (-float(v.get("confidence", 0.0)), str(v["hud_region_id"]))):
        out.append(
            HUDRegion(
                hud_region_id=str(item["hud_region_id"]),
                bbox=tuple(item["bbox"]),
                center=tuple(item["center"]),
                area=int(item["area"]),
                edge_side=str(item["edge_side"]),
                value_histogram=dict(sorted((int(k), int(v)) for k, v in item["value_histogram"].items())),
                seen_episode_indices=(int(episode_index),),
                seen_step_indices=tuple(int(v) for v in item.get("seen_step_indices", ())),
                change_step_indices=tuple(int(v) for v in item.get("change_step_indices", ())),
                stability_score=float(item.get("stability_score", 0.0)),
                change_repeat_score=float(item.get("change_repeat_score", 0.0)),
                world_overlap_rejection_score=float(item.get("world_overlap_rejection_score", 0.0)),
                confidence=float(item.get("confidence", 0.0)),
                ambiguity_flags=(),
            )
        )
    return tuple(out)


def _mark_ambiguous(regions: tuple[HUDRegion, ...]) -> tuple[HUDRegion, ...]:
    if len(regions) < 2:
        return regions
    first = regions[0]
    second = regions[1]
    first = HUDRegion(**{**first.to_dict(), "ambiguity_flags": tuple(sorted(set(first.ambiguity_flags) | {"close_score"}))})
    second = HUDRegion(**{**second.to_dict(), "ambiguity_flags": tuple(sorted(set(second.ambiguity_flags) | {"close_score"}))})
    return (first, second, *regions[2:])


def _cluster_regions(successful: list[tuple[int, HUDDetectionReport]]) -> tuple[tuple[tuple[int, HUDRegion], ...], ...]:
    clusters: list[list[tuple[int, HUDRegion]]] = []
    for episode_index, report in successful:
        for region in report.regions:
            best_index = None
            best_score = -1.0
            for index, cluster in enumerate(clusters):
                reference = cluster[-1][1]
                score = _region_similarity(reference, region)
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index is not None and best_score >= 0.55:
                clusters[best_index].append((episode_index, region))
            else:
                clusters.append([(episode_index, region)])
    normalized = [tuple(sorted(cluster, key=lambda item: item[0])) for cluster in clusters]
    normalized.sort(key=lambda cluster: (-len(cluster), -sum(item[1].confidence for item in cluster)))
    return tuple(normalized)


def _build_cross_reset_evidence(clusters) -> tuple[CrossResetHUDEvidence, ...]:
    out: list[CrossResetHUDEvidence] = []
    for index, cluster in enumerate(clusters):
        regions = tuple(item[1] for item in cluster)
        hist = Counter()
        for region in regions:
            hist.update(region.value_histogram)
        mean_conf = sum(region.confidence for region in regions) / max(len(regions), 1)
        out.append(
            CrossResetHUDEvidence(
                canonical_region_id=f"cross_hud_{index:03d}",
                episode_indices=tuple(item[0] for item in cluster),
                per_episode_region_ids={item[0]: item[1].hud_region_id for item in cluster},
                bbox_sequence=tuple(item[1].bbox for item in cluster),
                center_sequence=tuple(item[1].center for item in cluster),
                value_histogram_aggregate=dict(sorted((int(k), int(v)) for k, v in hist.items())),
                mean_confidence=float(mean_conf),
                position_consistency_across_resets=_position_consistency(regions),
                support_episode_count=len(regions),
            )
        )
    out.sort(key=lambda item: (-item.support_episode_count, -item.mean_confidence, -item.position_consistency_across_resets))
    return tuple(out)


def _regions_from_evidence(evidence: tuple[CrossResetHUDEvidence, ...]) -> tuple[HUDRegion, ...]:
    out: list[HUDRegion] = []
    for item in evidence:
        if not item.bbox_sequence:
            continue
        bbox = item.bbox_sequence[-1]
        out.append(
            HUDRegion(
                hud_region_id=item.canonical_region_id,
                bbox=bbox,
                center=item.center_sequence[-1],
                area=(bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1),
                edge_side=_edge_side_from_bbox(bbox),
                value_histogram=item.value_histogram_aggregate,
                seen_episode_indices=item.episode_indices,
                seen_step_indices=(),
                change_step_indices=(),
                stability_score=float(item.position_consistency_across_resets),
                change_repeat_score=float(min(1.0, item.support_episode_count / 3.0)),
                world_overlap_rejection_score=0.0,
                confidence=float(item.mean_confidence),
                ambiguity_flags=(),
            )
        )
    out.sort(key=lambda item: (-item.confidence, -len(item.seen_episode_indices), item.hud_region_id))
    if len(out) > 1 and abs(out[0].confidence - out[1].confidence) <= 0.05:
        out[0] = HUDRegion(**{**out[0].to_dict(), "ambiguity_flags": ("close_score",)})
        out[1] = HUDRegion(**{**out[1].to_dict(), "ambiguity_flags": ("close_score",)})
    return tuple(out)


def _build_cross_reset_mask(regions: tuple[HUDRegion, ...], episode_reports: list[HUDEpisodeReport]) -> HUDMask:
    height = max((episode.hud_report.mask.height for episode in episode_reports), default=0)
    width = max((episode.hud_report.mask.width for episode in episode_reports), default=0)
    active: set[tuple[int, int]] = set()
    for region in regions:
        x0, y0, x1, y1 = region.bbox
        for y in range(max(0, y0), min(height, y1 + 1)):
            for x in range(max(0, x0), min(width, x1 + 1)):
                active.add((x, y))
    return HUDMask(
        height=int(height),
        width=int(width),
        true_cell_count=len(active),
        rows_active=tuple(sorted({y for _, y in active})),
        cols_active=tuple(sorted({x for x, _ in active})),
        regions=tuple(region.hud_region_id for region in regions),
    )


def _region_similarity(left: HUDRegion, right: HUDRegion) -> float:
    if left.edge_side != right.edge_side:
        return 0.0
    iou = _bbox_iou(left.bbox, right.bbox)
    dist = sqrt((left.center[0] - right.center[0]) ** 2 + (left.center[1] - right.center[1]) ** 2)
    center_sim = max(0.0, 1.0 - dist / 8.0)
    hist_sim = _hist_similarity(left.value_histogram, right.value_histogram)
    border_loc_sim = _border_location_similarity(left, right)
    drift_sim = _bbox_drift_similarity(left.bbox, right.bbox)
    return 0.28 * iou + 0.24 * center_sim + 0.24 * hist_sim + 0.16 * border_loc_sim + 0.08 * drift_sim


def _position_consistency(regions: tuple[HUDRegion, ...]) -> float:
    if len(regions) <= 1:
        return 1.0
    mean_x = sum(item.center[0] for item in regions) / len(regions)
    mean_y = sum(item.center[1] for item in regions) / len(regions)
    dist = [sqrt((item.center[0] - mean_x) ** 2 + (item.center[1] - mean_y) ** 2) for item in regions]
    return max(0.0, min(1.0, 1.0 - (sum(dist) / len(dist)) / 8.0))


def _edge_side_from_bbox(bbox: tuple[int, int, int, int]) -> str:
    x0, y0, x1, y1 = bbox
    if y0 <= x0 and y0 <= (x1 - x0) and y0 <= y1:
        return "top"
    if x0 <= y0 and x0 <= (y1 - y0):
        return "left"
    return "bottom" if y1 >= x1 else "right"


def _bbox_iou(left, right) -> float:
    ix0 = max(left[0], right[0])
    iy0 = max(left[1], right[1])
    ix1 = min(left[2], right[2])
    iy1 = min(left[3], right[3])
    if ix1 < ix0 or iy1 < iy0:
        return 0.0
    inter = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
    l_area = (left[2] - left[0] + 1) * (left[3] - left[1] + 1)
    r_area = (right[2] - right[0] + 1) * (right[3] - right[1] + 1)
    return inter / max(l_area + r_area - inter, 1)


def _hist_similarity(left: dict[int, int], right: dict[int, int]) -> float:
    if not left or not right:
        return 0.0
    keys = set(left) | set(right)
    overlap = sum(min(int(left.get(k, 0)), int(right.get(k, 0))) for k in keys)
    total = sum(max(int(left.get(k, 0)), int(right.get(k, 0))) for k in keys)
    return overlap / max(total, 1)


def _border_location_similarity(left: HUDRegion, right: HUDRegion) -> float:
    lx0, ly0, lx1, ly1 = left.bbox
    rx0, ry0, rx1, ry1 = right.bbox
    if left.edge_side in {"top", "bottom"}:
        left_span = (lx0 + lx1) / 2.0
        right_span = (rx0 + rx1) / 2.0
    else:
        left_span = (ly0 + ly1) / 2.0
        right_span = (ry0 + ry1) / 2.0
    delta = abs(left_span - right_span)
    return max(0.0, min(1.0, 1.0 - delta / 10.0))


def _bbox_drift_similarity(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lw = max(1, left[2] - left[0] + 1)
    lh = max(1, left[3] - left[1] + 1)
    rw = max(1, right[2] - right[0] + 1)
    rh = max(1, right[3] - right[1] + 1)
    area_left = lw * lh
    area_right = rw * rh
    ratio = max(area_left, area_right) / max(1.0, min(area_left, area_right))
    return max(0.0, min(1.0, 1.0 - (ratio - 1.0) / 3.0))


def interpret_hud_hints_multi_reset(
    hud_bundle,
    poi_bundle,
) -> HUDHintReport:
    hud_report = hud_bundle.get("report") if isinstance(hud_bundle, dict) else None
    poi_report = poi_bundle.get("report") if isinstance(poi_bundle, dict) else None

    regions = tuple(getattr(hud_report, "regions", ()))
    if not regions:
        return HUDHintReport(
            hud_hints=(),
            matches=(),
            selected=HUDTargetSelection(
                selected_poi_id=None,
                ranked_poi_ids=(),
                top_match_hud_region_ids=(),
                ambiguous=False,
                failure_reason="no_hud_region",
            ),
            diagnostics=HUDHintDiagnostics(
                hud_region_count=0,
                poi_candidate_count=len(getattr(poi_report, "candidates", ())) if poi_report is not None else 0,
                match_count=0,
                ambiguous_match_count=0,
                rejected_match_reasons={},
                selected_match_margin=0.0,
                cross_reset_support_count=0,
            ),
        )

    poi_candidates = tuple(getattr(poi_report, "candidates", ()))
    if not poi_candidates:
        return HUDHintReport(
            hud_hints=(),
            matches=(),
            selected=HUDTargetSelection(
                selected_poi_id=None,
                ranked_poi_ids=(),
                top_match_hud_region_ids=(),
                ambiguous=False,
                failure_reason="no_poi_candidate",
            ),
            diagnostics=HUDHintDiagnostics(
                hud_region_count=len(regions),
                poi_candidate_count=0,
                match_count=0,
                ambiguous_match_count=0,
                rejected_match_reasons={},
                selected_match_margin=0.0,
                cross_reset_support_count=0,
            ),
        )

    hints = build_hud_hint_summaries(
        regions,
        getattr(hud_report, "mask", None),
        raw_hud_value_samples=hud_bundle.get("value_samples", ()) if isinstance(hud_bundle, dict) else (),
        cross_reset_hud_evidence=hud_bundle.get("cross_reset_evidence", ()) if isinstance(hud_bundle, dict) else (),
    )
    match_out = match_hud_hints_to_pois(
        hints,
        poi_candidates,
        cross_reset_poi_evidence=poi_bundle.get("cross_reset_evidence", ()) if isinstance(poi_bundle, dict) else (),
    )
    matches = tuple(match_out.get("matches", ()))
    matches_by_hud_region: dict[str, list] = {}
    for match in matches:
        matches_by_hud_region.setdefault(str(match.hud_region_id), []).append(match)
    for region_id in matches_by_hud_region:
        matches_by_hud_region[region_id] = sorted(
            matches_by_hud_region[region_id],
            key=lambda item: (-item.confidence, item.poi_id),
        )

    selection, selection_details = rank_pois_from_hud_matches(
        matches,
        poi_candidates,
        matches_by_hud_region=matches_by_hud_region,
        return_details=True,
    )
    selected_margin = float(selection_details.get("selected_match_margin", 0.0))

    return HUDHintReport(
        hud_hints=hints,
        matches=matches,
        selected=selection,
        diagnostics=HUDHintDiagnostics(
            hud_region_count=len(hints),
            poi_candidate_count=len(poi_candidates),
            match_count=len(matches),
            ambiguous_match_count=sum(1 for match in matches if "close_score" in set(match.ambiguity_flags)),
            rejected_match_reasons=dict(match_out.get("rejected_match_reasons", {})),
            selected_match_margin=float(selected_margin),
            cross_reset_support_count=max((int(match.support_episode_count) for match in matches), default=0),
            best_match_win_counts=dict(selection_details.get("best_match_win_counts", {})),
            aggregate_score_gap=float(selection_details.get("aggregate_score_gap", 0.0)),
        ),
    )
