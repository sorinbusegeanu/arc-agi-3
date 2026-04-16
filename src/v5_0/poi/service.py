from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Any

from v5_0.contracts.avatar_types import (
    AvatarIdentificationReport,
    CrossResetPOIEvidence,
    MultiResetAvatarReport,
    POICandidate,
    POIDiagnostics,
    POIDiscoveryReport,
    POIEpisode,
    POISelectedResult,
    ProbeTransitionRecord,
)
from v5_0.poi.candidate_extractor import extract_poi_components
from v5_0.poi.contact_logger import build_contact_logs
from v5_0.poi.merger import merge_poi_candidates
from v5_0.poi.ranker import rank_poi_candidates
from v5_0.poi.static_inventory import build_static_object_inventory


def discover_pois_for_episode(
    transitions: tuple[ProbeTransitionRecord, ...],
    avatar_report: AvatarIdentificationReport,
    episode_index: int,
) -> POIDiscoveryReport:
    if avatar_report.selected.failure_reason is not None:
        return POIDiscoveryReport(
            candidates=(),
            selected=POISelectedResult(
                selected_poi_ids=(),
                ambiguous=True,
                failure_reason="avatar_not_stable",
            ),
            diagnostics=POIDiagnostics(
                per_step_component_counts={},
                static_inventory_count=0,
                changed_component_count=0,
                merged_candidate_count=0,
                cross_reset_cluster_count=0,
                dropped_candidate_reasons={},
                avatar_overlap_rejections=0,
                ambiguous_candidates=0,
                contact_log_count=0,
                border_locked_rejections=0,
            ),
            contact_logs=(),
        )

    changed_components, changed_dropped = extract_poi_components(
        transitions,
        avatar_report.selected,
        avatar_report=avatar_report,
    )
    static_inventory, static_dropped = build_static_object_inventory(
        transitions,
        avatar_report.selected,
    )
    merged = merge_poi_candidates(changed_components, static_inventory)
    ranked, ambiguous_count = rank_poi_candidates(merged, avatar_report.selected.selected_bbox)

    contact_logs = build_contact_logs(
        episode_index=episode_index,
        transitions=transitions,
        poi_candidates=ranked,
        avatar_bbox=avatar_report.selected.selected_bbox,
    )

    selected_ids = tuple(candidate.poi_id for candidate in ranked[:3])
    ambiguous = bool(ambiguous_count > 0)
    selected = POISelectedResult(
        selected_poi_ids=selected_ids,
        ambiguous=ambiguous,
        failure_reason=(None if ranked else "no_poi_candidate"),
    )

    merged_dropped = Counter(changed_dropped)
    merged_dropped.update(static_dropped)
    diagnostics = POIDiagnostics(
        per_step_component_counts=_per_step_counts(changed_components),
        static_inventory_count=len(static_inventory),
        changed_component_count=len(changed_components),
        merged_candidate_count=len(merged),
        cross_reset_cluster_count=0,
        dropped_candidate_reasons=dict(sorted(merged_dropped.items())),
        avatar_overlap_rejections=int(changed_dropped.get("avatar_overlap", 0) + static_dropped.get("avatar_overlap", 0)),
        ambiguous_candidates=ambiguous_count,
        contact_log_count=len(contact_logs),
        border_locked_rejections=int(
            changed_dropped.get("border_locked_nonworld_candidate", 0)
            + static_dropped.get("border_locked_static_candidate", 0)
        ),
    )
    return POIDiscoveryReport(
        candidates=ranked,
        selected=selected,
        diagnostics=diagnostics,
        contact_logs=contact_logs,
    )


def discover_pois_multi_reset(
    avatar_multi_report: MultiResetAvatarReport,
) -> dict[str, Any]:
    poi_episodes: list[POIEpisode] = []
    successful_reports: list[tuple[int, POIDiscoveryReport]] = []
    for episode in avatar_multi_report.episodes:
        poi_report = discover_pois_for_episode(
            episode.transitions,
            episode.report,
            episode.episode_index,
        )
        poi_episodes.append(
            POIEpisode(
                episode_index=episode.episode_index,
                avatar_report=episode.report,
                poi_report=poi_report,
            )
        )
        if episode.report.selected.failure_reason is None:
            successful_reports.append((episode.episode_index, poi_report))

    clusters = _cluster_pois_across_resets(successful_reports)
    cross_evidence = _build_cross_reset_poi_evidence(clusters)
    hud_histograms = _extract_hud_candidate_histograms(successful_reports)

    # Get frame dimensions from first episode's transitions
    frame_width, frame_height = 64, 64  # Default
    if avatar_multi_report.episodes and avatar_multi_report.episodes[0].transitions:
        first_trans = avatar_multi_report.episodes[0].transitions[0]
        if first_trans.post_frame:
            frame_height = len(first_trans.post_frame)
            frame_width = len(first_trans.post_frame[0]) if frame_height > 0 else 64

    ranked_candidates = _rank_cross_reset_candidates(
        cross_evidence,
        hud_histograms=hud_histograms,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    final_candidates, dropped_redundant_ids = _deduplicate_redundant_candidates(ranked_candidates)
    combined_logs = tuple(
        item
        for episode in poi_episodes
        for item in episode.poi_report.contact_logs
    )

    selected = POISelectedResult(
        selected_poi_ids=tuple(candidate.poi_id for candidate in final_candidates[:5]),
        ambiguous=(len(final_candidates) > 1 and abs(final_candidates[0].confidence - final_candidates[1].confidence) <= 0.06),
        failure_reason=(None if final_candidates else "no_poi_candidate"),
    )
    diagnostics = POIDiagnostics(
        per_step_component_counts={},
        static_inventory_count=0,
        changed_component_count=0,
        merged_candidate_count=len(final_candidates),
        cross_reset_cluster_count=len(cross_evidence),
        dropped_candidate_reasons={
            **({"redundant_poi_dropped": len(dropped_redundant_ids)} if dropped_redundant_ids else {}),
            **{f"redundant_poi_id:{poi_id}": 1 for poi_id in dropped_redundant_ids},
        },
        avatar_overlap_rejections=0,
        ambiguous_candidates=sum(1 for candidate in final_candidates if "close_score" in set(candidate.ambiguity_flags)),
        contact_log_count=len(combined_logs),
        border_locked_rejections=sum(
            int(episode.poi_report.diagnostics.border_locked_rejections)
            for episode in poi_episodes
        ),
    )
    final_report = POIDiscoveryReport(
        candidates=final_candidates,
        selected=selected,
        diagnostics=diagnostics,
        contact_logs=combined_logs,
    )
    return {
        "episodes": tuple(poi_episodes),
        "cross_reset_evidence": cross_evidence,
        "report": final_report,
    }


def _per_step_counts(changed_components: tuple[dict[str, Any], ...]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for item in changed_components:
        counts[int(item["step_index"])] += 1
    return dict(sorted(counts.items()))


def _cluster_pois_across_resets(
    successful_reports: list[tuple[int, POIDiscoveryReport]],
) -> tuple[tuple[tuple[int, POICandidate], ...], ...]:
    clusters: list[list[tuple[int, POICandidate]]] = []
    for episode_index, report in successful_reports:
        for candidate in report.candidates:
            matched_index = None
            best = -1.0
            for index, cluster in enumerate(clusters):
                ref = cluster[-1][1]
                score = _poi_similarity(ref, candidate)
                if score > best:
                    best = score
                    matched_index = index
            if matched_index is not None and best >= 0.5:
                clusters[matched_index].append((episode_index, candidate))
            else:
                clusters.append([(episode_index, candidate)])
    normalized = [tuple(sorted(cluster, key=lambda item: (item[0], item[1].poi_id))) for cluster in clusters]
    normalized.sort(key=lambda cluster: (-len(cluster), -sum(item[1].confidence for item in cluster)))
    return tuple(normalized)


def _build_cross_reset_poi_evidence(
    clusters: tuple[tuple[tuple[int, POICandidate], ...], ...],
) -> tuple[CrossResetPOIEvidence, ...]:
    output: list[CrossResetPOIEvidence] = []
    for index, cluster in enumerate(clusters):
        episode_indices = tuple(item[0] for item in cluster)
        candidates = tuple(item[1] for item in cluster)
        hist = Counter()
        for candidate in candidates:
            hist.update(candidate.value_histogram)
        mean_conf = sum(candidate.confidence for candidate in candidates) / max(len(candidates), 1)
        output.append(
            CrossResetPOIEvidence(
                canonical_poi_id=f"cross_poi_{index:03d}",
                episode_indices=episode_indices,
                per_episode_poi_ids={episode_idx: candidate.poi_id for episode_idx, candidate in cluster},
                bbox_sequence=tuple(candidate.bbox for candidate in candidates),
                center_sequence=tuple(candidate.center for candidate in candidates),
                value_histogram_aggregate=dict(sorted(hist.items())),
                mean_confidence=float(mean_conf),
                position_consistency_across_resets=_position_consistency(candidates),
                support_episode_count=len(candidates),
            )
        )
    output.sort(key=lambda item: (-item.support_episode_count, -item.mean_confidence, -item.position_consistency_across_resets))
    return tuple(output)


def _rank_cross_reset_candidates(
    evidence: tuple[CrossResetPOIEvidence, ...],
    hud_histograms: tuple[dict[int, float], ...] = (),
    frame_width: int = 64,
    frame_height: int = 64,
) -> tuple[POICandidate, ...]:
    out: list[POICandidate] = []
    for item in evidence:
        if not item.bbox_sequence:
            continue

        # Border filtering: Skip POIs that are too close to edges (likely HUD)
        # Only filter if POI is within 4 pixels of TOP edge (where HUD typically appears)
        bbox = item.bbox_sequence[-1]
        x0, y0, x1, y1 = bbox
        dist_to_top = y0

        if dist_to_top < 4:
            # Skip POIs in top border zone (HUD area)
            continue

        candidate = POICandidate(
            poi_id=item.canonical_poi_id,
            bbox=item.bbox_sequence[-1],
            center=item.center_sequence[-1],
            area=(item.bbox_sequence[-1][2] - item.bbox_sequence[-1][0] + 1) * (item.bbox_sequence[-1][3] - item.bbox_sequence[-1][1] + 1),
            value_histogram=item.value_histogram_aggregate,
            seen_step_indices=(),
            support_episode_indices=item.episode_indices,
            source_kind="cross_reset",
            near_avatar_steps=(),
            min_avatar_distance=999.0,
            confidence=float(min(1.0, item.mean_confidence * (0.5 + 0.5 * min(1.0, item.support_episode_count / 3.0)))),
            ambiguity_flags=(),
        )
        corr = _hud_color_correlation_score(candidate.value_histogram, hud_histograms)
        adjusted_conf = float(min(1.0, candidate.confidence + corr))
        flags = tuple(candidate.ambiguity_flags) + (f"hud_corr:{corr:.3f}",)
        out.append(
            POICandidate(
                poi_id=candidate.poi_id,
                bbox=candidate.bbox,
                center=candidate.center,
                area=candidate.area,
                value_histogram=candidate.value_histogram,
                seen_step_indices=candidate.seen_step_indices,
                support_episode_indices=candidate.support_episode_indices,
                source_kind=candidate.source_kind,
                near_avatar_steps=candidate.near_avatar_steps,
                min_avatar_distance=candidate.min_avatar_distance,
                confidence=adjusted_conf,
                ambiguity_flags=flags,
            )
        )
    out.sort(key=lambda c: (-c.confidence, -len(c.support_episode_indices), c.poi_id))
    if len(out) > 1 and abs(out[0].confidence - out[1].confidence) <= 0.06:
        out[0] = _with_flag(out[0], "close_score")
        out[1] = _with_flag(out[1], "close_score")
    return tuple(out)


def _extract_hud_candidate_histograms(
    successful_reports: list[tuple[int, POIDiscoveryReport]],
) -> tuple[dict[int, float], ...]:
    out: list[dict[int, float]] = []
    for _, report in successful_reports:
        for item in tuple(getattr(report, "contact_logs", ())):
            if not isinstance(item, dict):
                continue
            for key in ("hud_values", "hud_value_histogram", "hud_color_histogram", "hud_candidate_histogram"):
                values = item.get(key)
                if isinstance(values, dict):
                    norm = _normalize_histogram(values)
                    if norm:
                        out.append(norm)
    return tuple(out)


def _normalize_histogram(values: dict[int, int] | dict[str, int]) -> dict[int, float]:
    filtered = {int(k): max(0.0, float(v)) for k, v in dict(values).items() if int(k) != 0}
    total = sum(filtered.values())
    if total <= 0:
        return {}
    return {k: (v / total) for k, v in filtered.items()}


def _dominant_non_background_colors(values: dict[int, int] | dict[int, float]) -> tuple[int, ...]:
    items = [(int(k), float(v)) for k, v in dict(values).items() if int(k) != 0 and float(v) > 0.0]
    items.sort(key=lambda item: (-item[1], item[0]))
    return tuple(item[0] for item in items[:2])


def _weighted_hist_overlap(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    keys = set(left) | set(right)
    numer = sum(min(float(left.get(k, 0.0)), float(right.get(k, 0.0))) for k in keys)
    denom = sum(max(float(left.get(k, 0.0)), float(right.get(k, 0.0))) for k in keys)
    return float(numer / max(denom, 1e-9))


def _hud_color_correlation_score(candidate_hist: dict[int, int], hud_histograms: tuple[dict[int, float], ...]) -> float:
    if not hud_histograms:
        return 0.0
    poi_norm = _normalize_histogram(candidate_hist)
    if not poi_norm:
        return 0.0
    poi_dom = set(_dominant_non_background_colors(poi_norm))
    best = 0.0
    for hud_hist in hud_histograms:
        hud_norm = _normalize_histogram(hud_hist)
        hud_dom = set(_dominant_non_background_colors(hud_norm))
        exact_dom_overlap = 1.0 if poi_dom and hud_dom and bool(poi_dom & hud_dom) else 0.0
        weighted_overlap = _weighted_hist_overlap(poi_norm, hud_norm)
        score = (0.12 * exact_dom_overlap) + (0.08 * weighted_overlap)
        if score > best:
            best = score
    return float(best)


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _deduplicate_redundant_candidates(
    candidates: tuple[POICandidate, ...],
) -> tuple[tuple[POICandidate, ...], tuple[str, ...]]:
    ordered = list(tuple(candidates or ()))
    kept: list[POICandidate] = []
    dropped_ids: list[str] = []
    for candidate in ordered:
        replaced = False
        remove_index = None
        for index, existing in enumerate(tuple(kept)):
            c_dom = set(_dominant_non_background_colors(candidate.value_histogram))
            e_dom = set(_dominant_non_background_colors(existing.value_histogram))
            if c_dom != e_dom:
                continue
            dist = sqrt((candidate.center[0] - existing.center[0]) ** 2 + (candidate.center[1] - existing.center[1]) ** 2)
            if dist > 2.0:
                continue
            area_ratio = max(candidate.area, existing.area) / max(1, min(candidate.area, existing.area))
            if area_ratio > 1.35:
                continue
            overlaps = _bbox_iou(candidate.bbox, existing.bbox) >= 0.6 or _bbox_contains(candidate.bbox, existing.bbox) or _bbox_contains(existing.bbox, candidate.bbox)
            if not overlaps:
                continue
            stronger = candidate
            weaker = existing
            if existing.confidence > candidate.confidence:
                stronger, weaker = existing, candidate
            elif existing.confidence == candidate.confidence:
                if len(existing.support_episode_indices) > len(candidate.support_episode_indices):
                    stronger, weaker = existing, candidate
                elif len(existing.support_episode_indices) == len(candidate.support_episode_indices):
                    if str(existing.poi_id) < str(candidate.poi_id):
                        stronger, weaker = existing, candidate
            if stronger is existing:
                dropped_ids.append(str(candidate.poi_id))
                replaced = True
            else:
                dropped_ids.append(str(existing.poi_id))
                remove_index = index
            break
        if remove_index is not None:
            kept[remove_index] = candidate
            replaced = True
        if not replaced:
            kept.append(candidate)
    kept.sort(key=lambda c: (-c.confidence, -len(c.support_episode_indices), c.poi_id))
    return tuple(kept), tuple(dropped_ids)


def _poi_similarity(left: POICandidate, right: POICandidate) -> float:
    iou = _bbox_iou(left.bbox, right.bbox)
    dist = sqrt((left.center[0] - right.center[0]) ** 2 + (left.center[1] - right.center[1]) ** 2)
    center_sim = max(0.0, 1.0 - dist / 8.0)
    hist_sim = _hist_sim(left.value_histogram, right.value_histogram)
    source_sim = 1.0 if left.source_kind == right.source_kind else (0.7 if {left.source_kind, right.source_kind} <= {"changed_component", "static_inventory", "changed_component+static_inventory"} else 0.4)
    return 0.35 * iou + 0.25 * center_sim + 0.25 * hist_sim + 0.15 * source_sim


def _position_consistency(candidates: tuple[POICandidate, ...]) -> float:
    if len(candidates) <= 1:
        return 1.0
    mean_x = sum(c.center[0] for c in candidates) / len(candidates)
    mean_y = sum(c.center[1] for c in candidates) / len(candidates)
    d = [sqrt((c.center[0] - mean_x) ** 2 + (c.center[1] - mean_y) ** 2) for c in candidates]
    return max(0.0, min(1.0, 1.0 - (sum(d) / len(d)) / 10.0))


def _with_flag(candidate: POICandidate, flag: str) -> POICandidate:
    if flag in set(candidate.ambiguity_flags):
        return candidate
    return POICandidate(
        poi_id=candidate.poi_id,
        bbox=candidate.bbox,
        center=candidate.center,
        area=candidate.area,
        value_histogram=candidate.value_histogram,
        seen_step_indices=candidate.seen_step_indices,
        support_episode_indices=candidate.support_episode_indices,
        source_kind=candidate.source_kind,
        near_avatar_steps=candidate.near_avatar_steps,
        min_avatar_distance=candidate.min_avatar_distance,
        confidence=candidate.confidence,
        ambiguity_flags=tuple(candidate.ambiguity_flags) + (flag,),
    )


def _bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    ix0 = max(left[0], right[0])
    iy0 = max(left[1], right[1])
    ix1 = min(left[2], right[2])
    iy1 = min(left[3], right[3])
    if ix1 < ix0 or iy1 < iy0:
        return 0.0
    inter = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
    left_area = (left[2] - left[0] + 1) * (left[3] - left[1] + 1)
    right_area = (right[2] - right[0] + 1) * (right[3] - right[1] + 1)
    union = max(left_area + right_area - inter, 1)
    return inter / union


def _hist_sim(left: dict[int, int], right: dict[int, int]) -> float:
    if not left or not right:
        return 0.0
    keys = set(left) | set(right)
    overlap = sum(min(int(left.get(k, 0)), int(right.get(k, 0))) for k in keys)
    total = sum(max(int(left.get(k, 0)), int(right.get(k, 0))) for k in keys)
    return float(overlap / max(total, 1))


def select_top_poi_candidates_for_contact(
    poi_report: POIDiscoveryReport,
    *,
    max_pois: int = 2,
) -> tuple[POICandidate, ...]:
    ordered = tuple(
        sorted(
            tuple(poi_report.candidates),
            key=lambda item: (-item.confidence, item.poi_id),
        )
    )
    return ordered[: max(0, int(max_pois))]
