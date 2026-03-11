from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Tuple

from codex_baseline_v2.shared.config import AvatarTrackingConfigV2
from codex_baseline_v2.shared.schemas import (
    AvatarAppearanceSignatureV2,
    AvatarTrackHypothesisV2,
    ObservationSummaryV2,
    ObjectRecordV2,
    SCHEMA_VERSION,
)
from codex_baseline_v2.shared.utils import BBox, bbox_distance, bbox_iou, merge_ranges, normalize_palette


def _signature_id(candidate: ObjectRecordV2) -> str:
    return "appearance:%s:%s:%s" % (
        candidate.color,
        candidate.bbox.width(),
        candidate.bbox.height(),
    )


def _candidate_signature(candidate: ObjectRecordV2, game_id: str, existing: Optional[AvatarAppearanceSignatureV2]) -> AvatarAppearanceSignatureV2:
    width_range = merge_ranges(existing.bbox_width_range, (candidate.bbox.width(), candidate.bbox.width())) if existing else (candidate.bbox.width(), candidate.bbox.width())
    height_range = merge_ranges(existing.bbox_height_range, (candidate.bbox.height(), candidate.bbox.height())) if existing else (candidate.bbox.height(), candidate.bbox.height())
    old_variants = existing.animation_variants if existing is not None else 0
    confidence = max(existing.confidence if existing is not None else 0.0, candidate.confidence)
    return AvatarAppearanceSignatureV2(
        schema_version=SCHEMA_VERSION,
        game_id=game_id,
        signature_id=_signature_id(candidate),
        palette=normalize_palette((existing.palette if existing is not None else []) + [candidate.color]),
        bbox_width_range=width_range,
        bbox_height_range=height_range,
        aspect_ratio_mean=((existing.aspect_ratio_mean if existing is not None else 0.0) * old_variants + candidate.aspect_ratio) / float(max(1, old_variants + 1)),
        mask_area_mean=((existing.mask_area_mean if existing is not None else 0.0) * old_variants + candidate.area) / float(max(1, old_variants + 1)),
        animation_variants=old_variants + 1,
        confidence=confidence,
    )


def _prediction_error(track: AvatarTrackHypothesisV2, candidate: ObjectRecordV2) -> float:
    predicted = track.predicted_centroid
    actual = candidate.centroid
    return abs(predicted[0] - actual[0]) + abs(predicted[1] - actual[1])


def _route_consistency(track: AvatarTrackHypothesisV2, candidate: ObjectRecordV2) -> float:
    predicted = track.predicted_centroid
    bbox = candidate.bbox
    cx, cy = candidate.centroid
    if bbox.x1 <= int(round(predicted[0])) <= bbox.x2 and bbox.y1 <= int(round(predicted[1])) <= bbox.y2:
        return 1.0
    return max(0.0, 1.0 - (abs(predicted[0] - cx) + abs(predicted[1] - cy)) / 8.0)


def _appearance_similarity(track: AvatarTrackHypothesisV2, candidate: ObjectRecordV2) -> float:
    if track.appearance_signature_id is None:
        return candidate.confidence
    parts = track.appearance_signature_id.split(":")
    if len(parts) < 4:
        return 0.5
    try:
        same_color = int(parts[1]) == int(candidate.color)
    except ValueError:
        same_color = False
    same_size = ("%sx%s" % (candidate.bbox.width(), candidate.bbox.height())) == parts[2] + ":" + parts[3] if len(parts) > 3 else False
    score = 0.0
    if same_color:
        score += 0.6
    if same_size:
        score += 0.2
    return min(1.0, score + candidate.confidence * 0.2)


def _motion_overlap(summary: ObservationSummaryV2, candidate: ObjectRecordV2) -> float:
    if not summary.active_regions:
        return 0.0
    return max((bbox_iou(region, candidate.bbox) for region in summary.active_regions), default=0.0)


def _next_prediction(track: AvatarTrackHypothesisV2, centroid: Tuple[float, float], velocity: Tuple[float, float]) -> Tuple[float, float]:
    return (float(centroid[0] + velocity[0]), float(centroid[1] + velocity[1]))


def _match_score(
    track: AvatarTrackHypothesisV2,
    candidate: ObjectRecordV2,
    summary: ObservationSummaryV2,
    cfg: AvatarTrackingConfigV2,
) -> float:
    pred_term = max(0.0, 1.0 - (_prediction_error(track, candidate) / 8.0)) * cfg.prediction_weight
    app_term = _appearance_similarity(track, candidate) * cfg.appearance_weight
    motion_term = _motion_overlap(summary, candidate) * cfg.motion_weight
    route_term = _route_consistency(track, candidate) * cfg.route_consistency_weight
    iou_term = bbox_iou(track.bbox, candidate.bbox)
    return pred_term + app_term + motion_term + route_term + iou_term


def _build_track(
    game_id: str,
    candidate: ObjectRecordV2,
    summary: ObservationSummaryV2,
    track_id: str,
    support_count: int,
    missing_count: int,
    posterior: float,
    velocity: Tuple[float, float],
) -> AvatarTrackHypothesisV2:
    return AvatarTrackHypothesisV2(
        schema_version=SCHEMA_VERSION,
        game_id=game_id,
        track_id=track_id,
        status="confirmed" if posterior >= 0.7 else "candidate",
        bbox=candidate.bbox,
        centroid=candidate.centroid,
        predicted_centroid=_next_prediction(
            AvatarTrackHypothesisV2(
                schema_version=SCHEMA_VERSION,
                game_id=game_id,
                track_id=track_id,
                status="candidate",
                bbox=candidate.bbox,
                centroid=candidate.centroid,
                predicted_centroid=candidate.centroid,
                velocity=velocity,
                appearance_signature_id=_signature_id(candidate),
                shape_signature_id=None,
                motion_signature_id=None,
                posterior=posterior,
                support_count=support_count,
                missing_count=missing_count,
                last_seen_episode_id=summary.episode_id,
                last_seen_step_idx=summary.step_idx,
                evidence_refs=[],
            ),
            candidate.centroid,
            velocity,
        ),
        velocity=velocity,
        appearance_signature_id=_signature_id(candidate),
        shape_signature_id=None,
        motion_signature_id=None,
        posterior=posterior,
        support_count=support_count,
        missing_count=missing_count,
        last_seen_episode_id=summary.episode_id,
        last_seen_step_idx=summary.step_idx,
        evidence_refs=[f"{summary.episode_id}:{summary.step_idx}:{candidate.object_id}"],
    )


def update_avatar_tracks_from_observation_summaries(
    summaries: List[ObservationSummaryV2],
    cfg: AvatarTrackingConfigV2,
    existing_tracks: Optional[List[AvatarTrackHypothesisV2]] = None,
    existing_signatures: Optional[List[AvatarAppearanceSignatureV2]] = None,
) -> tuple[list[AvatarTrackHypothesisV2], list[AvatarAppearanceSignatureV2]]:
    tracks: List[AvatarTrackHypothesisV2] = sorted(list(existing_tracks or []), key=lambda row: row.posterior, reverse=True)
    signature_map: Dict[str, AvatarAppearanceSignatureV2] = {row.signature_id: row for row in (existing_signatures or [])}
    next_track_index = len(tracks)
    game_id = summaries[0].game_id if summaries else (tracks[0].game_id if tracks else "unknown_game")

    for summary in summaries:
        candidates = list(summary.avatar_candidates)
        used_tracks: set[str] = set()
        new_tracks: List[AvatarTrackHypothesisV2] = []

        for candidate in candidates:
            best_idx: Optional[int] = None
            best_score = -1.0
            for idx, track in enumerate(tracks):
                if track.track_id in used_tracks:
                    continue
                score = _match_score(track, candidate, summary, cfg)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            signature_id = _signature_id(candidate)
            signature_map[signature_id] = _candidate_signature(candidate, game_id, signature_map.get(signature_id))
            if best_idx is None or best_score < cfg.track_prune_threshold:
                track_id = "track:%s:%04d" % (summary.episode_id, next_track_index)
                next_track_index += 1
                new_tracks.append(
                    _build_track(
                        game_id=game_id,
                        candidate=candidate,
                        summary=summary,
                        track_id=track_id,
                        support_count=1,
                        missing_count=0,
                        posterior=max(0.3, candidate.confidence),
                        velocity=(0.0, 0.0),
                    )
                )
                continue
            prev = tracks[best_idx]
            used_tracks.add(prev.track_id)
            velocity = (
                float(candidate.centroid[0] - prev.centroid[0]),
                float(candidate.centroid[1] - prev.centroid[1]),
            )
            posterior = min(1.0, 0.6 * prev.posterior + 0.4 * min(1.0, best_score / max(0.1, cfg.prediction_weight + cfg.appearance_weight + cfg.motion_weight + cfg.route_consistency_weight + 1.0)))
            updated = AvatarTrackHypothesisV2(
                schema_version=SCHEMA_VERSION,
                game_id=prev.game_id,
                track_id=prev.track_id,
                status="confirmed" if posterior >= cfg.track_confirm_threshold else prev.status,
                bbox=candidate.bbox,
                centroid=candidate.centroid,
                predicted_centroid=_next_prediction(prev, candidate.centroid, velocity),
                velocity=velocity,
                appearance_signature_id=signature_id,
                shape_signature_id=prev.shape_signature_id,
                motion_signature_id=prev.motion_signature_id,
                posterior=posterior,
                support_count=prev.support_count + 1,
                missing_count=0,
                last_seen_episode_id=summary.episode_id,
                last_seen_step_idx=summary.step_idx,
                evidence_refs=(prev.evidence_refs + [f"{summary.episode_id}:{summary.step_idx}:{candidate.object_id}"])[-12:],
            )
            tracks[best_idx] = updated

        for idx, track in enumerate(tracks):
            if track.track_id in used_tracks:
                continue
            if track.last_seen_episode_id == summary.episode_id and track.last_seen_step_idx == summary.step_idx:
                continue
            missing_count = track.missing_count + 1
            posterior = track.posterior * 0.85
            tracks[idx] = replace(
                track,
                missing_count=missing_count,
                posterior=posterior,
                status="stale" if missing_count > cfg.missing_tolerance else track.status,
            )
        tracks.extend(new_tracks)
        tracks = [track for track in tracks if track.posterior >= cfg.track_prune_threshold or track.missing_count <= cfg.missing_tolerance]
        tracks.sort(key=lambda row: (row.posterior, row.support_count, -row.missing_count), reverse=True)
        tracks = tracks[: cfg.max_hypotheses]

    signatures = sorted(signature_map.values(), key=lambda row: row.signature_id)
    return tracks, signatures


def relocalize_avatar_live(
    observation_summary: Optional[ObservationSummaryV2],
    existing_tracks: List[AvatarTrackHypothesisV2],
    cfg: AvatarTrackingConfigV2,
) -> tuple[Optional[AvatarTrackHypothesisV2], list[AvatarTrackHypothesisV2]]:
    if observation_summary is None:
        ranked = sorted(existing_tracks, key=lambda row: (row.posterior, row.support_count), reverse=True)[: cfg.max_hypotheses]
        return (ranked[0] if ranked else None), ranked
    if observation_summary.avatar_track_hypotheses and not observation_summary.avatar_candidates:
        merged = list(existing_tracks) + list(observation_summary.avatar_track_hypotheses)
        deduped: Dict[str, AvatarTrackHypothesisV2] = {}
        for track in merged:
            prev = deduped.get(track.track_id)
            if prev is None or (track.posterior, track.support_count, -track.missing_count) > (prev.posterior, prev.support_count, -prev.missing_count):
                deduped[track.track_id] = track
        ranked = sorted(deduped.values(), key=lambda row: (row.posterior, row.support_count, -row.missing_count), reverse=True)[: cfg.max_hypotheses]
        return (ranked[0] if ranked else None), ranked
    merged, _ = update_avatar_tracks_from_observation_summaries(
        [observation_summary],
        cfg,
        existing_tracks=existing_tracks,
        existing_signatures=None,
    )
    ranked = sorted(merged, key=lambda row: (row.posterior, row.support_count, -row.missing_count), reverse=True)[: cfg.max_hypotheses]
    return (ranked[0] if ranked else None), ranked
