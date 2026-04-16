from __future__ import annotations

from collections import Counter
from math import sqrt

from v5_0.avatar.candidate_extractor import extract_candidate_components
from v5_0.avatar.scorer import rank_tracks, score_step_candidates
from v5_0.avatar.track_builder import build_tracks
from v5_0.contracts.avatar_types import (
    AvatarDiagnostics,
    AvatarCandidate,
    AvatarIdentificationReport,
    AvatarSelectedResult,
    CrossResetAvatarEvidence,
    MultiResetAvatarReport,
    MultiResetDiagnostics,
    ProbeEpisode,
    ProbePlan,
    ProbeTransitionRecord,
)

_AMBIGUOUS_MARGIN = 0.08
_MIN_SUPPORT_STEPS = 2


def identify_avatar_candidates(
    transitions: tuple[ProbeTransitionRecord, ...],
) -> AvatarIdentificationReport:
    invalid_capture = any(
        record.invalid_action or record.pre_frame is None or record.post_frame is None
        for record in transitions
    )
    all_blocked = bool(transitions) and all(record.blocked_action for record in transitions if not record.invalid_action)

    per_step_components, dropped = extract_candidate_components(transitions)
    scored_steps = score_step_candidates(per_step_components)
    tracks = build_tracks(scored_steps)
    candidates = rank_tracks(tracks)

    per_step_counts = {
        index: len(step)
        for index, step in enumerate(scored_steps)
    }
    per_step_top_scores = {
        index: tuple(round(candidate.score, 4) for candidate in step[:3])
        for index, step in enumerate(scored_steps)
    }

    failure_reason = _determine_failure_reason(
        invalid_capture=invalid_capture,
        all_blocked=all_blocked,
        candidates=candidates,
    )
    margin = _ranking_margin(candidates)
    ambiguous = bool(candidates and len(candidates) > 1 and margin <= _AMBIGUOUS_MARGIN)
    if ambiguous and failure_reason is None:
        failure_reason = "ambiguous_avatar"

    selected = candidates[0] if candidates else None
    confidence = float(selected.score) if selected is not None else 0.0
    if selected is not None and len(selected.support_step_indices) < _MIN_SUPPORT_STEPS and failure_reason is None:
        failure_reason = "insufficient_support"

    result = AvatarSelectedResult(
        selected_candidate_id=(selected.candidate_id if failure_reason is None and selected is not None else None),
        selected_bbox=(selected.bbox if failure_reason is None and selected is not None else None),
        selected_center=(selected.center if failure_reason is None and selected is not None else None),
        confidence=confidence,
        failure_reason=failure_reason,
        ranking_margin_to_second=margin,
    )

    diagnostics = AvatarDiagnostics(
        per_step_candidate_counts=per_step_counts,
        per_step_top_scores=per_step_top_scores,
        total_candidate_count=len(candidates),
        total_track_count=len(tracks),
        dropped_candidate_reasons=dropped,
        ambiguous_ranking=ambiguous,
        no_motion=(len(candidates) == 0),
        all_blocked=all_blocked,
    )
    return AvatarIdentificationReport(candidates=candidates, selected=result, diagnostics=diagnostics)


def _determine_failure_reason(*, invalid_capture: bool, all_blocked: bool, candidates):
    if invalid_capture:
        return "invalid_probe_capture"
    if all_blocked:
        return "all_actions_blocked"
    if not candidates:
        return "no_moving_candidate"
    if len(candidates[0].support_step_indices) < _MIN_SUPPORT_STEPS:
        return "insufficient_support"
    return None


def _ranking_margin(candidates) -> float:
    if len(candidates) < 2:
        return 1.0 if candidates else 0.0
    return float(candidates[0].score - candidates[1].score)


def identify_avatar_candidates_multi_reset(
    episodes: tuple[tuple[ProbeTransitionRecord, ...], ...],
) -> MultiResetAvatarReport:
    probe_episodes: list[ProbeEpisode] = []
    failed_indices: list[int] = []
    failure_counts: Counter[str] = Counter()

    for episode_index, transitions in enumerate(episodes):
        report = identify_avatar_candidates(transitions)
        failure_reason = report.selected.failure_reason
        if failure_reason is not None:
            failure_counts[str(failure_reason)] += 1
            failed_indices.append(episode_index)
        probe_episodes.append(
            ProbeEpisode(
                episode_index=episode_index,
                seed=episode_index,
                plan=ProbePlan(game_id="unknown", level_id="unknown", action_sequence=()),
                transitions=transitions,
                report=report,
            )
        )

    successful = tuple(
        episode for episode in probe_episodes if episode.report.selected.failure_reason is None
    )
    required_support = 1 if len(probe_episodes) == 1 else 2
    if len(successful) < required_support:
        selected = AvatarSelectedResult(
            selected_candidate_id=None,
            selected_bbox=None,
            selected_center=None,
            confidence=0.0,
            failure_reason="insufficient_support",
            ranking_margin_to_second=0.0,
        )
        diagnostics = MultiResetDiagnostics(
            episode_count=len(probe_episodes),
            successful_episode_count=len(successful),
            failed_episode_count=len(probe_episodes) - len(successful),
            failure_reason_counts=dict(sorted(failure_counts.items())),
            cross_reset_ambiguous=(len(probe_episodes) > 1),
            stable_avatar_found=False,
            confidence_accumulated=0.0,
            dropped_episode_indices=tuple(sorted(failed_indices)),
        )
        return MultiResetAvatarReport(
            episodes=tuple(probe_episodes),
            cross_reset_evidence=(),
            selected=selected,
            diagnostics=diagnostics,
        )

    clusters = _match_selected_candidates_across_resets(successful)
    evidence = _build_cross_reset_evidence(clusters)
    selected, cross_ambiguous = _select_final_cross_reset_result(evidence)
    confidence_accumulated = _accumulate_confidence(evidence)
    selected_support = max(
        (
            int(item.support_episode_count)
            for item in evidence
            if item.canonical_candidate_id == selected.selected_candidate_id
        ),
        default=0,
    )
    stable = (
        selected.failure_reason is None
        and not cross_ambiguous
        and selected_support >= required_support
    )

    diagnostics = MultiResetDiagnostics(
        episode_count=len(probe_episodes),
        successful_episode_count=len(successful),
        failed_episode_count=len(probe_episodes) - len(successful),
        failure_reason_counts=dict(sorted(failure_counts.items())),
        cross_reset_ambiguous=cross_ambiguous,
        stable_avatar_found=stable,
        confidence_accumulated=confidence_accumulated,
        dropped_episode_indices=tuple(sorted(failed_indices)),
    )
    return MultiResetAvatarReport(
        episodes=tuple(probe_episodes),
        cross_reset_evidence=evidence,
        selected=selected,
        diagnostics=diagnostics,
    )


def _build_cross_reset_evidence(
    clusters: tuple[tuple[tuple[int, AvatarCandidate], ...], ...],
) -> tuple[CrossResetAvatarEvidence, ...]:
    evidence_items: list[CrossResetAvatarEvidence] = []
    for index, cluster in enumerate(clusters):
        episode_indices = tuple(item[0] for item in cluster)
        candidates = tuple(item[1] for item in cluster)
        pre_hist, post_hist = _aggregate_histograms(candidates)
        scores = [float(candidate.score) for candidate in candidates]
        mean_score = sum(scores) / max(len(scores), 1)
        variance = sum((score - mean_score) ** 2 for score in scores) / max(len(scores), 1)
        score_stddev = sqrt(variance)
        evidence_items.append(
            CrossResetAvatarEvidence(
                canonical_candidate_id=f"cross_reset_cluster_{index:03d}",
                episode_indices=episode_indices,
                per_episode_candidate_ids={episode_idx: candidate.candidate_id for episode_idx, candidate in cluster},
                bbox_sequence=tuple(candidate.bbox for candidate in candidates),
                center_sequence=tuple(candidate.center for candidate in candidates),
                value_histogram_pre_aggregate=pre_hist,
                value_histogram_post_aggregate=post_hist,
                mean_score=float(mean_score),
                score_stddev=float(score_stddev),
                shape_consistency_across_resets=_compute_shape_consistency_across_resets(candidates),
                position_consistency_across_resets=_compute_position_consistency(candidates),
                support_episode_count=len(candidates),
            )
        )
    return tuple(
        sorted(
            evidence_items,
            key=lambda item: (
                -item.support_episode_count,
                -item.mean_score,
                item.score_stddev,
                -(item.shape_consistency_across_resets + item.position_consistency_across_resets),
            ),
        )
    )


def _match_selected_candidates_across_resets(
    successful_episodes: tuple[ProbeEpisode, ...],
) -> tuple[tuple[tuple[int, AvatarCandidate], ...], ...]:
    clusters: list[list[tuple[int, AvatarCandidate]]] = []
    for episode in successful_episodes:
        selected_id = episode.report.selected.selected_candidate_id
        if selected_id is None:
            continue
        candidate = next(
            (
                item for item in episode.report.candidates
                if item.candidate_id == selected_id
            ),
            None,
        )
        if candidate is None:
            continue

        best_cluster_index: int | None = None
        best_similarity = -1.0
        for cluster_index, cluster in enumerate(clusters):
            reference = cluster[-1][1]
            similarity = _candidate_similarity(reference, candidate)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster_index = cluster_index

        if best_cluster_index is not None and best_similarity >= 0.45:
            clusters[best_cluster_index].append((episode.episode_index, candidate))
        else:
            clusters.append([(episode.episode_index, candidate)])

    normalized = [
        tuple(sorted(cluster, key=lambda item: item[0]))
        for cluster in clusters
    ]
    normalized.sort(
        key=lambda cluster: (
            -len(cluster),
            -sum(item[1].score for item in cluster),
            tuple(item[0] for item in cluster),
        )
    )
    return tuple(normalized)


def _aggregate_histograms(
    candidates: tuple[AvatarCandidate, ...],
) -> tuple[dict[int, int], dict[int, int]]:
    pre_counter: Counter[int] = Counter()
    post_counter: Counter[int] = Counter()
    for candidate in candidates:
        pre_counter.update(candidate.value_histogram_pre)
        post_counter.update(candidate.value_histogram_post)
    return dict(sorted(pre_counter.items())), dict(sorted(post_counter.items()))


def _compute_position_consistency(candidates: tuple[AvatarCandidate, ...]) -> float:
    if len(candidates) <= 1:
        return 1.0
    xs = [candidate.center[0] for candidate in candidates]
    ys = [candidate.center[1] for candidate in candidates]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    distances = [((x - mean_x) ** 2 + (y - mean_y) ** 2) ** 0.5 for x, y in zip(xs, ys)]
    mean_distance = sum(distances) / len(distances)
    return max(0.0, min(1.0, 1.0 - (mean_distance / 8.0)))


def _compute_shape_consistency_across_resets(candidates: tuple[AvatarCandidate, ...]) -> float:
    if len(candidates) <= 1:
        return 1.0
    widths = [candidate.bbox[2] - candidate.bbox[0] + 1 for candidate in candidates]
    heights = [candidate.bbox[3] - candidate.bbox[1] + 1 for candidate in candidates]
    area = [width * height for width, height in zip(widths, heights)]
    size_span = (max(area) - min(area)) / max(max(area), 1)
    return max(0.0, min(1.0, 1.0 - size_span))


def _accumulate_confidence(evidence: tuple[CrossResetAvatarEvidence, ...]) -> float:
    if not evidence:
        return 0.0
    best = evidence[0]
    max_support_seen = max(int(item.support_episode_count) for item in evidence)
    denominator = 1.0 if max_support_seen <= 1 else max(2.0, float(max_support_seen))
    support_factor = min(1.0, float(best.support_episode_count) / denominator)
    confidence = best.mean_score + (0.25 * support_factor)
    return max(0.0, min(1.0, float(confidence)))


def _select_final_cross_reset_result(
    evidence: tuple[CrossResetAvatarEvidence, ...],
) -> tuple[AvatarSelectedResult, bool]:
    if not evidence:
        return (
            AvatarSelectedResult(
                selected_candidate_id=None,
                selected_bbox=None,
                selected_center=None,
                confidence=0.0,
                failure_reason="insufficient_support",
                ranking_margin_to_second=0.0,
            ),
            True,
        )

    best = evidence[0]
    second = evidence[1] if len(evidence) > 1 else None
    margin = (
        best.mean_score - second.mean_score
        if second is not None
        else 1.0
    )
    ambiguous = False
    if second is not None:
        if (
            best.support_episode_count == second.support_episode_count
            and abs(best.mean_score - second.mean_score) <= 0.05
            and abs(best.score_stddev - second.score_stddev) <= 0.05
        ):
            ambiguous = True

    if ambiguous:
        return (
            AvatarSelectedResult(
                selected_candidate_id=None,
                selected_bbox=None,
                selected_center=None,
                confidence=0.0,
                failure_reason="ambiguous_avatar",
                ranking_margin_to_second=float(margin),
            ),
            True,
        )
    max_support_seen = max(int(item.support_episode_count) for item in evidence)
    required_support = 1 if max_support_seen <= 1 else 2
    if best.support_episode_count < required_support:
        return (
            AvatarSelectedResult(
                selected_candidate_id=None,
                selected_bbox=None,
                selected_center=None,
                confidence=0.0,
                failure_reason="insufficient_support",
                ranking_margin_to_second=float(margin),
            ),
            True,
        )

    selected_bbox = best.bbox_sequence[-1] if best.bbox_sequence else None
    selected_center = best.center_sequence[-1] if best.center_sequence else None
    return (
        AvatarSelectedResult(
            selected_candidate_id=best.canonical_candidate_id,
            selected_bbox=selected_bbox,
            selected_center=selected_center,
            confidence=_accumulate_confidence(evidence),
            failure_reason=None,
            ranking_margin_to_second=float(margin),
        ),
        False,
    )


def _candidate_similarity(left: AvatarCandidate, right: AvatarCandidate) -> float:
    iou = _bbox_iou(left.bbox, right.bbox)
    center_distance = (
        (left.center[0] - right.center[0]) ** 2 + (left.center[1] - right.center[1]) ** 2
    ) ** 0.5
    center_similarity = max(0.0, min(1.0, 1.0 - (center_distance / 10.0)))
    value_similarity = _histogram_similarity(left.value_histogram_post, right.value_histogram_post)
    shape_similarity = 1.0 - (
        abs((left.bbox[2] - left.bbox[0]) - (right.bbox[2] - right.bbox[0]))
        + abs((left.bbox[3] - left.bbox[1]) - (right.bbox[3] - right.bbox[1]))
    ) / 10.0
    shape_similarity = max(0.0, min(1.0, shape_similarity))
    if iou < 0.1 and center_distance > 3.0:
        return max(0.0, min(0.4, 0.4 * value_similarity))
    return (
        0.35 * iou
        + 0.25 * center_similarity
        + 0.20 * value_similarity
        + 0.20 * shape_similarity
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


def _histogram_similarity(left: dict[int, int], right: dict[int, int]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    overlap = sum(min(int(left.get(key, 0)), int(right.get(key, 0))) for key in keys)
    total = sum(max(int(left.get(key, 0)), int(right.get(key, 0))) for key in keys)
    if total <= 0:
        return 1.0
    return float(overlap / total)
