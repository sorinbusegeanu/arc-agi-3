from __future__ import annotations

from collections import Counter
from math import hypot

from v5_0.contracts.avatar_types import AvatarCandidate, ScoredStepCandidate, TrackCandidate

_ACTION_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}


def score_step_candidates(
    per_step_components,
) -> tuple[tuple[ScoredStepCandidate, ...], ...]:
    scored_steps: list[tuple[ScoredStepCandidate, ...]] = []
    for components in per_step_components:
        scored: list[ScoredStepCandidate] = []
        for component in components:
            expected_dx, expected_dy = _ACTION_DELTAS.get(component.action, (0, 0))
            direction_score = 0.5 if component.blocked_action else _direction_agreement(component.observed_dx, component.observed_dy, expected_dx, expected_dy)
            magnitude = hypot(component.observed_dx, component.observed_dy)
            movement_consistency = 0.5 if component.blocked_action else max(0.0, 1.0 - abs(magnitude - 1.0))
            shape_consistency = _shape_consistency(component.pre_non_background_cells, component.post_non_background_cells)
            compactness = _compactness_score(component.bbox, component.area)
            score = (
                0.36 * direction_score
                + 0.24 * movement_consistency
                + 0.24 * shape_consistency
                + 0.16 * compactness
            )
            scored.append(
                ScoredStepCandidate(
                    component=component,
                    score=float(score),
                    direction_agreement_score=float(direction_score),
                    movement_consistency_score=float(movement_consistency),
                    shape_consistency_score=float(shape_consistency),
                    compactness_score=float(compactness),
                )
            )
        scored.sort(
            key=lambda item: (
                -item.score,
                item.component.step_index,
                item.component.bbox,
                item.component.observed_dx,
                item.component.observed_dy,
            )
        )
        scored_steps.append(tuple(scored))
    return tuple(scored_steps)


def rank_tracks(tracks: tuple[TrackCandidate, ...]) -> tuple[AvatarCandidate, ...]:
    ranked: list[AvatarCandidate] = []
    for index, track in enumerate(
        sorted(
            tracks,
            key=lambda item: (
                -item.score,
                -item.support_count,
                -item.direction_agreement_score,
                item.bbox,
                item.center,
            ),
        )
    ):
        failure_flags: list[str] = []
        if track.support_count < 2:
            failure_flags.append("low_support")
        if track.direction_agreement_score < 0.5:
            failure_flags.append("weak_direction")
        if track.track_consistency_score < 0.4:
            failure_flags.append("weak_track_consistency")

        ranked.append(
            AvatarCandidate(
                candidate_id=f"candidate_{index:03d}",
                bbox=track.bbox,
                center=track.center,
                score=float(track.score),
                support_step_indices=track.support_step_indices,
                support_actions=track.support_actions,
                observed_motion_vectors=track.observed_motion_vectors,
                direction_agreement_score=float(track.direction_agreement_score),
                shape_consistency_score=float(track.shape_consistency_score),
                track_consistency_score=float(track.track_consistency_score),
                value_histogram_pre=dict(sorted(track.value_histogram_pre.items())),
                value_histogram_post=dict(sorted(track.value_histogram_post.items())),
                failure_flags=tuple(failure_flags),
            )
        )
    return tuple(ranked)


def aggregate_histograms(candidates: tuple[ScoredStepCandidate, ...]) -> tuple[dict[int, int], dict[int, int]]:
    pre_counter: Counter[int] = Counter()
    post_counter: Counter[int] = Counter()
    for candidate in candidates:
        pre_counter.update(candidate.component.value_histogram_pre)
        post_counter.update(candidate.component.value_histogram_post)
    return dict(sorted(pre_counter.items())), dict(sorted(post_counter.items()))


def _direction_agreement(observed_dx: float, observed_dy: float, expected_dx: int, expected_dy: int) -> float:
    magnitude = hypot(observed_dx, observed_dy)
    if magnitude <= 0.0:
        return 0.0
    expected_mag = hypot(float(expected_dx), float(expected_dy))
    if expected_mag <= 0.0:
        return 0.0
    dot = (observed_dx * expected_dx) + (observed_dy * expected_dy)
    cosine = dot / (magnitude * expected_mag)
    return max(0.0, min(1.0, cosine))


def _shape_consistency(
    pre_cells: tuple[tuple[int, int], ...],
    post_cells: tuple[tuple[int, int], ...],
) -> float:
    if not pre_cells or not post_cells:
        return 0.0
    left = _normalize_shape(pre_cells)
    right = _normalize_shape(post_cells)
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _normalize_shape(cells: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    return {(x - min_x, y - min_y) for x, y in cells}


def _compactness_score(bbox: tuple[int, int, int, int], area: int) -> float:
    width = max(1, bbox[2] - bbox[0] + 1)
    height = max(1, bbox[3] - bbox[1] + 1)
    box_area = width * height
    return max(0.0, min(1.0, float(area) / float(max(box_area, 1))))
