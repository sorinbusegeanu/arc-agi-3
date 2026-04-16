from __future__ import annotations

from math import hypot

from v5_0.avatar.scorer import aggregate_histograms
from v5_0.contracts.avatar_types import ScoredStepCandidate, TrackCandidate


def build_tracks(scored_per_step: tuple[tuple[ScoredStepCandidate, ...], ...]) -> tuple[TrackCandidate, ...]:
    nodes: list[tuple[int, ScoredStepCandidate]] = []
    node_id = 0
    for step_candidates in scored_per_step:
        for candidate in step_candidates:
            nodes.append((node_id, candidate))
            node_id += 1

    if not nodes:
        return ()

    best_path_by_node: dict[int, tuple[tuple[ScoredStepCandidate, ...], tuple[float, ...], float]] = {}
    for current_id, current in nodes:
        best_path = ((current,), (), float(current.score))
        for previous_id, previous_path in best_path_by_node.items():
            last = previous_path[0][-1]
            link_score = _link_score(last, current)
            if link_score <= 0.0:
                continue
            proposal_score = previous_path[2] + float(current.score) + link_score
            if proposal_score > best_path[2]:
                best_path = (previous_path[0] + (current,), previous_path[1] + (link_score,), proposal_score)
            elif proposal_score == best_path[2]:
                if _path_key(previous_path[0] + (current,)) < _path_key(best_path[0]):
                    best_path = (previous_path[0] + (current,), previous_path[1] + (link_score,), proposal_score)
        best_path_by_node[current_id] = best_path

    unique: dict[tuple[tuple[int, tuple[int, int, int, int]], ...], TrackCandidate] = {}
    for path_candidates, link_scores, total_score in best_path_by_node.values():
        key = tuple((item.component.step_index, item.component.bbox) for item in path_candidates)
        unique[key] = _to_track(path_candidates, link_scores, total_score)

    tracks = tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                -item.score,
                -item.support_count,
                -item.direction_agreement_score,
                item.bbox,
                item.center,
            ),
        )
    )
    return tracks


def _to_track(
    candidates: tuple[ScoredStepCandidate, ...],
    link_scores: tuple[float, ...],
    total_score: float,
) -> TrackCandidate:
    support_steps = tuple(item.component.step_index for item in candidates)
    support_actions = tuple(item.component.action for item in candidates)
    motions = tuple((float(item.component.observed_dx), float(item.component.observed_dy)) for item in candidates)
    direction = sum(item.direction_agreement_score for item in candidates) / max(len(candidates), 1)
    shape = sum(item.shape_consistency_score for item in candidates) / max(len(candidates), 1)
    continuity = sum(link_scores) / len(link_scores) if link_scores else 0.75

    representative = max(
        candidates,
        key=lambda item: (item.score, item.component.step_index, item.component.bbox),
    )
    value_pre, value_post = aggregate_histograms(candidates)

    mean_step_score = sum(item.score for item in candidates) / max(len(candidates), 1)
    support_strength = min(1.0, len(candidates) / 6.0)
    confidence = (
        0.40 * support_strength
        + 0.24 * direction
        + 0.20 * shape
        + 0.16 * continuity
    )
    confidence = max(0.0, min(1.0, confidence * (0.35 + 0.65 * support_strength)))

    return TrackCandidate(
        support_step_indices=support_steps,
        support_actions=support_actions,
        observed_motion_vectors=motions,
        bbox=representative.component.bbox,
        center=representative.component.post_center,
        value_histogram_pre=value_pre,
        value_histogram_post=value_post,
        direction_agreement_score=float(direction),
        shape_consistency_score=float(shape),
        track_consistency_score=float(continuity),
        score=float(confidence),
        support_count=len(candidates),
    )


def _link_score(previous: ScoredStepCandidate, current: ScoredStepCandidate) -> float:
    if current.component.step_index <= previous.component.step_index:
        return 0.0

    step_gap = current.component.step_index - previous.component.step_index
    gap_penalty = max(0.0, 1.0 - 0.2 * (step_gap - 1))

    continuity_distance = hypot(
        current.component.pre_center[0] - previous.component.post_center[0],
        current.component.pre_center[1] - previous.component.post_center[1],
    )
    spatial = max(0.0, 1.0 - (continuity_distance / 3.0))

    prev_vals = set(previous.component.value_histogram_post)
    curr_vals = set(current.component.value_histogram_pre)
    value_overlap = 1.0 if prev_vals == curr_vals and prev_vals else (0.6 if prev_vals & curr_vals else 0.0)

    action_coherence = 1.0 if current.component.action != "" else 0.0
    return gap_penalty * (0.50 * spatial + 0.30 * value_overlap + 0.20 * action_coherence)


def _path_key(candidates: tuple[ScoredStepCandidate, ...]) -> tuple[tuple[int, tuple[int, int, int, int]], ...]:
    return tuple((item.component.step_index, item.component.bbox) for item in candidates)
