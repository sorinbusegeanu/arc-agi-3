from __future__ import annotations

from math import hypot

from v5_0.avatar.scorer import aggregate_histograms
from v5_0.contracts.avatar_types import ScoredStepCandidate, TrackCandidate

_ACTION_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}


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
    entry_bbox = _infer_entry_bbox(candidates[0])
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
        entry_bbox=entry_bbox,
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


def _infer_entry_bbox(candidate: ScoredStepCandidate) -> tuple[int, int, int, int]:
    component = candidate.component
    bbox = tuple(component.bbox)
    if (
        abs(float(component.observed_dx)) < 0.5
        and abs(float(component.observed_dy)) < 0.5
        and (
            bool(component.blocked_action)
            or (
                abs(float(component.pre_center[0]) - float(component.post_center[0])) < 0.5
                and abs(float(component.pre_center[1]) - float(component.post_center[1])) < 0.5
            )
        )
    ):
        return bbox
    if component.pre_non_background_cells and not component.post_non_background_cells:
        return bbox
    if not component.pre_non_background_cells and component.post_non_background_cells:
        if _is_edge_clamped_post_only_component(component, bbox):
            return bbox
    dx_sign, dy_sign = _ACTION_DELTAS.get(str(component.action), (0, 0))
    width = max(1, int(bbox[2]) - int(bbox[0]) + 1)
    height = max(1, int(bbox[3]) - int(bbox[1]) + 1)
    shift_x = int(round(abs(float(component.observed_dx)))) or (width if dx_sign != 0 else 0)
    shift_y = int(round(abs(float(component.observed_dy)))) or (height if dy_sign != 0 else 0)
    return (
        int(bbox[0]) - int(dx_sign * shift_x),
        int(bbox[1]) - int(dy_sign * shift_y),
        int(bbox[2]) - int(dx_sign * shift_x),
        int(bbox[3]) - int(dy_sign * shift_y),
    )


def _is_edge_clamped_post_only_component(
    component: ScoredStepCandidate.component.__class__,
    bbox: tuple[int, int, int, int],
) -> bool:
    action = str(component.action)
    frame_width = max(0, int(getattr(component, "frame_width", 0) or 0))
    frame_height = max(0, int(getattr(component, "frame_height", 0) or 0))
    if action == "UP":
        return int(bbox[1]) <= 0
    if action == "LEFT":
        return int(bbox[0]) <= 0
    if action == "DOWN" and frame_height > 0:
        return int(bbox[3]) >= frame_height - 1
    if action == "RIGHT" and frame_width > 0:
        return int(bbox[2]) >= frame_width - 1
    return False


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
