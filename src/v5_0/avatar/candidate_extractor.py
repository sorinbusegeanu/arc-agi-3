from __future__ import annotations

from collections import Counter
from typing import Iterable

from v5_0.contracts.avatar_types import CandidateComponent, ProbeTransitionRecord


def extract_candidate_components(
    transitions: tuple[ProbeTransitionRecord, ...],
) -> tuple[tuple[CandidateComponent, ...], dict[str, int]]:
    per_step: list[tuple[CandidateComponent, ...]] = []
    dropped_reasons: Counter[str] = Counter()
    for transition in transitions:
        if transition.invalid_action:
            dropped_reasons["invalid_action"] += 1
            per_step.append(())
            continue
        if transition.pre_frame is None or transition.post_frame is None:
            dropped_reasons["missing_frame"] += 1
            per_step.append(())
            continue
        if not transition.pre_frame or not transition.post_frame:
            dropped_reasons["empty_frame"] += 1
            per_step.append(())
            continue

        changed = _changed_cells(transition.pre_frame, transition.post_frame)
        if not changed:
            dropped_reasons["no_changed_cells"] += 1
            per_step.append(())
            continue

        pre_bg = _dominant_value(transition.pre_frame)
        post_bg = _dominant_value(transition.post_frame)
        components: list[CandidateComponent] = []
        for component_cells in _connected_components(changed):
            cells_sorted = tuple(sorted(component_cells))
            pre_hist = Counter(int(transition.pre_frame[y][x]) for x, y in cells_sorted)
            post_hist = Counter(int(transition.post_frame[y][x]) for x, y in cells_sorted)
            pre_non_bg = tuple((x, y) for x, y in cells_sorted if int(transition.pre_frame[y][x]) != pre_bg)
            post_non_bg = tuple((x, y) for x, y in cells_sorted if int(transition.post_frame[y][x]) != post_bg)
            center_pre = _center(pre_non_bg or cells_sorted)
            center_post = _center(post_non_bg or cells_sorted)
            components.append(
                CandidateComponent(
                    step_index=transition.step_index,
                    action=transition.action,
                    blocked_action=transition.blocked_action,
                    bbox=_bbox(cells_sorted),
                    area=len(cells_sorted),
                    pre_center=center_pre,
                    post_center=center_post,
                    observed_dx=center_post[0] - center_pre[0],
                    observed_dy=center_post[1] - center_pre[1],
                    pre_non_background_cells=tuple(sorted(pre_non_bg)),
                    post_non_background_cells=tuple(sorted(post_non_bg)),
                    value_histogram_pre=dict(sorted(pre_hist.items())),
                    value_histogram_post=dict(sorted(post_hist.items())),
                )
            )
        components = sorted(
            components,
            key=lambda item: (item.step_index, item.bbox, item.area, item.observed_dx, item.observed_dy),
        )
        per_step.append(tuple(components))
    return tuple(per_step), dict(sorted(dropped_reasons.items()))


def _changed_cells(
    pre_frame: tuple[tuple[int, ...], ...],
    post_frame: tuple[tuple[int, ...], ...],
) -> set[tuple[int, int]]:
    height = min(len(pre_frame), len(post_frame))
    if height <= 0:
        return set()
    width = min(len(pre_frame[0]), len(post_frame[0])) if pre_frame[0] and post_frame[0] else 0
    changed: set[tuple[int, int]] = set()
    for y in range(height):
        row_pre = pre_frame[y]
        row_post = post_frame[y]
        for x in range(min(width, len(row_pre), len(row_post))):
            if int(row_pre[x]) != int(row_post[x]):
                changed.add((x, y))
    return changed


def _dominant_value(frame: tuple[tuple[int, ...], ...]) -> int:
    values = [int(value) for row in frame for value in row]
    if not values:
        return 0
    return int(Counter(values).most_common(1)[0][0])


def _connected_components(cells: Iterable[tuple[int, int]]) -> tuple[set[tuple[int, int]], ...]:
    remaining = set(cells)
    components: list[set[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = {start}
        while stack:
            x, y = stack.pop()
            for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    components.sort(key=lambda comp: (len(comp), _bbox(tuple(comp))), reverse=True)
    return tuple(components)


def _bbox(cells: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return (min(xs), min(ys), max(xs), max(ys))


def _center(cells: tuple[tuple[int, int], ...]) -> tuple[float, float]:
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return (sum(xs) / len(xs), sum(ys) / len(ys))
