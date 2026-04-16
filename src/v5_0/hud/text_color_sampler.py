from __future__ import annotations

from collections import Counter

from v5_0.contracts.avatar_types import HUDCellSample, HUDMask, HUDRegion, ProbeTransitionRecord


def sample_hud_cell_values(
    hud_mask: HUDMask,
    hud_regions: tuple[HUDRegion, ...],
    transitions: tuple[ProbeTransitionRecord, ...],
    episode_index: int,
) -> tuple[HUDCellSample, ...]:
    if hud_mask.true_cell_count <= 0:
        return ()

    active_cells = _active_cells_from_regions(hud_regions, hud_mask.width, hud_mask.height)
    out: list[HUDCellSample] = []
    for record in transitions:
        if record.post_frame is None:
            continue
        frame = record.post_frame
        for x, y in sorted(active_cells, key=lambda cell: (cell[1], cell[0])):
            if y >= len(frame) or x >= len(frame[y]):
                continue
            out.append(
                HUDCellSample(
                    row=int(y),
                    col=int(x),
                    value=int(frame[y][x]),
                    episode_index=int(episode_index),
                    step_index=int(record.step_index),
                )
            )
    return tuple(out)


def extract_hud_color_summaries(
    hud_samples: tuple[HUDCellSample, ...],
    hud_regions: tuple[HUDRegion, ...],
) -> dict[str, dict[int, int]]:
    summaries: dict[str, dict[int, int]] = {}
    if not hud_regions:
        return summaries

    for region in hud_regions:
        x0, y0, x1, y1 = region.bbox
        counter = Counter()
        for sample in hud_samples:
            if x0 <= sample.col <= x1 and y0 <= sample.row <= y1:
                counter[int(sample.value)] += 1
        summaries[region.hud_region_id] = dict(sorted(counter.items()))
    return summaries


def _active_cells_from_regions(
    regions: tuple[HUDRegion, ...],
    width: int,
    height: int,
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for region in regions:
        x0, y0, x1, y1 = region.bbox
        for y in range(max(0, y0), min(height, y1 + 1)):
            for x in range(max(0, x0), min(width, x1 + 1)):
                cells.add((x, y))
    return cells
