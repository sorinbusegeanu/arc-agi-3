from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def compute_proxies(
    fp_report: Dict[str, Any],
    *,
    min_target_color_rarity: float,
    initial_target_counts: Optional[Dict[int, int]] = None,
    initial_component_count: Optional[int] = None,
) -> Dict[str, Any]:
    proxies: Dict[str, Any] = {
        "target_depletion_ratio": 0.0,
        "filled_area_ratio": 0.0,
        "stability_ratio": 0.0,
        "uniformity_score": 0.0,
        "symmetry_score": 0.0,
        "component_consolidation": 0.0,
        "component_count": 0,
        "target_counts": {},
    }

    state = fp_report.get("state_summary") or {}
    grids = state.get("grid_summaries") or []
    if not grids:
        return proxies
    grid = grids[0]
    height = grid.get("height")
    width = grid.get("width")
    if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
        return proxies
    total_cells = height * width

    bg_candidates = grid.get("bg_candidates") or []
    bg_color = None
    if bg_candidates and isinstance(bg_candidates[0], (list, tuple)):
        bg_color = bg_candidates[0][0]

    color_hist = grid.get("color_histogram") or {}
    if not isinstance(color_hist, dict):
        return proxies

    non_bg_counts = {
        int(color): int(count)
        for color, count in color_hist.items()
        if (bg_color is None or int(color) != int(bg_color))
    }
    non_bg_total = sum(non_bg_counts.values())

    if non_bg_total > 0:
        max_non_bg = max(non_bg_counts.values())
        proxies["filled_area_ratio"] = max_non_bg / non_bg_total

    max_color = max(color_hist.values()) if color_hist else 0
    proxies["uniformity_score"] = max_color / total_cells if total_cells > 0 else 0.0

    symmetry_candidates = grid.get("symmetry_candidates") or {}
    if isinstance(symmetry_candidates, dict) and symmetry_candidates:
        proxies["symmetry_score"] = max(float(v) for v in symmetry_candidates.values())

    diff = fp_report.get("diff_summary") or {}
    changed_cells = diff.get("changed_cells_count")
    if isinstance(changed_cells, int) and total_cells > 0:
        proxies["stability_ratio"] = max(0.0, 1.0 - (changed_cells / total_cells))

    target_colors = _infer_target_colors(non_bg_counts, total_cells, min_target_color_rarity)
    target_counts = {color: non_bg_counts.get(color, 0) for color in target_colors}
    proxies["target_counts"] = target_counts

    if initial_target_counts:
        initial_total = sum(initial_target_counts.values())
        current_total = sum(target_counts.values())
        if initial_total > 0:
            proxies["target_depletion_ratio"] = max(0.0, min(1.0, (initial_total - current_total) / initial_total))

    component_count = _component_count(grid, target_colors)
    proxies["component_count"] = component_count
    if initial_component_count and initial_component_count > 0:
        proxies["component_consolidation"] = max(
            0.0, min(1.0, (initial_component_count - component_count) / initial_component_count)
        )

    return proxies


def _infer_target_colors(
    non_bg_counts: Dict[int, int],
    total_cells: int,
    min_target_color_rarity: float,
) -> List[int]:
    targets: List[int] = []
    for color, count in non_bg_counts.items():
        rarity = count / total_cells if total_cells > 0 else 1.0
        if rarity < min_target_color_rarity:
            targets.append(int(color))
    return sorted(targets)


def _component_count(grid: Dict[str, Any], target_colors: List[int]) -> int:
    comps = grid.get("connected_components") or []
    if not isinstance(comps, list):
        return 0
    if not target_colors:
        return len(comps)
    count = 0
    for comp in comps:
        color = comp.get("color") if isinstance(comp, dict) else None
        if color is None:
            continue
        if int(color) in target_colors:
            count += 1
    return count
