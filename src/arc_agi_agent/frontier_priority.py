from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .coord_selectors import CoordCandidate
from .full_explorer_config import FullExplorerConfig


@dataclass(frozen=True)
class CoordActionCandidate:
    action_id: str
    x: int
    y: int
    selector: str
    score: int


def score_candidates(
    action_id: str,
    candidates: List[CoordCandidate],
    selector_scores: Dict[str, int],
    global_noop_counts: Dict[Tuple[str, int, int], int],
    state_noop_counts: Dict[Tuple[str, int, int], int],
    action_coord_tried: Dict[str, set[Tuple[int, int]]],
    action_attempts: Dict[str, int],
    bg_penalties: Dict[Tuple[int, int], bool],
    cfg: FullExplorerConfig,
) -> List[CoordActionCandidate]:
    scored: List[CoordActionCandidate] = []
    min_attempts = min(action_attempts.values()) if action_attempts else 0
    for cand in candidates:
        score = selector_scores.get(cand.selector, 0)
        key = (action_id, cand.x, cand.y)
        if state_noop_counts.get(key, 0) >= cfg.ban_noop_after:
            score -= 2
        if bg_penalties.get((cand.x, cand.y), False):
            score -= 1
        if (cand.x, cand.y) not in action_coord_tried.get(action_id, set()):
            score += 2
        if action_attempts.get(action_id, 0) <= min_attempts:
            score += 1
        scored.append(
            CoordActionCandidate(
                action_id=action_id,
                x=cand.x,
                y=cand.y,
                selector=cand.selector,
                score=score,
            )
        )
    scored.sort(key=lambda c: (-c.score, c.action_id, c.y, c.x))
    return scored


def selector_base_scores(cfg: FullExplorerConfig) -> Dict[str, int]:
    return {
        "changed_bbox_focus": 3,
        "adjacent_boundary_cells": 2,
        "region_frontier_cells": 2,
        "object_centroids": 2,
        "object_bbox_corners": 2,
        "grid_edges_midpoints": 1,
        "grid_corners": 1,
        "color_hotspots": 1,
    }
