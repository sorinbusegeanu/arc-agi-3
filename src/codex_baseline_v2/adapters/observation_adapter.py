from __future__ import annotations

from typing import Any, Dict, List, Optional


def adapt_observation(step: Dict[str, Any]) -> Optional[List[List[int]]]:
    grid_stack = step.get("grid_stack_t")
    if isinstance(grid_stack, list) and grid_stack:
        frame = grid_stack[-1]
        if isinstance(frame, list):
            return [[int(v) for v in row] for row in frame]
    grid = step.get("grid")
    if isinstance(grid, list):
        return [[int(v) for v in row] for row in grid]
    return None
