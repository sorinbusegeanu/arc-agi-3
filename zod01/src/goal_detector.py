from __future__ import annotations

from .types import CanonicalState


def is_terminal_win(state: CanonicalState) -> bool:
    return state.state == "WIN"


def progress_score(prev: CanonicalState | None, curr: CanonicalState) -> float:
    score = 0.0
    if curr.win_levels > 0:
        score += curr.levels_completed / max(1, curr.win_levels)
    else:
        score += curr.levels_completed * 0.1
    if prev is not None and curr.levels_completed > prev.levels_completed:
        score += 1.0
    return score
