from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MatchedBranchDecision:
    selected_branch: str
    on_success: float
    off_success: float
    gain: float
    reason: str


def select_matched_branch(
    on_success: float,
    off_success: float,
    *,
    on_levels: int = 0,
    off_levels: int = 0,
    on_solved_games: int = 0,
    off_solved_games: int = 0,
    tolerance: float = 0.0025,
) -> MatchedBranchDecision:
    gain = float(on_success) - float(off_success)
    if gain > float(tolerance):
        return MatchedBranchDecision("hgt_on", on_success, off_success, gain, "macro_success")
    if gain < -float(tolerance):
        return MatchedBranchDecision("hgt_off", on_success, off_success, gain, "macro_success")
    if int(on_levels) != int(off_levels):
        branch = "hgt_on" if int(on_levels) > int(off_levels) else "hgt_off"
        return MatchedBranchDecision(branch, on_success, off_success, gain, "levels_completed")
    if int(on_solved_games) != int(off_solved_games):
        branch = "hgt_on" if int(on_solved_games) > int(off_solved_games) else "hgt_off"
        return MatchedBranchDecision(branch, on_success, off_success, gain, "solved_games")
    return MatchedBranchDecision("hgt_off", on_success, off_success, gain, "conservative_tie")


def matched_jobs(jobs: list[tuple[int, Any, int, int]]) -> tuple[list[tuple[int, Any, int, int]], list[tuple[int, Any, int, int]]]:
    """Return independent lists with identical actor/spec/budget/seed tuples."""
    return list(jobs), list(jobs)
