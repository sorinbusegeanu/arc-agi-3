from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Protocol

HYPOTHESIS_IDS = tuple(f"H{index:02d}" for index in range(1, 16))
DEFAULT_HYPOTHESIS_DECISION = "INSUFFICIENT_EVIDENCE"
_DEFAULT_LEVELS_PER_GAME = 5
_VALID_DECISIONS = {
    "VALID",
    "PARTIALLY_VALID",
    "INVALID",
    "INSUFFICIENT_EVIDENCE",
}


class GameProgress(Protocol):
    game_id: str
    wins: int
    levels_completed: int
    steps: int
    first_win_step: int


def hypothesis_statuses(
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return explicit H01-H15 states without treating structural proxies as validation."""
    statuses = {
        hypothesis_id: DEFAULT_HYPOTHESIS_DECISION
        for hypothesis_id in HYPOTHESIS_IDS
    }
    if overrides is None:
        return statuses
    for hypothesis_id, decision in overrides.items():
        if hypothesis_id not in statuses:
            raise ValueError(f"unknown hypothesis id: {hypothesis_id}")
        value = str(decision)
        if value not in _VALID_DECISIONS:
            raise ValueError(f"invalid hypothesis decision: {value}")
        statuses[hypothesis_id] = value
    return statuses


def format_hypothesis_line(
    statuses: Mapping[str, str] | None = None,
) -> str:
    resolved = hypothesis_statuses(statuses)
    return "hypotheses " + " ".join(
        f"{hypothesis_id}={resolved[hypothesis_id]}"
        for hypothesis_id in HYPOTHESIS_IDS
    )


def _group_games(rows: Iterable[GameProgress]) -> dict[str, list[GameProgress]]:
    grouped: dict[str, list[GameProgress]] = defaultdict(list)
    for row in rows:
        grouped[str(row.game_id)].append(row)
    return grouped


def solved_game_ids(rows: Iterable[GameProgress]) -> tuple[str, ...]:
    """Return unique solved game IDs once each, independent of actor lane count."""
    grouped = _group_games(rows)
    return tuple(
        sorted(
            game_id
            for game_id, lane_rows in grouped.items()
            if any(int(row.wins) > 0 for row in lane_rows)
        )
    )


def solved_game_steps(rows: Iterable[GameProgress]) -> tuple[tuple[str, int], ...]:
    """Return each solved game with the earliest observed exact first-win actor step."""
    grouped = _group_games(rows)
    result: list[tuple[str, int]] = []
    for game_id, lane_rows in grouped.items():
        candidates = []
        for row in lane_rows:
            if int(row.wins) <= 0:
                continue
            first = int(getattr(row, "first_win_step", 0) or 0)
            if first <= 0:
                first = int(getattr(row, "steps", 0) or 0)
            if first > 0:
                candidates.append(first)
        if candidates:
            result.append((game_id, min(candidates)))
    return tuple(sorted(result))


def game_summary(
    rows: Iterable[GameProgress],
    *,
    levels_per_game: int = _DEFAULT_LEVELS_PER_GAME,
) -> tuple[float, float, int, int]:
    """Return distinct-game current-run win rate and partial level-completion rate.

    These values intentionally describe only observations since the current process
    started; they are not a retained-competence estimate from restored memory.
    Multiple actor lanes for the same game cannot inflate progress: the maximum
    completed-level count observed for that game is used. When no game-specific
    level count is available, five levels per game is the declared denominator.
    """
    grouped = _group_games(rows)
    if not grouped:
        return 0.0, 0.0, 0, 0
    games = len(grouped)
    won = sum(any(int(row.wins) > 0 for row in lane_rows) for lane_rows in grouped.values())
    denominator_per_game = max(1, int(levels_per_game))
    solved_levels = sum(
        min(
            denominator_per_game,
            max((max(0, int(row.levels_completed)) for row in lane_rows), default=0),
        )
        for lane_rows in grouped.values()
    )
    total_levels = games * denominator_per_game
    return 100.0 * won / games, 100.0 * solved_levels / total_levels, int(won), int(games)


def game_rates(rows: Iterable[GameProgress]) -> tuple[float, float]:
    """Return distinct-game current-run win and partial level-completion rates."""
    win_rate, level_rate, _solved_games, _games = game_summary(rows)
    return win_rate, level_rate


def format_game_rate_line(rows: Iterable[GameProgress]) -> str:
    """Format explicitly run-local progress; restored competence must be measured separately."""
    rows = tuple(rows)
    win_rate, level_rate, solved_games, games = game_summary(rows)
    solved = solved_game_steps(rows)
    suffix = "" if not solved else " (" + ", ".join(f"{game_id}:{steps}" for game_id, steps in solved) + ")"
    return (
        f"current_run_wins={win_rate:.1f}% current_run_levels_solved={level_rate:.1f}% "
        f"current_run_solved_games={solved_games}/{games}{suffix}"
    )
