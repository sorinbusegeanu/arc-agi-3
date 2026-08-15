from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Protocol

HYPOTHESIS_IDS = tuple(f"H{index:02d}" for index in range(1, 16))
DEFAULT_HYPOTHESIS_DECISION = "INSUFFICIENT_EVIDENCE"
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


def game_rates(rows: Iterable[GameProgress]) -> tuple[float, float]:
    """Return game-level win and level-advance rates, invariant to actor lane count."""
    grouped: dict[str, list[GameProgress]] = defaultdict(list)
    for row in rows:
        grouped[str(row.game_id)].append(row)
    if not grouped:
        return 0.0, 0.0
    games = len(grouped)
    won = sum(any(int(row.wins) > 0 for row in lane_rows) for lane_rows in grouped.values())
    advanced = sum(
        any(int(row.levels_completed) > 0 for row in lane_rows)
        for lane_rows in grouped.values()
    )
    return 100.0 * won / games, 100.0 * advanced / games


def format_game_rate_line(rows: Iterable[GameProgress]) -> str:
    win_rate, level_rate = game_rates(rows)
    return f"win_rate={win_rate:.1f}% level_rate={level_rate:.1f}%"
