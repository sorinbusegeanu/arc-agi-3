from __future__ import annotations

from typing import Any


def build_game_leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -int(row.get("best_levels_solved", 0) or 0),
            int(row.get("best_solved_levels_total_steps") or 10**12),
            int(row.get("best_total_steps_for_best_solved") or 10**12),
            str(row.get("game_id", "")),
        ),
    )


def build_level_leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("best_steps_executed", 10**12) or 10**12),
            str(row.get("game_id", "")),
            int(row.get("level_index", 0) or 0),
        ),
    )
