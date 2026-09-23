from __future__ import annotations

from v9.memory.m7_strategy import M7Strategy


def relative_efficiency(strategies: tuple[M7Strategy, ...]) -> dict[int, float]:
    comparable = tuple(row for row in strategies if row.expected_cost is not None)
    if len(comparable) < 2 or len({row.target_outcome for row in comparable}) != 1:
        return {}
    best = min(float(row.expected_cost) for row in comparable if row.expected_cost is not None)
    return {row.uid.lo: best / float(row.expected_cost) for row in comparable if row.expected_cost is not None}



def strategy_frontier(strategies: tuple[M7Strategy, ...]) -> dict[int, float | None]:
    grouped: dict[tuple[int, object], list[M7Strategy]] = {}
    for strategy in strategies:
        grouped.setdefault((strategy.target_environment_id, strategy.target_outcome), []).append(strategy)
    result: dict[int, float | None] = {}
    for rows in grouped.values():
        efficiencies = relative_efficiency(tuple(rows))
        for row in rows:
            result[row.uid.lo] = efficiencies.get(row.uid.lo)
    return result
