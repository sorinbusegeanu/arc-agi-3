from __future__ import annotations

from v9.cognition.strategies import relative_efficiency
from v9.memory.m7_strategy import M7Strategy


def choose_strategy(strategies: tuple[M7Strategy, ...], *, target_environment_id: int) -> M7Strategy | None:
    eligible = tuple(
        row
        for row in strategies
        if row.target_environment_id == int(target_environment_id) and row.reliability > 0
    )
    if not eligible:
        return None

    efficiencies = relative_efficiency(eligible)
    return min(
        eligible,
        key=lambda row: (
            -row.reliability,
            -(row.primary_valence_sum / max(1, row.reliability_trials)),
            -(efficiencies.get(row.uid.lo) if efficiencies.get(row.uid.lo) is not None else -1.0),
            row.uid,
        ),
    )


def replan(current: M7Strategy, alternatives: tuple[M7Strategy, ...], *, target_environment_id: int) -> M7Strategy | None:
    same_outcome = tuple(
        row
        for row in alternatives
        if row.uid != current.uid and row.target_outcome == current.target_outcome
    )
    return choose_strategy(same_outcome, target_environment_id=target_environment_id)
